# -*- coding: utf-8 -*-
"""label-target-predict convert utility in openpose task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import copy
import math
import cv2
import mxnet as mx


__all__ = ['Encoder', 'Decoder']


# for openpose, add extra joint to coco-17 points

coco_num_limb = 17
coco_limb = list(
    zip([17, 0, 1, 0, 2, 17, 5, 7, 17, 6,  8, 17, 11, 13, 17, 12, 14],
        [ 0, 1, 3, 2, 4,  5, 7, 9,  6, 8, 10, 11, 13, 15, 12, 14, 16])
)


# Encoder

def add_extra_joint(keypoints):
    """
    add one extra joint to coco-17
    [N, 51] => [N, 54]

    """
    keypoints = keypoints.reshape(-1, 17, 3)
    neck_x = (keypoints[:, 5, 0] + keypoints[:, 6, 0]) / 2
    neck_y = (keypoints[:, 5, 1] + keypoints[:, 6, 1]) / 2
    neck_vis = np.minimum(keypoints[:, 5, 2], keypoints[:, 6, 2])
    neck = np.concatenate((neck_x[:, np.newaxis], 
                           neck_y[:, np.newaxis], 
                           neck_vis[:, np.newaxis]), axis=1)
    keypoints = np.concatenate((keypoints, neck[:, np.newaxis, :]), axis=1)
    return keypoints.reshape(-1, 18*3)


def generate_gauss(pos_xy, sigma, feat_size):
    """
    generate gauss heatmap

    Parameters
    ----------
    pos_xy : tuple, [x,y]
        position x and y
    sigma: float
        guassian sigma
    feat_size: tuple, [h, w]
        feature size

    """
    assert 0 <= pos_xy[0] < feat_size[1]
    assert 0 <= pos_xy[1] < feat_size[0]
    heatmap = np.zeros(feat_size, dtype=np.float32)

    tmp_size = sigma * 3
    ul = [int(pos_xy[0] - tmp_size), int(pos_xy[1] - tmp_size)]
    br = [int(pos_xy[0] + tmp_size + 1), int(pos_xy[1] + tmp_size + 1)]
    
    # generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # the gaussian is not normalized, we want the center value to be equal to 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
    
    gauss_box = [ul[0], ul[1], br[0], br[1]]
    heatmap_box = [0, 0, feat_size[1], feat_size[0]]
    
    # usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], feat_size[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], feat_size[0]) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], feat_size[1])
    img_y = max(0, ul[1]), min(br[1], feat_size[0])
    
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmap


def get_heatmap(keypoints, input_size, target_size, keep_invis, sigma):
    """compute vectormap

    Parameters
    -----------
    keypoints : array, [N, 51]
        keypoints annotation, in coco-format
    input_size : tuple, [height, width]
        input image size
    target_size : tuple, [height, width]
        target feature size
    keep_invis: bool
        if train on occlupied point
    sigma: float
        gaussian sigma

    Returns
    --------
    heatmap: array, [num_keypoint, h, w]

    heatmap_weight: array, [num_keypoint, h, w]

    """

    stride = input_size[0] // target_size[0]
    num_instance = keypoints.shape[0]
    num_keypoint = keypoints.shape[1] // 3
    keypoints = keypoints.reshape(num_instance, num_keypoint, 3)
    keypoints[:,:,0:2] /= stride

    heatmap = np.zeros((num_keypoint, target_size[0], target_size[1]), dtype=np.float32)
    heatmap_weight = np.zeros((num_keypoint, target_size[0], target_size[1]), dtype=np.float32)

    for kps_ind in range(num_keypoint):
        for inst_ind in range(num_instance):
            each_pos = keypoints[inst_ind, kps_ind]
            if keep_invis:
                if each_pos[-1] == 0 :
                    continue
            else:
                if each_pos[-1] != 2:
                    continue
            each_heatmap = generate_gauss(each_pos[:2], sigma, target_size)
            heatmap[kps_ind] = np.maximum(heatmap[kps_ind], each_heatmap)
            heatmap_weight[kps_ind, :, :] = 1.0

    return heatmap, heatmap_weight


def compute_vectormap(v_start, v_end, vectormap, countmap, limb_ind):
    """compute each vectormap

    Parameters
    -----------
    v_start: array, [x, y, vis]
        limb start point
    v_end: array, [x, y, vis]
        limb end point

    Returns
    --------
    vectormap: array
        filled in limb_ind

    """
    _, height, width = vectormap.shape[:3]

    # compute unit vector
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap
    norm_x = vector_x / length
    norm_y = vector_y / length

    # set vectormap region
    threshold = 8
    min_x = max(int(min(v_start[0], v_end[0]) - threshold), 0)
    min_y = max(int(min(v_start[1], v_end[1]) - threshold), 0)
    max_x = min(int(max(v_start[0], v_end[0]) + threshold), width)
    max_y = min(int(max(v_start[1], v_end[1]) + threshold), height)
    
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - v_start[0]
            bec_y = y - v_start[1]
            dist = abs(bec_x * norm_y - bec_y * norm_x)
            # orthogonal distance is < then threshold
            if dist > threshold:
                continue
            vectormap[limb_ind * 2 + 0][y][x] = norm_x
            vectormap[limb_ind * 2 + 1][y][x] = norm_y
            countmap[limb_ind][y][x] += 1
    return vectormap


def get_vectormap(keypoints, input_size, target_size, keep_invis, limb=None, num_limb=None):
    """compute vectormap

    Parameters
    -----------
    keypoints : array, [N, 51]
        keypoints annotation, in coco-format
    input_size : tuple, [height, width]
        input image size
    target_size : tuple, [height, width]
        target feature size

    Returns
    --------
    vectormap: array, [num_limb*2, h, w]

    vectormap_weight: array, [num_limb*2, h, w]

    """

    if limb == None:
        limb = coco_limb
    if num_limb == None:
        num_limb = coco_num_limb
    
    # [num_limb*2, H, W]
    vectormap = np.zeros((num_limb * 2, input_size[0], input_size[1]), dtype=np.float32)
    vectormap_weight = np.zeros((num_limb * 2, input_size[0], input_size[1]), dtype=np.float32)
    # [num_limb, H, W]
    counter = np.zeros((num_limb, input_size[0], input_size[1]), dtype=np.int16)
    
    for kps in keypoints:
        for limb_ind, (a, b) in enumerate(limb):
            v_start = kps[a*3 : a*3+3]
            v_end = kps[b*3 : b*3+3]
            if keep_invis:
                if v_start[-1] == 0 or v_end[-1] == 0:
                    continue
            else:
                if v_start[-1] != 2 or v_end[-1] != 2:
                    continue
            vectormap = compute_vectormap(v_start, v_end, vectormap, counter, limb_ind)
            vectormap_weight[limb_ind*2 : limb_ind*2+2, :, :] = 1.0
    vectormap = vectormap.transpose((1, 2, 0))  # [H, W, num_limb*2]

    # normalize the PAF by counter
    nonzero_vector = np.nonzero(counter)
    for i, y, x in zip(nonzero_vector[0], nonzero_vector[1], nonzero_vector[2]):
        if counter[i][y][x] <= 0:
            continue
        vectormap[y][x][i * 2 + 0] /= counter[i][y][x]
        vectormap[y][x][i * 2 + 1] /= counter[i][y][x]
    # normalize vectormap_weight
    # TBD
    print('vectormap_weight to be done')
    
    mapholder = []
    for i in range(vectormap.shape[2]):
        a = cv2.resize(vectormap[:, :, i], (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        mapholder.append(a)
    vectormap = np.array(mapholder)
    return vectormap.astype(np.float16), vectormap_weight


class Encoder(object):
    """Label to target transform for openpose task

    Parameters
    ----------
    input_size : tuple, [h, w]
        network input size, such as (540, 960). 
        If batch>1, set input_size here.
    target_size: tuple, [h, w]
        network output size, such as (540, 960)
        If batch>1, set target_size here.

    """
    def __init__(self, sigma, input_size=None, target_size=None, keep_invis=False):
        self.sigma = sigma
        self.input_size = input_size
        self.target_size = target_size
        self.keep_invis = keep_invis

    def __call__(self, keypoints, input_size=None, target_size=None):
        """convert label to target

        Parameters
        ----------
        keypoints : array, [N, 51]
            keypoints annnotation
        input_size : tuple, [h, w]
            network input size, such as (540, 960). 
            If batch == 1, set input_size here.
        target_size: tuple, [h, w]
            network output size, such as (540, 960)
            If batch == 1, set target_size here.

        """
        if input_size is None:
            input_size = self.input_size
        if target_size is None:
            target_size = self.target_size
        assert input_size is not None
        assert target_size is not None

        # add neck point
        keypoints = add_extra_joint(keypoints)
        heatmap, heatmap_weight = get_heatmap(keypoints.copy(), input_size, target_size, self.keep_invis, self.sigma)
        paf, paf_weight = get_vectormap(keypoints, input_size, target_size, self.keep_invis)

        return heatmap, heatmap_weight, paf, paf_weight


# Decoder


def smoothen(heatmap, gauss_sigma):
    # heatmap: (channel, feat_height,  feat_width)
    channel, height, width = heatmap.shape
    heatmap = heatmap.reshape((-1, 1, height, width))
    heatmap = mx.nd.array(heatmap)

    # gaussian kernel
    kernel_x = np.arange(-1, 2)
    kernel_y = np.arange(-1, 2)
    kernel_x, kernel_y = np.meshgrid(kernel_x, kernel_y)
    kernel_x = kernel_x.reshape((-1,))
    kernel_y = kernel_y.reshape((-1,))
    dist = kernel_x ** 2 + kernel_y ** 2
    kernel = np.exp(dist / (-2.0 * gauss_sigma * gauss_sigma))
    kernel /= kernel.sum()
    kernel = mx.nd.array(kernel.reshape(1, 1, 3, 3))
    
    # smoothen
    heatmap = mx.nd.Convolution(
        data = heatmap,
        weight = kernel,
        kernel = (3, 3),
        num_filter = 1,
        pad = (1, 1),
        stride = (1, 1),
        no_bias = True)
    return heatmap.asnumpy().reshape(channel, height, width)


# heatmap => peaks
def compute_peaks_from_heatmap(heatmap):
    """compute peaks from heatmap, following two steps:
    1. gaussian smooth
    2. find local maximum

    Parameters
    ----------
    heatmap : array, [num_keypoint, h, w]

    Returns
    -------
    peaks: array: [num_keypoint, h, w]
        peaks

    """
    # smooth
    heatmap_smooth = smoothen(heatmap, gauss_sigma=1)

    # find pinks
    channel, height, width = heatmap.shape
    heatmap_smooth = mx.nd.array(heatmap_smooth.reshape((-1, 1, height, width)))
    heatmap_pool = mx.nd.Pooling(
            data=heatmap_smooth,
            kernel=(3, 3), 
            stride=(1, 1), 
            pad=(1, 1),
            pool_type='max')
    peaks = mx.nd.where(mx.nd.equal(heatmap_pool, heatmap_smooth),
                        heatmap_pool,
                        mx.nd.zeros_like(heatmap_pool))
    return peaks.asnumpy().reshape(channel, height, width)


class Decoder(object):
    """Target to predict transform for openpose task

    """
    def __init__(self, num_keypoint, target_size, stride, num_limb=None):
        self.num_keypoint = num_keypoint
        self.target_size = target_size
        self.stride = stride
        if num_limb is None:
            self.num_limb = coco_num_limb

    def __call__(self, heatmap, paf):
        """convert target to predict

        Parameters
        ----------
        heatmap : array, [num_keypoint, h, w]
        paf : array, [num_limb, h, w]

        Returns
        -------
        TODO

        """
        peaks = compute_peaks_from_heatmap(heatmap)

        heatmap = np.transpose(heatmap, (1,2,0))
        paf = np.transpose(paf, (1,2,0))
        peaks = np.transpose(peaks, (1,2,0))

        from .pafprocess import pafprocess
        pafprocess.process_paf(peaks, heatmap, paf)

        keypoints = []
        for human_id in range(pafprocess.get_num_humans()):
            kps = []
            for part_idx in range(self.num_keypoint):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    kps.extend([0, 0, 0])
                else:
                    x = pafprocess.get_part_x(c_idx)
                    y = pafprocess.get_part_y(c_idx)
                    score = pafprocess.get_part_score(c_idx)
                    kps.extend([x, y, score])
            #if np.sum(kps) > 0:
            #    score = pafprocess.get_score(human_id)
            #    kps.append(score)
            keypoints.extend(kps)
        keypoints = np.array(keypoints, dtype=np.float32)
        keypoints = keypoints.reshape(pafprocess.get_num_humans(), -1)
        return keypoints

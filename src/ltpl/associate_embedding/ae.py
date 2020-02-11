# -*- coding: utf-8 -*-
"""label-target-predict convert utility in pose task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


__all__ = ['Encoder', 'Decoder']


class GenerateHeatmap():
    """
    ported from https://github.com/princeton-vl/pose-ae-train
    """
    def __init__(self, num_parts, sigma, keep_invis, output_res=None):
        self.output_res = output_res    # [h, w]
        self.num_parts = num_parts
        self.keep_invis = keep_invis
        self.sigma = sigma      # self.output_res/64
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints, output_res=None):
        if output_res is None:
            output_res = self.output_res
        assert output_res is not None

        hms = np.zeros(shape = (self.num_parts, self.output_res[0], self.output_res[1]), dtype = np.float32)
        hms_weight = np.zeros_like(hms)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if (self.keep_invis and pt[2] == 0.0) or ( (not self.keep_invis) and pt[2] == 2.0):
                        continue
                x, y = int(pt[0]), int(pt[1])
                if x<0 or y<0 or x>=self.output_res[1] or y>=self.output_res[0]:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c,d = max(0, -ul[0]), min(br[0], self.output_res[1]) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], self.output_res[0]) - ul[1]

                cc,dd = max(0, ul[0]), min(br[0], self.output_res[1])
                aa,bb = max(0, ul[1]), min(br[1], self.output_res[0])
                hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
                hms_weight[idx, :, :] = 1.0
        return hms, hms_weight


class KeypointsRef():
    """
    ported from https://github.com/princeton-vl/pose-ae-train
    """
    def __init__(self, max_num_people, num_parts, keep_invis):
        self.max_num_people = max_num_people
        self.num_parts = num_parts
        self.keep_invis = keep_invis

    def __call__(self, keypoints, output_res):
        visible_nodes = np.zeros((self.max_num_people, self.num_parts, 2))
        visible_weight = np.zeros_like(visible_nodes)
        for i in range(len(keypoints)):
            tot = 0
            for idx, pt in enumerate(keypoints[i]):
                x, y = int(pt[0]), int(pt[1])
                if (self.keep_invis and pt[2] == 0.0) or ( (not self.keep_invis) and pt[2] == 2.0):
                        continue
                if x>=0 and y>=0 and x<output_res[0] and y<output_res[1]:
                    visible_nodes[i][tot] = (idx * output_res[0] * output_res[1] + y * output_res[1] + x, 1)
                    visible_weight[i, tot, :] = 1.0
                    tot += 1
        return visible_nodes, visible_weight


class Encoder(object):
    """Label to target transform for Associative Embedding task

    Parameter
    ---------
    input_size : tuple, [h, w]
        input size
    target_size : tuple, [h, w]
        target size
    num_keypoint : int
        keypoint sum
    gauss_sigma: float
        gaussian sigma
    max_num_people: int
        max people num in one image, default is 30

    """
    def __init__(self, input_size, target_size, num_keypoint, gauss_sigma, max_num_people=30, keep_invis=True):
        self.input_size = input_size
        self.target_size = target_size
        self.stride = input_size[0] // target_size[0]
        self.num_keypoint = num_keypoint
        self.guass_sigma = gauss_sigma
        self.keep_invis = keep_invis
        self.generateHeatmap = GenerateHeatmap(num_keypoint, gauss_sigma, keep_invis, output_res=target_size)
        self.keypointsRef = KeypointsRef(max_num_people, num_keypoint, keep_invis)

    def __call__(self, keypoints):
        """convert label to target

        """
        keypoints = keypoints.reshape(-1, self.num_keypoint, 3)
        keypoints[:, :, :2] /= self.stride
        heatmap, heatmap_weight = self.generateHeatmap(keypoints)
        ref, ref_weight = self.keypointsRef(keypoints, self.target_size)
        return heatmap, heatmap_weight, ref, ref_weight


class Decoder(object):
    """Target to predict transform for Associative Embedding task

    """
    def __init__(self):
        pass

    def __call__(self):
        """convert target to predict

        """
        pass

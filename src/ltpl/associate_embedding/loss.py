"""det loss."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mxnet.gluon.loss import Loss, _reshape_like, _apply_weighting


def singleTagLoss(F, pred_tag, keypoints):
    """associative embedding loss for one image

    Parameters
    ----------
    pred_tag : [num_keypoint, H, W]
    keypoints : [num_people, num_keypoint, 2], 2 is [index, valid]

    Returns
    -------
    pull: 
        pull loss
    push: 
        push loss

    """
    num_keypoint = pred_tag.shape[0]
    target_h = pred_tag.shape[1]
    target_w = pred_tag.shape[2]

    eps = 1e-6
    sigma = 1.0
    tags = []
    pull = 0
    for i in keypoints:
        tmp = []
        for j in i:
            if j[1]>0:
                index_c = j[0].asscalar() // (target_h * target_w)
                index_h = (j[0].asscalar() - index_c * target_h * target_w) // target_w
                index_w = j[0].asscalar() - index_c * target_h * target_w - index_h * target_w
                tmp.append(pred_tag[int(index_c), int(index_h), int(index_w)])
        if len(tmp) == 0:
            continue
        tmp = F.stack(*tmp).reshape(-1)
        tags.append(tmp.mean())
        pull = pull + F.mean(F.power(tmp - tags[-1].repeat(tmp.shape[0]), 2))

    if len(tags) == 0:
        return F.zeros([1]), F.zeros([1])

    tags = F.stack(*tags).reshape(-1)
    num = tags.shape[0]
    A = tags.expand_dims(axis=0).tile((num, 1))
    B = A.transpose(axes=(1,0))

    diff = F.power(A-B, 2) / (2 * sigma * sigma)
    push = F.exp(-diff)
    push = (F.sum(push) - num)                  # rm identify matrix

    pull = pull / (num + eps)
    push = push / ((num-1)*num + eps)
    return pull, push


class GroupingLoss(Loss):
    r"""Calculates Grouping loss.

    refer to paper <Associative Embedding: End-to-End Learning for Joint Detection and Grouping>

    Parameter
    ---------
    pred_tag : ndarray
        predict tag, [B, num_keypoint, H, W]
    keypointRef : ndarray
        keypoint reference, [B, num_people, num_keypoint, 2], 2 is [index, valid]

    """
    def __init__(self, alpha=1.0,
                 weight=None, batch_axis=0,
                 size_average=True, **kwargs):
        super(GroupingLoss, self).__init__(weight, batch_axis, **kwargs)
        self._size_average = size_average
        self._eps = 1e-6
        self._alpha = alpha

    def hybrid_forward(self, F, pred_tag, keypointRef, sample_weight=None):
        """Loss forward"""
        tags = []
        num_batch = keypointRef.shape[0]
        
        pull, push = 0.0, 0.0
        for i in range(num_batch):
            each_pull, each_push = singleTagLoss(F, pred_tag[i], keypointRef[i])
            pull += each_pull
            push += each_push
        loss = pull + push * self._alpha

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            return F.mean(loss, axis=self._batch_axis, exclude=True)
        else:
            return F.sum(loss, axis=self._batch_axis, exclude=True)

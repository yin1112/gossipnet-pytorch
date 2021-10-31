# -*- coding:utf-8 -*-
# @Time : 2021/9/6 19:14
# @Author: yin
# @File : dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import numpy as np
from cv2 import imread, resize
import torch
from nms_net_pytorch import  cfg


DEBUG = False


def load_roi(need_images, roi, is_training=False):
    if DEBUG:
        print('loading ', roi)

    # make a copy so we don't keep the loaded image around
    roi = dict(roi)

    im_scale = 1.0
    if need_images:
        roi['image'], im_scale = load_image(roi['filename'], roi['flipped'])
        roi['image'] = roi['image'][None, ...]
        # don't do multiplications inplace
        if 'dets' in roi:
            roi['dets'] = roi['dets'] * im_scale
        if 'gt_boxes' in roi:
            roi['gt_boxes'] = roi['gt_boxes'] * im_scale
    roi['im_scale'] = im_scale
    return roi


def load_image(path, flipped):
    target_size = cfg.image_target_size
    max_size = cfg.image_max_size

    im = imread(path)
    if len(im.shape) == 2:
        im = np.tile(im[..., None], (1, 1, 3))
    h, w = im.shape[:2]
    im_size_min = min(h, w)
    im_size_max = max(h, w)
    im_scale = target_size / im_size_min
    if round(im_scale * im_size_max) > max_size:
        im_scale = max_size / im_size_max

    if flipped:
        im = im[:, ::-1, :].copy()
    im = resize(im, im_scale)
    return im, im_scale

class ShuffledDataset(torch.utils.data.Dataset):
    def __init__(self, imdb , is_training):
        self._imdb = imdb
        self._roidb = imdb['roidb']
        self._batch_size = 1
        self._need_images = False
        self.is_training = is_training
        self._cur = 0
        self._perm = np.arange(len(self._roidb))
        if self.is_training:
            self._shuffle()

    def _shuffle(self):
        if self.is_training:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        if DEBUG:
            print(self._perm)
        self._cur = 0

    def next_batch(self):
        if self._cur + self._batch_size > self._perm.size:
            self._shuffle()

        db_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        assert len(db_inds) == 1

        roi = self._roidb[db_inds[0]]
        roi = load_roi(self._need_images, roi, is_training=True)
        return roi
    def __len__(self):
        return len(self._roidb)



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self ,roi ):
        # self._imdb = imdb
        self._roidb = roi

    def __getitem__(self , index ):
        roi = self._roidb[index]
        return roi
    def __len__(self):
        return len(self._roidb)
class ValDataset(torch.utils.data.Dataset):
    def __init__(self ,roi ):
        self._roidb = roi

    def __getitem__(self , index ):
        roi = self._roidb[index]
        return roi
    def __len__(self):
        return len(self._roidb)

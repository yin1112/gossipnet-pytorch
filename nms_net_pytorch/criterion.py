# -*- coding:utf-8 -*-
# @Time : 2021/9/6 19:27
# @Author: yin
# @File : criterion.py
import torch
from torch import nn


class Criterion(nn.Module):
    def __init__(self  ,device):
        super(Criterion, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    def forward(self , prediction_score , labels):
        return self.loss(prediction_score , labels)


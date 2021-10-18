# -*- coding:utf-8 -*-
# @Time : 2021/9/6 19:14
# @Author: yin
# @File : matching.py

import numpy
import numpy as np
import torch
def DetectionMatching(ious ,score ,ignore  ):

    assert ious.shape[0] ==score.shape[0] ,"DetectionMatching expects dim 1 of input 1 and dim 1 of input 2 to be the same ( %s != %s)"%(ious.shape[0],score.shape[0])
    assert ious.shape[1] == ignore.shape[0] , "DetectionMatching expects dim 2 of input 1 and dim 1 of input 3 to be the same ( %s != %s)"%(ious.shape[0],ignore.shape[0])
    t_score = score.reshape(-1)

    det_order = torch.argsort(-t_score)
    gt_order = ignore.argsort()
    n_dets = ious.shape[0]
    n_gt = ious.shape[1]

    is_matched = torch.zeros(n_gt ,dtype= torch.int64)
    labels = torch.zeros(n_dets)
    assignments = torch.ones(n_dets)
    assignments = -1*assignments
    weights = torch.ones_like(score)


    iou_thresh  = 0.5
    for _det_i in range(n_dets):
        det = det_order[_det_i]
        # print(_det_i)
        iou = iou_thresh
        match = -1
        for _gt_i in range(n_gt):
            gt = gt_order[_gt_i]
            # print(gt)
            if is_matched[gt] and ignore[gt] ==0:
                continue
            if match>-1 and ignore[gt] :
                break
            # print(det , gt)
            if ious[det,gt] <iou :
                continue
            iou = ious[det,gt]
            match = gt
        if match>-1:
            is_matched[match] = 1
            labels[det] = 1
            assignments[det] = match
            if ignore[match]:
                weights[det] = 0
    # lables表示预测框是否匹配上， weights =1正常匹配 ， =0 匹配到应该被忽略的真实框，  assignments 表示匹配到哪一个真实框0

    
    return labels , weights , assignments

if __name__ == "__main__" :
    dt = 4
    gt = 3
    # _ious = torch.randn(dt*gt).reshape(dt,gt)
    # _score = torch.randn(dt)
    # _ignore = torch.tensor([ i >0.5 for i in np.random.random(gt)])
    print(__file__)
    _ious =  np.load("/home/wusl/qinhua/gossipnet-master/nms_net_pytorch/scores1.npy")
    _score = np.load("/home/wusl/qinhua/gossipnet-master/nms_net_pytorch/scores2.npy")
    _ignore = np.load("/home/wusl/qinhua/gossipnet-master/nms_net_pytorch/scores3.npy")
    print(_ious)
    print(_score)
    print(_ignore)
    print("----------------")
    _ious = torch.tensor(_ious)
    _score = torch.tensor(_score)
    _ignore = torch.tensor(_ignore)
    labels , weights , assignments = DetectionMatching(_ious,_score,_ignore)
    print("----------------")
    print(labels)
    print(weights)
    print(assignments)
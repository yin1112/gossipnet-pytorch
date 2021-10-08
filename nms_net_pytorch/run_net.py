# -*- coding:utf-8 -*-
# @Time : 2021/9/8 15:22
# @Author: yin
# @File : run_net.py
import numpy as np
import torch
torch.set_printoptions(profile="full")
np.set_printoptions(threshold = 1e6)
from nms_net_pytorch import cfg
from nms_net_pytorch.criterion import Criterion
from nms_net_pytorch.matching import DetectionMatching

class Run_net:


    def __init__(self ,gnet ,device ,num_classes ,class_weights ):
        self.num_classes = num_classes.to(device)
        self.class_weights = class_weights.to(device)
        self.criterion = Criterion().to(device)
        self.gnet = gnet
        self.device = device

    def setdata(self , dets,det_scores ,det_classes, gt_boxes ,gt_classes, gt_crowd):
        
        self.dets = dets
        self.det_scores = det_scores
        self.det_classes = det_classes
        self.gt_boxes = gt_boxes
        self.gt_classes = gt_classes
        self.gt_crowd = gt_crowd

    def run_net(self ):



        self.multiclass = self.num_classes > 1
        ''' 
            dets_boxdata:  torch.Size([519, 7])
            gt_boxesdata:  torch.Size([17, 7])

        '''
        self.dets_boxdata = self._xyxy_to_boxdata(self.dets)
        self.gt_boxesdata = self._xyxy_to_boxdata(self.gt_boxes)

        '''
            det_anno_iou =   torch.Size([519, 17])
            det_det_iou =    torch.Size([519, 519])
        '''
        self.det_anno_iou = self._iou(self.dets_boxdata, self.gt_boxesdata, crowd=self.gt_crowd)
        self.det_det_iou = self._iou(self.dets_boxdata, self.dets_boxdata)


        if self.multiclass:
            '''
                det_anno_iou =  [449, 9]
            '''
            same_class = torch.eq(self.det_classes.reshape(-1, 1),
                                  self.gt_classes.reshape(1, -1))
            zeros = torch.zeros_like(self.det_anno_iou).to(self.device)
            self.det_anno_iou = torch.where(same_class, self.det_anno_iou, zeros)


        '''
            neighbor_pair_idxs = torch.Size([52569, 2])

        '''
        neighbor_pair_idxs = (self.det_det_iou >= cfg.gnet.neighbor_thresh).nonzero( as_tuple=False)
        pair_c_idxs = neighbor_pair_idxs[:, 0]
        pair_n_idxs = neighbor_pair_idxs[:, 1]
        self.num_dets = self.dets.shape[0]


        '''
            pw_feats = torch.Size([52569, 167])
        '''
        pw_feats = (self._geometry_feats(pair_c_idxs, pair_n_idxs) * cfg.gnet.pw_feat_multiplyer) # check ok 
        '''
            new_score = torch.Size([519, 1])
        '''
        new_score = self.gnet.forward(self.num_dets, pw_feats, pair_c_idxs, pair_n_idxs)
        
        '''
            labels = torch.Size([519])
            weights = torch.Size([519, 1]) 
            det_gt_matching = torch.Size([519])
        '''
        labels, weights, det_gt_matching = \
            DetectionMatching(self.det_anno_iou, new_score, self.gt_crowd) #check ok 

        '''
            self.class_weights = (81,)
            det_crowd = torch.Size([519])
            det_class = torch.Size([519])
        '''
        self.class_weights = self.class_weights.reshape(-1)
        if self.class_weights is None:  
            self.class_weights = torch.ones((self.num_classes + 1), dtype=torch.float32).to(self.device)
        # else:
        #     class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

        self.gt_crowd = self.gt_crowd.reshape(-1)
        if self.gt_crowd.shape[0] > 0:
            det_crowd = self.gt_crowd[torch.clamp(det_gt_matching, min=0).to(torch.long)]
            det_crowd[det_gt_matching==-1] = 0
        else:
            det_crowd = torch.zeros_like(labels, dtype=torch.bool).to(self.device)
        
        if self.gt_crowd.shape[0] > 0:
            det_class = self.gt_classes[torch.clamp(det_gt_matching, min=0).to(torch.long)].to(torch.long)
            det_class[det_gt_matching==-1] = 0

        else:
            det_class = torch.zeros_like(labels, dtype=torch.long).to(self.device)
        
        zeros = torch.zeros_like(det_class).to(self.device)

        det_class = torch.where(
            torch.logical_and(det_gt_matching >= 0, torch.logical_not(det_crowd)),
            det_class,
            zeros
        )

        sample_weights = self.class_weights[det_class.to(torch.long)]

        weights = (weights.reshape(-1, 1) * sample_weights.reshape(-1, 1)).reshape(-1)

        prediction = new_score.reshape(-1)

        sample_loss = self.criterion(prediction, labels)

        weights_losses = sample_loss.reshape(-1, 1) * weights.reshape(-1, 1)

        loss_unnormed = torch.sum(weights_losses)
        loss_normed = torch.mean(weights_losses)

        if cfg.train.normalize_loss:
            loss = loss_normed * cfg.train.loss_multiplyer
        else:
            loss = loss_unnormed * cfg.train.loss_multiplyer

        return new_score , labels, weights, det_gt_matching ,\
                loss_unnormed , loss_normed , loss
       
    def _xyxy_to_boxdata(self, a):
        ax1 = a[:, 0].reshape(-1, 1)
        ay1 = a[:, 1].reshape(-1, 1)
        ax2 = a[:, 2].reshape(-1, 1)
        ay2 = a[:, 3].reshape(-1, 1)

        aw = ax2 - ax1
        ah = ay2 - ay1

        area = torch.mul(aw, ah)
        return torch.cat([ax1, ay1, aw, ah, ax2, ay2, area], dim=1)

    def _iou(self, a, b, crowd=None):
        a = a.transpose(1, 0)
        b = b.transpose(1, 0)
        a_area = torch.reshape(a[6], [-1, 1])
        b_area = torch.reshape(b[6], [1, -1])

        intersection = self._intersection(a, b)  # 求交集
        union = torch.sub(torch.add(a_area, b_area), intersection)  # 并集
        iou = torch.div(intersection, union)
        if crowd is None:
            return iou
        else:
            ioa = torch.div(intersection, a_area)
            crowd = crowd.reshape(1, -1).repeat(a_area.shape[0], 1)
            return torch.where(crowd, ioa, iou)

    def _intersection(self, a, b):
        ax1 = a[0]
        ay1 = a[1]
        ax2 = a[4]
        ay2 = a[5]

        bx1 = b[0]
        by1 = b[1]
        bx2 = b[4]
        by2 = b[5]

        x1 = torch.maximum(torch.reshape(ax1, [-1, 1]), torch.reshape(bx1, [1, -1]))
        y1 = torch.maximum(torch.reshape(ay1, [-1, 1]), torch.reshape(by1, [1, -1]))
        x2 = torch.minimum(torch.reshape(ax2, [-1, 1]), torch.reshape(bx2, [1, -1]))
        y2 = torch.minimum(torch.reshape(ay2, [-1, 1]), torch.reshape(by2, [1, -1]))

        w = torch.clamp(torch.sub(x2, x1), min=0)
        h = torch.clamp(torch.sub(y2, y1), min=0)
        intersection = torch.mul(w, h)
        return intersection

    def _geometry_feats(self, c_idxs, n_idxs):

        if self.multiclass:

            mc_score_idxs = torch.arange(self.num_dets).reshape(-1, 1)

            mc_score_idxs = torch.cat(
                (mc_score_idxs, torch.ones_like(mc_score_idxs) * ((self.det_classes - 1).reshape(-1, 1))), 1)

            tmp_scores = torch.zeros([self.num_dets, self.num_classes])

            tmp_scores[mc_score_idxs[:, 0], mc_score_idxs[:, 1]] = self.det_scores


        else:
            tmp_scores = self.det_scores.unsqueeze(-1)
           
        c_score = tmp_scores[c_idxs]
        n_score = tmp_scores[n_idxs]
        tmp_ious = self.det_det_iou.unsqueeze(-1)
        ious = tmp_ious[c_idxs, n_idxs]

        x1, y1, w, h, _, _, _ = (self.dets_boxdata.transpose(1, 0))

        c_w = w[c_idxs]
        c_h = h[c_idxs]
        c_scale = ((c_w + c_h) / 2.0).reshape(-1, 1)
        c_cx = x1[c_idxs] + c_w / 2.0
        c_cy = y1[c_idxs] + c_h / 2.0

        n_w = w[n_idxs]
        n_h = h[n_idxs]
        n_cx = x1[n_idxs] + n_w / 2.0
        n_cy = y1[n_idxs] + n_h / 2.0

        # normalized x, y distance
        x_dist = (n_cx - c_cx).reshape(-1, 1)
        y_dist = (n_cy - c_cy).reshape(-1, 1)
        l2_dist = torch.sqrt(x_dist ** 2 + y_dist ** 2) / c_scale
        x_dist /= c_scale
        y_dist /= c_scale

        # scale difference
        log2 = torch.tensor(np.log(2.0), dtype=torch.float32)
        w_diff = (torch.log(n_w / c_w) / log2).reshape(-1, 1)
        h_diff = (torch.log(n_h / c_h) / log2).reshape(-1, 1)
        aspect_diff = ((torch.log(n_w / n_h) - torch.log(c_w / c_h)) / log2).reshape(-1, 1)

        all = torch.cat([c_score, n_score, ious, x_dist, y_dist, l2_dist, w_diff, h_diff, aspect_diff], dim=1)
        all.requires_grad = False

        # print(all.shape)
        return all
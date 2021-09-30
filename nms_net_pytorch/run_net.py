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
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Run_net:

    gnet = None
    optimizer = None
    num_classes = None
    dets = None
    det_scores = None
    det_classes = None
    gt_boxes = None
    gt_crowd = None
    gt_classes = None
    class_weights = None
    cnt =0

    def __init__(self  ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = Criterion().to(device)

    def setdata(self ,gnet,optimizer, num_classes , dets,det_scores ,det_classes, gt_boxes ,gt_classes, gt_crowd,class_weights = None):
        self.gnet = gnet
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.dets = dets
        self.det_scores = det_scores
        self.det_classes = det_classes
        self.gt_boxes = gt_boxes
        self.gt_classes = gt_classes
        self.gt_crowd = gt_crowd
        self.class_weights = class_weights

    def run_net(self , ema,istraining = False ):

        # print("-----------------")
        # print(self.det_classes)
        # print("---------------")
        # print(self.gt_classes)
        # print("-------------")
        # input()

        # sto = self.gt_boxes
        # with open('scores.txt','w') as f:
        #    f.write( str(sto) )
        # print("input : ")
        # input()

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
            zeros = torch.zeros_like(self.det_anno_iou)
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
        # pw_feats = Variable(pw_feats , requires_grad = True )
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

        # print(labels.requires_grad)
        # print(weights.requires_grad)
        # print(det_gt_matching.requires_grad)
        # new_score = np.load("/home/wusl/qinhua/gossipnet-master/nms_net_pytorch/scores.npy")
        # _ious =  np.load("/home/wusl/qinhua/gossipnet-master/nms_net_pytorch/scores1.npy")
        # _score = np.load("/home/wusl/qinhua/gossipnet-master/nms_net_pytorch/scores2.npy")
        # _ignore = np.load("/home/wusl/qinhua/gossipnet-master/nms_net_pytorch/scores3.npy")
        # _ious = torch.tensor(_ious)
        # _score = torch.tensor(_score)
        # _ignore = torch.tensor(_ignore)
        # new_score =torch.tensor(new_score)
        # labels , weights , det_gt_matching = DetectionMatching(_ious,_score,_ignore)


        # print(labels.requires_grad)
        # print(weights.requires_grad)
        # print(det_gt_matching.requires_grad)
        # print(new_score.requires_grad)
        # input()
        # new_score = new_score.reshape(-1)
        # weights =weights.reshape(-1)
        # mask = weights>0
        # new_score = new_score[mask]
        # weights = weights[mask]
        # labels = labels[mask]
        # mask = torch.argsort(-new_score)
        # new_score = new_score[mask]
        # # weights = weights[mask]
        # labels = labels[mask]
        # print(new_score.shape)
        # print(labels.shape)
        # print(weights.shape)
        # print(det_gt_matching.shape)
        # print(self.gt_classes.shape)
        # print(self.gt_boxes.shape)
        # print(self.det_scores.shape)
        # sto = new_score
        # with open('scores0.txt','w') as f:
        #     f.write( str(sto) )
        # sto = labels
        # with open('scores1.txt','w') as f:
        #     f.write( str(sto) )
        # sto = det_gt_matching
        # with open('scores2.txt','w') as f:
        #     f.write( str(sto) )
        # print("input : ")
        # input()

        '''
            self.class_weights = (81,)
            det_crowd = torch.Size([519])
            det_class = torch.Size([519])
        '''
        self.class_weights = self.class_weights.reshape(-1)
        if self.class_weights is None:  
            self.class_weights = torch.ones((self.num_classes + 1), dtype=torch.float32)
        # else:
        #     class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.gt_crowd = self.gt_crowd.reshape(-1)
        if self.gt_crowd.shape[0] > 0:
            det_crowd = self.gt_crowd[torch.clamp(det_gt_matching, min=0).to(torch.long)]
            det_crowd[det_gt_matching==-1] = 0
        else:
            det_crowd = torch.zeros_like(labels, dtype=torch.bool)
        
        if self.gt_crowd.shape[0] > 0:
            det_class = self.gt_classes[torch.clamp(det_gt_matching, min=0).to(torch.long)].to(torch.long)
            det_class[det_gt_matching==-1] = 0

        else:
            det_class = torch.zeros_like(labels, dtype=torch.long)
        
        zeros = torch.zeros_like(det_class)

        det_class = torch.where(
            torch.logical_and(det_gt_matching >= 0, torch.logical_not(det_crowd)),
            det_class,
            zeros
        )

        sample_weights = self.class_weights[det_class.to(torch.long)]

        weights = (weights.reshape(-1, 1) * sample_weights.reshape(-1, 1)).reshape(-1)


        # sto = weights
        # with open('scores.txt','w') as f:
        #     f.write( str(sto) )
        # print("input : ")
        # input()

        #Optimization of scores for positive tags
        # mask =  np.logical_or ( np.logical_and(labels.reshape(-1)>0 , new_score.reshape(-1)<=0),np.logical_and(labels.reshape(-1)==0 , new_score.reshape(-1)>0) ) 
        # print(mask)
        # if sum(mask==True)>0:
        #     k1  = np.linspace(.0, 2.00,sum(mask == True),
        #                    endpoint=True)
        #     weights[mask] = weights[mask] + k1
        # mask = np.logical_and(labels.reshape(-1)==1 , new_score.reshape(-1)<=0)
        # weights = torch.where(mask , weights*5  , 0.5*weights)
        # print(mask.shape)
        # print(k1.shape)
        # print(weights[mask].shape)
        # print(weights.shape)
        # input()




        prediction = new_score.reshape(-1)

        # print(weights.requires_grad)
        # print(prediction.requires_grad)
        # print(prediction.shape , labels.shape , weights.shape)


        sample_loss = self.criterion(prediction, labels)

        # print(sample_loss.requires_grad)


        weights_losses = sample_loss.reshape(-1, 1) * weights.reshape(-1, 1)

        # print(weights_losses.requires_grad)

        # input()
        # sto = weights_losses
        # with open('scores2.txt','w') as f:
        #     f.write( str(sto) )
        
        # sto = labels
        # with open('scores3.txt','w') as f:
        #     f.write( str(sto) )
        # print("input : ")
        # input()

        loss_unnormed = torch.sum(weights_losses)
        loss_normed = torch.mean(weights_losses)





        if cfg.train.normalize_loss:
            loss = loss_normed * cfg.train.loss_multiplyer
        else:
            loss = loss_unnormed * cfg.train.loss_multiplyer

        if istraining:

        # print(loss)
        # input()
            self.optimizer.zero_grad()            
            loss.backward()

            # self.cnt+=1            
            # if self.cnt >10:

            #     with open('weights.txt','w') as f:
            #         for name, parms in self.gnet.named_parameters():	
            #                 sto = ('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            #         ' -->grad_value:',parms.grad)
            #                 f.write( str(sto) )
            #     print("---")
            #     input()
                # for m in self.gnet.modules(): # 继承nn.Module的方法
                #     if type(m)==torch.nn.modules.linear.Linear:
                #         print(m.weight)
                # print("--------------")
                # input()
            # print("--------------------")
            # input()
            
            
            self.optimizer.step()
            ema.update()
        
        # sto = new_score
        # with open('scores.txt','w') as f:
        #    f.write( str(sto) )
        # print("input : ")
        # input()
        return new_score ,labels , loss_unnormed, loss_normed ,weights

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
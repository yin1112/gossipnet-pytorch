# -*- coding:utf-8 -*-
# @Time : 2021/9/6 19:14
# @Author: yin
# @File : train.py
import os
from datetime import datetime
import threading
from pprint import pprint
import argparse

from numpy.core.fromnumeric import shape
import torch
import numpy as np
import imdb
from nms_net_pytorch import cfg
from nms_net_pytorch.class_weights import class_equal_weights
from nms_net_pytorch.config import cfg_from_file
from nms_net_pytorch.criterion import Criterion
from nms_net_pytorch.dataset import   ShuffledDataset ,load_roi
from nms_net_pytorch.matching import DetectionMatching
from nms_net_pytorch.net import Gnet
from nms_net_pytorch.run_net import Run_net
from nms_net_pytorch.ema import EMA
import pickle
torch.set_printoptions(profile="full")
np.set_printoptions(threshold = 1e6)


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_dataset():
    train_imdb = imdb.get_imdb(cfg.train.imdb , is_training=True)
    # train_imdb = load_obj("t1")


    val_imdb = imdb.get_imdb(cfg.train.val_imdb , is_training=False)

    return ShuffledDataset(train_imdb) , train_imdb 


# sto = new_score
# with open('scores0.txt','w') as f:
#     f.write( str(sto) )
# sto = labels
# with open('scores1.txt','w') as f:
#     f.write( str(sto) )
# print("ok")
# input()
def val_run(run_net, net, val_imdb , device , optimizer , class_weights ,ema):
    
    ema.apply_shadow()
    
    

    roidb = val_imdb['roidb']
    net.eval()
    all_labels = []
    all_scores = []
    all_classes = []
    for i, roi in enumerate(roidb):
        
        if 'dets' not in roi or roi['dets'].size == 0:
            continue
        roi = load_roi(False, roi)

        
        dets = roi['dets']
        dets_class = roi['det_classes']
        dets_score = roi['det_scores']
        gt_boxes = roi['gt_boxes']
        gt_classes = roi['gt_classes']
        gt_crowd = roi['gt_crowd']

        dets = torch.tensor(dets).to(device)
        det_classes = torch.tensor(dets_class).to(device)
        det_scores = torch.tensor(dets_score).to(device)
        gt_boxes = torch.tensor(gt_boxes).to(device)
        gt_classes = torch.tensor(gt_classes).to(device)
        gt_crowd = torch.tensor(gt_crowd).to(device)
        run_net.setdata( net, optimizer ,  torch.tensor(val_imdb['num_classes']), dets,
                                 det_scores, det_classes, gt_boxes, gt_classes,
                                 gt_crowd, class_weights)
        new_score ,labels , _, _ , weights =  run_net.run_net(ema , istraining = False)



        mask = (weights > 0.0).reshape(-1)
        new_score = new_score.reshape(-1)





        roi_det_classes = torch.tensor(roi['det_classes']).reshape(-1)
        all_labels.append(labels[mask].detach().numpy())
        all_scores.append(new_score[mask].detach().numpy())
        all_classes.append(roi_det_classes[mask])
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    ema.restore()

    return compute_aps(scores, classes, labels, val_imdb)

def compute_aps(scores, classes, labels, val_imdb):
    ord = np.argsort(-scores)
    scores = scores[ord]
    labels = labels[ord]
    classes = classes[ord]

    sto = scores
    with open('scores2.txt','w') as f:
        f.write( str(sto) )
    sto = labels
    with open('scores3.txt','w') as f:
        f.write( str(sto) )

    num_objs = sum(np.sum(np.logical_not(roi['gt_crowd']))
                   for roi in val_imdb['roidb'])
    multiclass_ap = _compute_ap(scores, labels, num_objs)


    all_cls = np.unique(classes)
    print(num_objs , all_cls)
    cls_ap = []
    for cls in iter(all_cls):
        mask = classes == cls
        c_scores = scores[mask]
        c_labels = labels[mask]
        cls_gt = (np.logical_and(np.logical_not(roi['gt_crowd']),
                                 roi['gt_classes'] == cls)
                  for roi in val_imdb['roidb'])
        c_num_objs = sum(np.sum(is_cls_gt)
                         for is_cls_gt in cls_gt)
        cls_ap.append(_compute_ap(c_scores, c_labels, c_num_objs))

    mAP = np.mean(cls_ap)
    return mAP, multiclass_ap, cls_ap

 # print(num_objs)
    # print(fp.shape)
    # print(tp.shape)
    # print(fp)
    # print(tp)

    # with open('scores2.txt','w') as f:
    #     f.write( str(sto) )
    # sto = fp
    # with open('scores3.txt','w') as f:
    #     f.write( str(sto) )
    # sto = recall
    # with open('scores2.txt','w') as f:
    #     f.write( str(sto) )
    # sto = precision
    # with open('scores3.txt','w') as f:
    #     f.write( str(sto) )

    # print("input : ")
    # input()
    # print(recall.shape)
    # print(precision.shape)
    # print(recall)
    # print(precision)
def _compute_ap(scores, labels, num_objs):
    # computer recall & precision
    fp = np.cumsum((labels == 0).astype(dtype=np.int32)).astype(dtype=np.float32)
    tp = np.cumsum((labels == 1).astype(dtype=np.int32)).astype(dtype=np.float32)
    recall = tp / num_objs
    precision = tp / (fp + tp)

   
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    recall = np.concatenate(([0], recall, [recall[-1], 2]), axis=0)
    precision = np.concatenate(([1], precision, [0, 0]), axis=0)
    # computer AP
    c_recall = np.linspace(.0, 1.00,100 + 1,
                           endpoint=True)
    inds = np.searchsorted(recall, c_recall, side='left')
    c_precision = precision[inds]
    ap = np.average(c_precision) * 100
    return ap


        #for layer,param in self.gnet.state_dict().items(): # param is weight or bias(Tensor) 
	      #    print (layer,param)
        #input()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
class Train:

    num_classes = None
    run_net = None
    dets = None
    det_scores = None
    det_classes = None
    gt_boxes = None
    gt_crowd = None
    gt_classes = None

    def __init__(self ,save_dict = 'dict/dict3.pth' , load_dict ='dict/dict3.pth'):
        self.train_sets,self.train_imdb= get_dataset()
        
        self.load_dict =""
        self.save_dict =save_dict
        if  self.load_dict!=""  and os.path.exists(self.load_dict):
            print("load weights")
            self.gnet = torch.load(self.load_dict)
        



        #do_val
        self.do_val = len(cfg.train.val_imdb) > 0
        if self.do_val:
            self.val_imdb = imdb.get_imdb(cfg.train.val_imdb, is_training=False)
            # self.val_imdb['roidb'] = self.val_imdb['roidb'][0:100]
            # assert self.train_imdb['num_classes'] == self.val_imdb['num_classes']

        self.run_net = Run_net()
        self.num_classes =torch.tensor(self.train_imdb['num_classes'])
        self.gnet = Gnet(self.train_imdb['num_classes'] * 2  + 7 )
        self.ema = EMA(self.gnet , 0.7)
        self.class_weights = class_equal_weights(self.train_imdb)



        if cfg.train.optimizer == 'adam':
            self.optimizer =  torch.optim.SGD(self.gnet.parameters() ,lr = 0.001  )
        elif cfg.train.optimizer == 'sgd':
            self.optimizer = torch.optim.sgd(
                momentum=cfg.train.momentum ,lr = 0.0001)
        else:
            raise ValueError('unknown optimizer {}'.format(cfg.train.optimizer))

    def trainer(self):

        print("trainlen :%d , vallen :%d"%(len(self.train_sets) , len(self.val_imdb['roidb'])))


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gnet =self.gnet.to(device)
        start_iter = 1
        for it in range(start_iter , cfg.train.num_iter+1):

            self.gnet.train()
            roi = self.train_sets.next_batch()
            dets = roi['dets']
            dets_class = roi['det_classes']
            dets_score = roi['det_scores']
            gt_boxes = roi['gt_boxes']
            gt_classes = roi['gt_classes']
            gt_crowd = roi['gt_crowd']

            self.dets = torch.tensor(dets).to(device)
            self.det_classes = torch.tensor(dets_class).to(device)
            self.det_scores = torch.tensor(dets_score).to(device)
            self.gt_boxes = torch.tensor(gt_boxes).to(device)
            self.gt_classes = torch.tensor(gt_classes).to(device)
            self.gt_crowd = torch.tensor(gt_crowd).to(device)

                
            if(self.dets.shape[0] ==1):
                continue

            self.run_net.setdata(self.gnet,self.optimizer , self.num_classes,self.dets,
                                    self.det_scores,self.det_classes,self.gt_boxes,self.gt_classes,
                                    self.gt_crowd,self.class_weights)
            
            # for i in range(20):
            score , _ , loss_unnormed, loss_normed ,_ = self.run_net.run_net(self.ema,istraining = True )
                
            if it % cfg.train.display_iter ==0:

                print(('{}  iter {:6d}  '
                       'data loss normalized {:8g}   '
                       'unnormalized {:8g}').format(
                    datetime.now(), it, 
                    loss_normed, loss_unnormed))

            if self.do_val and it % cfg.train.val_iter == 0:
                print('{}  starting validation'.format(datetime.now()))
                
                val_map, mc_ap, pc_ap  =  val_run(self.run_net, self.gnet, self.val_imdb , device , self.optimizer , self.class_weights , self.ema)
                print(('{}  iter {:6d}   validation pass:   mAP {:5.1f}   '
                       'multiclass AP {:5.1f}').format(
                      datetime.now(), it, val_map, mc_ap))
            if self.save_dict!="" and (it % cfg.train.save_iter == 0 or (self.do_val and it % cfg.train.val_iter == 0) ):
                self.ema.apply_shadow()
                torch.save(self.gnet ,self.save_dict)
                self.ema.restore()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=False, action='store_true')
    parser.add_argument('-c', '--config', default='conf.yaml')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    args, unparsed = parser.parse_known_args()

    cfg_from_file(args.config)
    if args.visualize:
        cfg.gnet.load_imfeats = True
    pprint(cfg)

    t = Train()
    t.trainer()

if __name__ == '__main__':
    main()

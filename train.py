# -*- coding:utf-8 -*-
# @Time : 2021/9/8 15:22
# @Author: yin
# @File : run_net.py
from __future__ import division

import os
import argparse
import time
import numpy as np
from pprint import pprint
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn    
from nms_net_pytorch import cfg
from nms_net_pytorch.dataset import   ShuffledDataset 
from nms_net_pytorch.net import Gnet
from nms_net_pytorch.run_net import Run_net
import imdb
from nms_net_pytorch.config import cfg_from_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-co', '--config', default='conf.yaml')
    parser.add_argument('-cu','--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='weights/', type=str)
    args, unparsed = parser.parse_known_args()

    cfg_from_file(args.config)
    return parser.parse_args()


def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder)

    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    
    train_cfg = cfg.train

    print("Setting Arguments.. : ")
    pprint(cfg)
    print("----------------------------------------------------------")
    print('Loading the dataset...')     

    train_sets,train_imdb= get_dataset(True)
    val_sets , val_imdb = get_dataset(False)
    assert train_imdb['num_classes'] == val_imdb['num_classes']

    print('Training model on:', train_cfg.imdb)
    print('The training_dataset size:', len(train_sets))
    print('The val_dataset size:',len(val_sets))
    print("----------------------------------------------------------")   

    model = Gnet( 9,device )
    #print(model)
    model.to(device).train()

    run_net = Run_net(model , device,torch.tensor(train_imdb['num_classes']))

    # keep training
    if train_cfg.resume is not None:
        print('keep training model: %s' % (train_cfg.resume))
        model.load_state_dict(torch.load(train_cfg.resume, map_location=device))

    # optimizer setup
    lr = train_cfg.lr_multi_step
    optimizer = optim.SGD(model.parameters(), 
                            lr=lr[0][1], 
                            momentum=train_cfg.momentum,
                            weight_decay=train_cfg.weight_decay
                            )
    max_iter = train_cfg.max_iter


    t0 = time.time()
    for iter in range(max_iter):

        tmp_lr = None
        for i in lr:
            if iter < i[0]:
                set_lr(optimizer, i[1])
                tmp_lr = i[1]
                break
        
        # set data
        run_net.setdata(*get_per_data(train_sets,device))
        # apply(run_net.setdata , get_per_data(train_sets , device))
        # forward
        _ , _, _, _ ,\
                loss_unnormed , loss_normed , loss= run_net.run_net()

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % train_cfg.display_iter ==0:

            t1 = time.time()
            print('[Epoch %d/%d][lr %.6f]'
                '[loss_unnormed:  %.2f ][ loss_normed:  %.2f ][time:  %.2f]] '
                    % (iter+1, max_iter, tmp_lr,
                        loss_unnormed, loss_normed, t1-t0),
                    flush=True)

            t0 = time.time()

        # evaluation
        if  len(val_sets)>0 and (iter + 1) % train_cfg.val_iter == 0:

            model.trainable = False
            model.eval()
            all_labels = []
            all_scores = []
            all_classes = []
            for i, roi in enumerate(val_imdb['roidb']):
                
                #set data

                run_net.setdata(*get_per_data(val_sets , device))
                # evaluate
                new_score , labels, weights, _ ,\
                    _ , _ , _= run_net.run_net()

                mask = (weights > 0.0).reshape(-1)
                new_score = new_score.reshape(-1)
                roi_det_classes = torch.tensor(roi['det_classes']).reshape(-1)
                all_labels.append(labels[mask].detach().numpy())
                all_scores.append(new_score[mask].detach().numpy())
                all_classes.append(roi_det_classes[mask])
            scores = np.concatenate(all_scores, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            classes = np.concatenate(all_classes, axis=0)

            mAP, multiclass_ap, _ =compute_aps(scores, classes, labels, val_imdb)

            t1 = time.time()
            print('[Epoch %d/%d]'
                '[mAP:  %.2f ][multiclass_ap:  %.2f ][time:  %.2f]] '
                    % (iter+1, max_iter,
                        mAP, multiclass_ap, t1-t0),
                    flush=True)
            t0 = time.time()

            # convert to training mode.
            model.trainable = True
            model.train()

        # save model
        if (iter + 1) % train_cfg.save_iter == 0:
            print('Saving state, epoch:', iter + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        '_' + repr(iter + 1) + '.pth')
                        )  


def get_dataset(is_training = True):
    train_imdb = imdb.get_imdb(cfg.train.imdb , is_training=is_training)
    return ShuffledDataset(train_imdb) , train_imdb 

def get_per_data(train_sets , device):
    roi = train_sets.next_batch()
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
    if(dets.shape[0] ==1):
        return get_per_data(train_sets , device)
    else :
        return dets,det_scores,det_classes,\
                        gt_boxes,gt_classes,gt_crowd
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_aps(scores, classes, labels, val_imdb):
    ord = np.argsort(-scores)
    scores = scores[ord]
    labels = labels[ord]
    classes = classes[ord]

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

if __name__ == '__main__':
    train()

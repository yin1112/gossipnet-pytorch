from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data
from nms_net_pytorch import cfg
from nms_net_pytorch.class_weights import class_equal_weights
from nms_net_pytorch.config import cfg_from_file
from nms_net_pytorch.criterion import Criterion
from nms_net_pytorch.dataset import   ShuffledDataset ,load_roi
from nms_net_pytorch.matching import DetectionMatching
from nms_net_pytorch.net import Gnet
from nms_net_pytorch.run_net import Run_net
from nms_net_pytorch.ema import EMA
import imdb

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)

    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    
    train_cfg = cfg.train

    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the dataset...')     

    train_sets,train_imdb= get_dataset(True)
    val_sets , val_imdb = get_dataset(False)
    assert train_imdb['num_classes'] == val_imdb['num_classes']

    print('Training model on:', train_cfg.imdb)
    print('The training_dataset size:', len(train_sets))
    print('The val_dataset size:',len(val_sets))
    print("----------------------------------------------------------")   

    model = Gnet(train_imdb['num_classes'] * 2  + 7 ,device )
    #print(model)
    model.to(device).train()

    run_net = Run_net(device,torch.tensor(train_imdb['num_classes']),class_equal_weights(train_imdb))

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
    max_epoch = train_cfg.max_epoch


    t0 = time.time()
    for epoch in range(max_epoch):

        tmp_lr = None
        for i in lr:
            if epoch < i[0]:
                set_lr(optimizer, i[1])
                tmp_lr = i[1]
                break
        
        # set data
        run_net.setdata(get_per_data(train_sets , device))
        # forward
        _ , _, _, _ ,\
                loss_unnormed , loss_normed , loss= run_net.run_net()

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % train_cfg.display_iter:

            t1 = time.time()
            print('[Epoch %d/%d][lr %.6f]'
                '[loss_unnormed:  %.2f ][ loss_normed:  %.2f ][time:  %.2f]] '
                    % (epoch+1, max_epoch, tmp_lr,
                        loss_unnormed, loss_normed, t1-t0),
                    flush=True)

            t0 = time.time()

        # evaluation
        if  len(val_sets)>0 and (epoch + 1) % train_cfg.val_iter == 0:

            model.trainable = False
            model.eval()
            all_labels = []
            all_scores = []
            all_classes = []
            for i, roi in enumerate(val_imdb['roidb']):
                
                #set data
                run_net.setdata(get_per_data(val_sets , device))
                # evaluate
                new_score , labels, weights, det_gt_matching ,\
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

            mAP, multiclass_ap, cls_ap =compute_aps(scores, classes, labels, val_imdb)

            t1 = time.time()
            print('[Epoch %d/%d]'
                '[mAP:  %.2f ][multiclass_ap:  %.2f ][time:  %.2f]] '
                    % (epoch+1, max_epoch,
                        mAP, multiclass_ap, t1-t0),
                    flush=True)
            t0 = time.time()

            # convert to training mode.
            model.trainable = True
            model.train()

        # save model
        if (epoch + 1) % train_cfg.save_iter == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        '_' + repr(epoch + 1) + '.pth')
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

    sto = scores
    with open('scores2.txt','w') as f:
        f.write( str(sto) )
    sto = labels
    with open('scores3.txt','w') as f:
        f.write(str(sto) )

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

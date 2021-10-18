# -*- coding:utf-8 -*-
# @Time : 2021/9/6 19:14
# @Author: yin
# @File : net.py

import torch
import torch.nn as nn
from nms_net_pytorch.config import cfg
from torch_scatter import scatter_max

import numpy as np
# torch.set_printoptions(profile="full")
# np.set_printoptions(threshold = 1e6)


def fc(infeature , outfeature ,fc_num , Have_ReLu = True,Have_Bn =  False):
    pw_fc =[]
    pw_fc .append(nn.Linear(infeature, outfeature))
    if Have_Bn:
        pw_fc.append(nn.BatchNorm1d(outfeature))
    if Have_ReLu:
        pw_fc .append(nn.ReLU())
    for i in range(fc_num-1):
        pw_fc .append(nn.Linear(outfeature, outfeature))
        if Have_Bn:
            pw_fc.append(nn.BatchNorm1d(outfeature))    
        if Have_ReLu:
            pw_fc .append(nn.ReLU())
    return torch.nn.Sequential(*pw_fc)


class Block(nn.Module):
    def __init__(self , device):
        super(Block,self).__init__()

        self.device = device
        #make other detections
        self.reduce_dim = fc(cfg.gnet.shortcut_dim , cfg.gnet.reduced_dim , 1,Have_ReLu=True ,Have_Bn=False )

        #pairwise computations
        self.pw_fc = fc(cfg.gnet.reduced_dim*2 +cfg.gnet.pwfeat_narrow_dim ,cfg.gnet.pairfeat_dim ,cfg.gnet.num_block_pw_fc,Have_ReLu=True, Have_Bn=False)


        #after pooling
        self.fc1 = fc(cfg.gnet.pairfeat_dim ,cfg.gnet.pairfeat_dim ,cfg.gnet.num_block_fc,Have_ReLu=True , Have_Bn=False )
        self.fc2 = fc(cfg.gnet.pairfeat_dim ,cfg.gnet.shortcut_dim ,1,Have_ReLu= False, Have_Bn=False )
        self.output_Relu = torch.nn.ReLU()
    def forward(self,infeats ,  pw_feats ,dets_num ,c_idxs ,n_idxs):
        

        t_infeats = self.reduce_dim(infeats)


        c_feats = t_infeats[c_idxs]
        n_feats = t_infeats[n_idxs]

        is_id_row = torch.eq(c_idxs,n_idxs).reshape(-1,1).to(self.device)
        zeros = torch.zeros_like(n_feats).to(self.device)
        n_feats = torch.where(is_id_row,zeros,n_feats)


        feats = torch.cat((pw_feats,c_feats  , n_feats),1) #get pairwise_context
        

        feats = self.pw_fc(feats)
        #pooling        
        c_idxs = c_idxs.reshape(-1,1)
        tmp_feats ,idx = scatter_max(feats, c_idxs, dim=0)

        out_feats = self.fc1(tmp_feats)
        out_feats = self.fc2(out_feats)
        output = self.output_Relu(out_feats+infeats)

        return output



class Gnet(nn.Module):
    def __init__(self ,g_feats_num ,device ):
        super(Gnet,self).__init__()

        self.g_feats_num = g_feats_num
        self.device = device
        self.pw_feats_fc = self._pw_feats_fc()

        self.block = torch.nn.ModuleList()
        for i in range(cfg.gnet.num_blocks):
            self.block.append(Block(device))

        self.predict_fc = fc(cfg.gnet.shortcut_dim ,cfg.gnet.predict_fc_dim, cfg.gnet.num_predict_fc , Have_ReLu = False, Have_Bn=False )
        self.predict_logits = nn.Linear(cfg.gnet.predict_fc_dim  , 1 )


        self._init_weight()
    def forward(self,num_dets,pw_feats ,c_idxs , n_idxs):


        out_feats = torch.zeros([num_dets, cfg.gnet.shortcut_dim] ).to(self.device)

        pw_feats = self.pw_feats_fc(pw_feats)
        

        for each_block in self.block:
            out_feats = each_block.forward(out_feats , pw_feats , num_dets , c_idxs,n_idxs)
            
        out_feats = self.predict_fc(out_feats)
        out_feats = self.predict_logits(out_feats)  



        return out_feats

    def _pw_feats_fc(self):
        pw_feats_fc = []
        pw_feats_fc.append(fc(self.g_feats_num , cfg.gnet.pwfeat_dim , cfg.gnet.num_pwfeat_fc , Have_ReLu=True , Have_Bn=False) )

        pw_feats_fc.append(fc(cfg.gnet.pwfeat_dim ,cfg.gnet.pwfeat_narrow_dim , 1 , Have_ReLu=True , Have_Bn=False))

        return torch.nn.Sequential(*pw_feats_fc)

    def _init_weight(self):
        for m in self.modules(): 
            
            if type(m)==torch.nn.modules.linear.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif type(m)==torch.nn.modules.batchnorm.BatchNorm1d:
                pass

if __name__ == '__main__':
    print("-------------------111")
    print(cfg.gnet.num_blocks)
    print("------------------11")

    pw_feats_fc = torch.nn.Linear(10,5)
    input = torch.randn((10,10,10))
    # print(input)
    output = pw_feats_fc(input)

    # total = torch.sum(input)
    # ave = torch.div(total , 20)
    # fancha = torch.div((input - ave)**2 ,10)
    print(input.shape)
    print(output.shape)

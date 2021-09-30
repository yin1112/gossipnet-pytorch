# -*- coding:utf-8 -*-
# @Time : 2021/9/6 19:14
# @Author: yin
# @File : net.py

import torch
import torch.nn as nn
from nms_net_pytorch.config import cfg
from torch.autograd import Variable
# from pytorch_scatter import torch_scatter

import numpy as np
torch.set_printoptions(profile="full")
np.set_printoptions(threshold = 1e6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    def __init__(self):
        super(Block,self).__init__()

        #make other detections
        self.reduce_dim = fc(cfg.gnet.shortcut_dim , cfg.gnet.reduced_dim , 1,Have_ReLu=True ,Have_Bn=False )

        #pairwise computations
        self.pw_fc = fc(cfg.gnet.reduced_dim*2 +cfg.gnet.pwfeat_narrow_dim ,cfg.gnet.pairfeat_dim ,cfg.gnet.num_block_pw_fc,Have_ReLu=True, Have_Bn=False)


        #after pooling
        self.fc1 = fc(cfg.gnet.pairfeat_dim ,cfg.gnet.pairfeat_dim ,cfg.gnet.num_block_fc,Have_ReLu=True , Have_Bn=False )
        self.fc2 = fc(cfg.gnet.pairfeat_dim ,cfg.gnet.shortcut_dim ,1,Have_ReLu= False, Have_Bn=False )
        self.output_Relu = torch.nn.ReLU()
    def forward(self,infeats ,  pw_feats ,dets_num ,c_idxs ,n_idxs):
        


        #infeats 上一个block的输入 ， pw_feats 人工特征
        t_infeats = self.reduce_dim(infeats)


        c_feats = t_infeats[c_idxs]#取得各自预测框信息
        n_feats = t_infeats[n_idxs]#取得各自预测框信息

        is_id_row = torch.eq(c_idxs,n_idxs).reshape(-1,1)
        zeros = torch.zeros_like(n_feats)
        n_feats = torch.where(is_id_row,zeros,n_feats)


        feats = torch.cat((pw_feats,c_feats  , n_feats),1) #得到pairwise_context
        

        feats = self.pw_fc(feats)#Fc序列
        #pooling        
        c_idxs = c_idxs.reshape(-1,1)
        idx = torch.cat((c_idxs,torch.arange(c_idxs.shape[0]).reshape(-1,1)),1)
        tmp_feats = torch.zeros(dets_num , cfg.gnet.pairfeat_dim)
        tmp_feats = -1 * tmp_feats
        tmp_feats[idx[:,0]] = torch.max(tmp_feats[idx[:,0]] , feats[idx[:,1]])




        # print(tmp_feats.requires_grad) # False
        out_feats = self.fc1(tmp_feats)
        # print(feats.requires_grad) #True
        out_feats = self.fc2(out_feats)
        # print(feats.requires_grad) #True

        # input()
        # print(tmp_feats.shape)
        # print(dets_num ,cfg.gnet.pairfeat_dim )

        # tmp_feats = feats.new_empty((dets_num , cfg.gnet.pairfeat_dim))
        # tmp_feats.detach_()
        # tmp_feats.is_leaf=False
        # tmp_feats[idx[:,0]] = feats[idx[:,1]].clone() -1

        # print(tmp_feats)
        # print(tmp_feats.shape)
        # print(tmp_feats.is_leaf)
        # print(feats.requires_grad)
        # input()
        # print(tmp_feats.grad_fn)

        # input()
        # print(dets_num ,  cfg.gnet.pairfeat_dim)
        # input()
        # print(tmp_feats)
        # print(tmp_feats.shape)
        # print(tmp_feats.is_leaf)
        # print(tmp_feats.requires_grad)
        # print(tmp_feats.grad_fn)
        # input()
        # print(tmp_feats.requires_grad)



        # print(tmp_feats)    
        # sto = feats[:50]
        # with open('scores.txt','w') as f:
        #     f.write( str(sto) )
        # print("input 5 : ")
        # input()
        # print(tmp_feats.requires_grad)

        # input()   
        
        # print(feats.grad_fn)
        # print(feats.requires_grad)
        # input()
        # feats = Variable(feats , requires_grad = True)
        output = self.output_Relu(out_feats+infeats)

        # print(output.is_leaf)
        # print(output.grad_fn)
        # input()
        # output = Variable(output , requires_grad = True)
        return output



class Gnet(nn.Module):
    def __init__(self ,g_feats_num ):
        super(Gnet,self).__init__()

        self.g_feats_num = g_feats_num

        self.pw_feats_fc = self._pw_feats_fc()

        self.block = torch.nn.ModuleList()
        for i in range(cfg.gnet.num_blocks):
            self.block.append(Block())

        self.predict_fc = fc(cfg.gnet.shortcut_dim ,cfg.gnet.predict_fc_dim, cfg.gnet.num_predict_fc , Have_ReLu = False, Have_Bn=False )
        self.predict_logits = nn.Linear(cfg.gnet.predict_fc_dim  , 1 )
        # self.predict_logits_sig = nn.Sigmoid()
        self._init_weight()
    def forward(self,num_dets,pw_feats ,c_idxs , n_idxs):



        out_feats = torch.zeros([num_dets, cfg.gnet.shortcut_dim] )
        out_feats = Variable(out_feats , requires_grad = True)
        # sto = pw_feats[:50]
        # with open('scores.txt','w') as f:
        #     f.write( str(sto) )
        # print("input 1 : ")
        # input()   
        # print(pw_feats.shape)
        # sto = pw_feats[:3 , :]
        # with open('scores.txt','w') as f:
        #     f.write( str(sto) )
        # print("input 2 : ")
        # input()  
        pw_feats = self.pw_feats_fc(pw_feats)
        
        #pw_feats = Variable(pw_feats , requires_grad = True)
        # sto = pw_feats[:3 , :]
        # with open('scores.txt','w') as f:
        #     f.write( str(sto) )
        # print("input 2 : ")
        # input()  

        # print(pw_feats.is_leaf)

        # print("---------------")

        for each_block in self.block:
            out_feats = each_block.forward(out_feats , pw_feats , num_dets , c_idxs,n_idxs)
            
            # print(num_dets)

        out_feats = self.predict_fc(out_feats)


        
        out_feats = self.predict_logits(out_feats)  



        return out_feats

    def _pw_feats_fc(self):
        pw_feats_fc = []
        pw_feats_fc.append(fc(self.g_feats_num , cfg.gnet.pwfeat_dim , cfg.gnet.num_pwfeat_fc , Have_ReLu=True , Have_Bn=False) )

        pw_feats_fc.append(fc(cfg.gnet.pwfeat_dim ,cfg.gnet.pwfeat_narrow_dim , 1 , Have_ReLu=True , Have_Bn=False))

        return torch.nn.Sequential(*pw_feats_fc)

    def _init_weight(self):
        # print(self)
        cnt =0
        for m in self.modules(): # 继承nn.Module的方法
            
            if type(m)==torch.nn.modules.linear.Linear:
                cnt+=1
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            elif type(m)==torch.nn.modules.batchnorm.BatchNorm1d:
                pass
        print("成功初始化%d层全连接层"%(cnt))

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

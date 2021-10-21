import paddle.nn as nn
import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.io import Dataset,DataLoader
import glob
import os
from tqdm import tqdm
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import backbone

class DANet(nn.Layer):
    def __init__(self,name_scope,out_chs=20,in_chs=2048,inter_chs=512):
        super(DANet,self).__init__(name_scope)
        name_scope = self.full_name()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.inter_chs = inter_chs if inter_chs else in_chs


        self.backbone = backbone.ResNet(101)
        self.conv5p = nn.Sequential(
            nn.Conv2D(self.in_chs, self.inter_chs, 3, padding=1),
            nn.BatchNorm(self.inter_chs,act='relu'),
        )
        self.conv5c = nn.Sequential(
            nn.Conv2D(self.in_chs, self.inter_chs, 3, padding=1),
            nn.BatchNorm(self.inter_chs,act='relu'),
        )

        self.sp = PAM_module(self.inter_chs)
        self.sc = CAM_module(self.inter_chs)

        self.conv6p = nn.Sequential(
            nn.Conv2D(self.inter_chs, self.inter_chs, 3, padding=1),
            nn.BatchNorm(self.inter_chs,act='relu'),
        )
        self.conv6c = nn.Sequential(
            nn.Conv2D(self.inter_chs, self.inter_chs, 3, padding=1),
            nn.BatchNorm(self.inter_chs,act='relu'),
        )

        self.conv7p = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2D(self.inter_chs, self.out_chs, 1),
        )
        self.conv7c = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2D(self.inter_chs, self.out_chs, 1),
        )
        self.conv7pc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2D(self.inter_chs, self.out_chs, 1),
        )

    def forward(self,x):

        feature = self.backbone(x)

        p_f = self.conv5p(feature)
        p_f = self.sp(p_f)
        p_f = self.conv6p(p_f)
        p_out = self.conv7p(p_f)

        c_f = self.conv5c(feature)
        c_f = self.sc(c_f)
        c_f = self.conv6c(c_f)
        c_out = self.conv7c(c_f)

        sum_f = p_f+c_f
        sum_out = self.conv7pc(sum_f)

        up=nn.Upsample(size=x.shape[2:])
        p_out = up(p_out)
        c_out = up(c_out)
        sum_out = up(sum_out)
        return [p_out, c_out, sum_out]
        # return sum_out

class PAM_module(nn.Layer):
    def __init__(self,in_chs,inter_chs=None):
        super(PAM_module,self).__init__()
        self.in_chs = in_chs
        self.inter_chs = inter_chs if inter_chs else in_chs
        self.conv_query = nn.Conv2D(self.in_chs,self.inter_chs,1)
        self.conv_key = nn.Conv2D(self.in_chs,self.inter_chs,1)
        self.conv_value = nn.Conv2D(self.in_chs,self.inter_chs,1)
        self.gamma = paddle.static.create_parameter([1], dtype='float32')
        self._softmax=nn.Softmax()
    
    def forward(self,x):
        b,c,h,w = x.shape

        f_query = self.conv_query(x)
        f_query = paddle.reshape(f_query,(b, -1, h*w))
        f_query = paddle.transpose(f_query,(0, 2, 1)) 

        f_key = self.conv_key(x)
        f_key = paddle.reshape(f_key,(b, -1, h*w))

        f_value = self.conv_value(x)
        f_value = paddle.reshape(f_value,(b, -1, h*w))
        f_value = paddle.transpose(f_value,(0, 2, 1)) 


        f_similarity = paddle.bmm(f_query, f_key)                        # [h*w, h*w]
        f_similarity = self._softmax(f_similarity)
        f_similarity = paddle.transpose(f_similarity,(0, 2, 1))

        f_attention = paddle.bmm(f_similarity, f_value)                        # [h*w, c]
        f_attention = paddle.reshape(f_attention,(b,c,h,w))

        out = self.gamma*f_attention + x
        return out

class CAM_module(nn.Layer):
    def __init__(self,in_chs,inter_chs=None):
        super(CAM_module,self).__init__()
        self.in_chs = in_chs
        self.inter_chs = inter_chs if inter_chs else in_chs
        self.gamma = paddle.static.create_parameter([1], dtype='float32')

    def forward(self,x):
        b,c,h,w = x.shape

        f_query = paddle.reshape(x,(b, -1, h*w))
        f_key = paddle.reshape(x,(b, -1, h*w))
        f_key = paddle.transpose(f_key,(0, 2, 1)) 
        f_value = paddle.reshape(x,(b, -1, h*w))

        f_similarity = paddle.bmm(f_query, f_key)                        # [h*w, h*w]
        f_similarity_max = paddle.max(f_similarity, -1, keepdim=True)
        f_similarity_max_reshape = paddle.expand_as(f_similarity_max,f_similarity)
        f_similarity = f_similarity_max_reshape-f_similarity

        f_similarity = nn.functional.softmax(f_similarity)
        f_similarity = paddle.transpose(f_similarity,(0, 2, 1)) 

        f_attention = paddle.bmm(f_similarity,f_value)                        # [h*w, c]
        f_attention = paddle.reshape(f_attention,(b,c,h,w))

        out = self.gamma*f_attention + x
        return out
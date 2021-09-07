'''
"CBAM: Convolutional Block Attention Module (ECCV2018)"
'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ..view import Flatten
from ..fc import FCLayer,FCBlock_v2
from ..convbase import ConvLayer,ConvBaseBlock
from ..trivial import ActivationLayer

__all__=['CBAMBlock']

class SpatialPool(nn.Module):
    '''
    Pool C*H*W data into n*H*W where n is the number of pooling types
    '''
    def __init__(self,pooling_types,*args,**kwargs):
        super(SpatialPool,self).__init__()
        pooling_modules=[]
        for t in pooling_types:
            if t=='max':
                pooling_modules.append(nn.AdaptiveMaxPool2d((1,1)))
            elif t=='avg':
                pooling_modules.append(nn.AdaptiveAvgPool2d((1,1)))
        self.pooling=nn.ModuleList(pooling_modules)
    def forward(self,x):
        y=sum([m(x) for m in self.pooling])
        return y

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, pooling_types=None,*args,**kwargs):
        super(ChannelGate, self).__init__()
        if pooling_types is None:
            pooling_types=['max','avg']
        gate_channels = [gate_channel,gate_channel // reduction_ratio,gate_channel]
        activation=['relu','none']
        modules = [
            SpatialPool(pooling_types),
            Flatten(),
            FCBlock_v2(gate_channels, bn=True, activate=activation),
        ]
        self.main=nn.Sequential(*modules)
    def forward(self, x):
        y=self.main(x)
        return y.unsqueeze(-1).unsqueeze(-1).expand_as(x)

class ChannelPool(nn.Module):
    '''
    Stack channel-wise max and mean.
    '''
    def __init__(self,dim=1):
        super(ChannelPool,self).__init__()
        self.dim=dim
    def forward(self,x):
        dim=self.dim
        return torch.cat([torch.max(x,dim), torch.mean(x,dim)], dim=dim)

class SpatialGate(nn.Module):
    def __init__(self,*args,**kwargs):
        super(SpatialGate, self).__init__()
        self.main=nn.Sequential(
            ChannelPool(),
            ConvBaseBlock([2,1],7,-1,'sigmoid',bn_track=True),
        )
    def forward(self, x):
        return self.main(x)

class Gate(nn.Module):
    def __init__(self,activation='sigmoid',*args,**kwargs):
        super(Gate,self).__init__()
        self.act=ActivationLayer(activation)
        self.channel=ChannelGate(*args,**kwargs)
        self.spatial=SpatialGate(*args,**kwargs)
    def forward(self,x):
        channel_att=self.channel(x)
        spatial_att=self.spatial(x)
        att=self.act(channel_att*spatial_att)
        return att

class CBAMBlock(nn.Module):
    '''
    CBAM. Possible args include: activation,gate_channel(*),dim
    '''
    def __init__(self, *args,**kwargs):
        super(CBAMBlock, self).__init__()
        self.att=Gate(*args,**kwargs)
    def forward(self,x):
        att=self.att(x)
        return att * x
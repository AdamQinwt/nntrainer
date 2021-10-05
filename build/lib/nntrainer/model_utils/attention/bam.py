'''
"BAM: Bottleneck Attention Module (BMVC2018)"
'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ..view import Flatten
from ..fc import FCLayer,FCBlock_v2
from ..convbase import ConvLayer,ConvBaseBlock
from ..trivial import ActivationLayer

__all__=['BAMBlock']

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1,*args,**kwargs):
        super(ChannelGate, self).__init__()
        modules=[nn.AdaptiveAvgPool2d((1,1)),Flatten()]
        gate_channels = [gate_channel]+([gate_channel // reduction_ratio] * num_layers)
        modules.append(FCBlock_v2(gate_channels,bn=True,activate='relu'))
        modules.append(nn.Linear(gate_channels[-1], gate_channel))
        self.main=nn.Sequential(*modules)
    def forward(self, x):
        y=self.main(x)
        return y.unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, *args,**kwargs):
        super(SpatialGate, self).__init__()
        gate_channels=[gate_channel]+[gate_channel//reduction_ratio]*dilation_conv_num
        self.main=nn.Sequential(
            ConvBaseBlock(gate_channels,1,-1,'relu',bn_track=True),
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1)
        )
    def forward(self, x):
        return self.main(x).expand_as(x)

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
        return att+1

class BAMBlock(nn.Module):
    def __init__(self, *args,**kwargs):
        '''
        BAM. Possible args include: activation,gate_channel(*),reduction_ratio,dilation_conv_num,num_layers
        :param args:
        :param kwargs:
        '''
        super(BAMBlock, self).__init__()
        self.att=Gate(*args,**kwargs)
    def forward(self,x):
        att=self.att(x)
        return att * x
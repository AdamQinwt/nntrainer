'''
Residual Attention Network for Image Classification from CVPR 2017
'''

import torch
import torch.nn as nn
from ..convbase import ResConvLayers

class DownSample(nn.Module):
    def __init__(self,nchannels,nconv=1,pool=2):
        super(DownSample,self).__init__()
        self.pre=ResConvLayers(nchannels=nchannels)
        self.pool=nn.MaxPool2d(pool,pool)
        self.conv=ResConvLayers(num=nconv,nchannels=nchannels)
    def forward(self,x):
        y=self.pre(x)
        return self.conv(y),self.pool(y)

class UpSampleA(nn.Module):
    def __init__(self,nchannels,tgt_size):
        super(UpSampleA,self).__init__()
        self.up=nn.UpsamplingBilinear2d(tgt_size)
        self.conv=ResConvLayers(nchannels=nchannels)
    def forward(self,x,skip=None):
        x=self.up(x)
        if skip:
            x=x+skip
        return self.conv(x)

class UpSampleB(nn.Module):
    def __init__(self,nchannels,tgt_size):
        super(UpSampleB,self).__init__()
        self.up=nn.UpsamplingBilinear2d(tgt_size)
        self.conv=ResConvLayers(nchannels=nchannels)
    def forward(self,x,skip):
        x=self.up(x)+1.0
        return self.conv(x*skip)

class ResAttBlock(nn.Module):
    def __init__(self,nchannels,orig_size,nfold=2,bottom_nconv=2,trunk_nconv=1,*args,**kwargs):
        super(ResAttBlock,self).__init__()
        downsample=[]
        upsample=[]
        sizes=[orig_size]
        for i in range(nfold):
            from_size=sizes[i]
            downsample.append(DownSample(nchannels))
            if i==0:
                upsample.append(UpSampleB(nchannels, from_size))
            else:
                upsample.append(UpSampleA(nchannels,from_size))
            sizes.append([(orig_size[0]+1)>>1,(orig_size[1]+1)>>1])
        self.nfold=nfold
        self.down=nn.ModuleList(downsample)
        self.up=nn.ModuleList(upsample)
        self.trunk=ResConvLayers(trunk_nconv,nchannels=nchannels)
        self.bottom=ResConvLayers(bottom_nconv,nchannels=nchannels)
    def forward(self,x):
        skips=[]
        nfold=self.nfold
        for i in range(nfold):
            skipx,x=self.down[i](x)
            skips.append(skips)
        x=self.bottom(x)
        skips[0]=self.trunk(skips[0])
        for i in range(nfold):
            x=self.up[nfold-1-i](x,skips[nfold-1-i])
        return x
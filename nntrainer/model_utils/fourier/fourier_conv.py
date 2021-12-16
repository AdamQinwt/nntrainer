# module from Resolution-robust Large Mask Inpainting with Fourier Convolutions(https://arxiv.org/pdf/2109.07161v2.pdf)

import torch
import torch.nn as nn
from nntrainer.model_utils.convbase import ConvLayer
from nntrainer.model_utils.attention import SELayer
from torch.fft import rfft2,irfft2
class FourierConv(nn.Module):
    def __init__(self,in_channels,out_channels=None,conv='conv'):
        super(FourierConv,self).__init__()
        if out_channels is None:
            out_channels=in_channels
        self.out_channels=out_channels
        if conv=='conv':
            self.conv=ConvLayer(in_channels<<1,out_channels<<1,1)
        elif conv=='se':
            self.conv=SELayer(in_channels<<1,8)
        else:
            raise ValueError()
        self.output=nn.Conv2d(out_channels,out_channels,1,padding=1)
    def forward(self,x):
        out_channels=self.out_channels
        X=rfft2(x)
        # print(X.size())
        X=torch.cat([X.real,X.imag],dim=1)
        processed=self.conv(X)
        processed=torch.complex(processed[:,:out_channels],processed[:,out_channels:])
        processed=irfft2(processed)[...,1:-1,1:-1]
        # print(processed.size())
        return self.output(processed)

class FFC(nn.Module):
    def __init__(self,in_channels,out_channels=None,fourier_conv='conv',aggregate='cat'):
        super(FFC,self).__init__()
        if out_channels is None:
            out_channels=in_channels
        self.local_branch=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.global_branch=FourierConv(in_channels,out_channels,conv=fourier_conv)
        if aggregate=='cat':
            self.output=ConvLayer(out_channels<<1,out_channels,ks=1)
        elif aggregate=='sum':
            self.output=None
        else:
            raise ValueError()
        self.aggregate=aggregate
    def forward(self,x):
        l=self.local_branch(x)
        g=self.global_branch(x)
        # print(x.size(),l.size(),g.size())
        if self.aggregate=='cat':
            r=self.output(torch.cat([l,g],1))
        elif self.aggregate=='sum':
            r=l+g
        return r

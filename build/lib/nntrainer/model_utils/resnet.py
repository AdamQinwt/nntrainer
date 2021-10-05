import torch
import torch.nn as nn
from nntrainer.model_utils.convbase import ConvLayer,ConvBaseBlock
from nntrainer.model_utils.model_parser import Factory

class ResNetBlock_small(nn.Module):
    '''
        Basic block for resnet-18
    '''
    def __init__(self,in_channel,out_channel,*args,**kwargs):
        super(ResNetBlock_small,self).__init__()
        self.downsample=nn.Sequential(
            ConvLayer(in_channel,out_channel,stride=(2,2)),
            ConvLayer(out_channel,out_channel,activation='none'),
        )
        self.side=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),stride=(2,2),bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.out=ConvBaseBlock([out_channel,out_channel,out_channel],ks=3,pool=-1,activate=['relu','none'])

    def forward(self,x):
        y1=self.downsample(x)+self.side(x)
        y2=self.out(y1)
        return y1+y2

class ResNetBlock_small_pre(nn.Module):
    '''
        Basic pre-block for resnet-18(No downsampling)
    '''
    def __init__(self,nchannel,*args,**kwargs):
        super(ResNetBlock_small_pre,self).__init__()
        self.main=ConvBaseBlock([nchannel,nchannel,nchannel],ks=3,pool=-1,activate=['relu','none'])

    def forward(self,x):
        y=self.main(x)
        return x+y

class ResNetBlock_big(nn.Module):
    '''
    Basic block for resnet-34
    '''
    def __init__(self,in_channel,out_channel,nout=1,*args,**kwargs):
        super(ResNetBlock_big,self).__init__()
        self.downsample=nn.Sequential(
            ConvLayer(in_channel,out_channel,stride=(2,2),ks=3),
            ConvLayer(out_channel,out_channel),
        )
        self.side=nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.out=nn.Sequential(*[ResNetBlock_small_pre(out_channel) for i in range(nout)])

    def forward(self,x):
        y1=self.downsample(x)+self.side(x)
        y2=self.out(y1)
        return y1+y2

class ResNetBlock_large_bottleneck_downsample(nn.Module):
    '''
    Basic block for resnet-50
    '''
    def __init__(self,in_channel,mid_channel,out_channel,*args,**kwargs):
        super(ResNetBlock_large_bottleneck_downsample,self).__init__()
        self.downsample=nn.Sequential(
            ConvLayer(in_channel,mid_channel,ks=1),
            ConvLayer(mid_channel,mid_channel,stride=(2,2),ks=3),
            ConvLayer(mid_channel, out_channel, ks=1,activation='none'),
        )
        self.side=nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.act=nn.ReLU(inplace=True)

    def forward(self,x):
        y1=self.downsample(x)+self.side(x)
        y2=self.act(y1)
        return y1+y2

class ResNetBlock_large_bottleneck(nn.Module):
    '''
    Basic block for resnet-50
    '''
    def __init__(self,in_channel,mid_channel,out_channel,*args,**kwargs):
        super(ResNetBlock_large_bottleneck,self).__init__()
        self.main=nn.Sequential(
            ConvLayer(in_channel,mid_channel,ks=1),
            ConvLayer(mid_channel,mid_channel,stride=1,ks=3),
            ConvLayer(mid_channel, out_channel, ks=1,activation='none'),
        )
        self.act=nn.ReLU(inplace=True)

    def forward(self,x):
        y1=self.main(x)+x
        y2=self.act(y1)
        return y1+y2

class ResNetBlock_large(nn.Module):
    '''
    Basic block for resnet-50
    '''
    def __init__(self,in_channel,mid_channel,out_channel,nout,*args,**kwargs):
        super(ResNetBlock_large,self).__init__()
        self.downsample=ResNetBlock_large_bottleneck_downsample(in_channel,mid_channel,out_channel)
        self.out=nn.Sequential(
            *[ResNetBlock_large_bottleneck(out_channel,mid_channel,out_channel) for i in range(nout)]
        )

    def forward(self,x):
        y1=self.downsample(x)
        y2=self.out(y1)
        return y1+y2

class ResNetFactory(Factory):
    def __init__(self):
        super(ResNetFactory,self).__init__()
        self.register_dict({
            'resnet_small':ResNetBlock_small,
            'resnet_small_pre':ResNetBlock_small_pre,
            'resnet_big':ResNetBlock_big,
            'resnet_large':ResNetBlock_large,
        })
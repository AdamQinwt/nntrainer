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

class ResNetBlock_large(nn.Module):
    '''
    Basic block for resnet-50
    '''
    def __init__(self,in_channel,out_channel,*args,**kwargs):
        super(ResNetBlock_large,self).__init__()
        self.downsample=nn.Sequential(
            ConvLayer(in_channel,in_channel,stride=(2,2),ks=(1,1)),
            ConvLayer(in_channel,in_channel),
            ConvLayer(in_channel, out_channel, stride=(2, 2), ks=(1, 1),activation='none'),
        )
        self.side=nn.Sequential(
            ConvLayer(in_channel,out_channel,stride=(2,2),ks=(1,1))
        )
        self.out=ConvBaseBlock([out_channel,in_channel,in_channel,out_channel],ks=[1,3,1],pool=-1,activate=['relu','relu','none'])

    def forward(self,x):
        y1=self.downsample(x)+self.side(x)
        y2=self.out(y1)
        return y1+y2

class ResNetFactory(Factory):
    def __init__(self):
        super(ResNetFactory,self).__init__()
        self.register_dict({
            'resnet_small':ResNetBlock_small,
            'resnet_small_pre':ResNetBlock_small_pre,
            'resnet_large':ResNetBlock_large,
        })
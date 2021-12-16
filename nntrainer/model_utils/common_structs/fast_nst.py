import torch
import torch.nn as nn
from nntrainer.model_utils.trivial import UnitLayer,ActivationLayer,AggressiveLayer

class TransformerNetwork(UnitLayer):
    def __init__(self,channel=3,tanh=None):
        super(TransformerNetwork,self).__init__()
        conv_block=[
            ConvLayer(channel,32,9,1,activation='relu'),
            ConvLayer(32,64,3,2,activation='relu'),
            ConvLayer(64,128,3,2,activation='relu'),
        ]
        res_block=[ResLayers(128,3,5)]
        deconv_block=[
            DeConvLayer(128,64,3,2,1,activation='relu'),
            DeConvLayer(64,32,3,2,1,activation='relu'),
        ]
        if tanh:
            final_block=[
                ConvLayer(32,1,9,1,norm='none',activation='tanh'),
                AggressiveLayer(tanh)
            ]
        else:
            final_block=[
                ConvLayer(32,channel,9,1,norm='none')
            ]
        self.main=nn.Sequential(*(conv_block+res_block+deconv_block+final_block))

class ConvLayer(UnitLayer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,norm='instance',activation='none'):
        super(ConvLayer,self).__init__()
        pad=nn.ReflectionPad2d(kernel_size>>1)
        conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        if norm=='instance':
            norm=nn.InstanceNorm2d(out_channels,affine=True)
        elif norm=='batch':
            norm=nn.BatchNorm2d(out_channels,affine=True)
        else:
            norm=None
        act=ActivationLayer(activation)
        blocks=[pad,conv,norm,act] if norm else [pad,conv,act]
        self.main=nn.Sequential(*blocks)

class DeConvLayer(UnitLayer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,output_padding,norm='instance',activation='none'):
        super(DeConvLayer,self).__init__()
        conv=nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,kernel_size>>1,output_padding)
        if norm=='instance':
            norm=nn.InstanceNorm2d(out_channels,affine=True)
        elif norm=='batch':
            norm=nn.BatchNorm2d(out_channels,affine=True)
        else:
            norm=None
        act=ActivationLayer(activation)
        blocks=[conv,norm,act] if norm else [conv,act]
        self.main=nn.Sequential(*blocks)

class ResLayer(nn.Module):
    def __init__(self,num_channels,kernel_size):
        super(ResLayer,self).__init__()
        self.main=nn.Sequential(
            ConvLayer(num_channels,num_channels,kernel_size,stride=1),
            nn.ReLU(),
            ConvLayer(num_channels,num_channels,kernel_size,stride=1),
        )
    def forward(self,x):
        return self.main(x)+x

class ResLayers(UnitLayer):
    def __init__(self,num_channels,kernel_size,num):
        super(ResLayers,self).__init__()
        blocks=[ResLayer(num_channels,kernel_size) for i in range(num)]
        self.main=nn.Sequential(*blocks)
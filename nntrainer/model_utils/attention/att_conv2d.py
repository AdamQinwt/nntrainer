import torch
import torch.nn as nn
from nntrainer.model_utils.convbase import ConvLayer,ConvBaseBlock
from nntrainer.model_utils.model_parser import Factory
from nntrainer.model_utils.trivial import UnitLayer,EmptyLayer

from .squeeze_extraction import SELayer
from .residual_attention import ResAttBlock
from .bam import BAMBlock
from .cbam import CBAMBlock

class ResNetAttentionBlock_small(nn.Module):
    '''
        Basic block for resnet-18
    '''
    def __init__(self,attention_block,in_channel,out_channel,*args,**kwargs):
        super(ResNetAttentionBlock_small,self).__init__()
        self.downsample=nn.Sequential(
            ConvLayer(in_channel,out_channel,stride=(2,2)),
            ConvLayer(out_channel,out_channel,activation='none'),
            attention_block(*args,**kwargs)
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

class ResNetAttentionBlock_small_pre(nn.Module):
    '''
        Basic pre-block for resnet-18(No downsampling)
    '''
    def __init__(self,attention_block,nchannel,*args,**kwargs):
        super(ResNetAttentionBlock_small_pre,self).__init__()
        self.main=nn.Sequential(
            ConvBaseBlock([nchannel,nchannel,nchannel],ks=3,pool=-1,activate=['relu','none']),
            attention_block(*args, **kwargs)
        )

    def forward(self,x):
        y=self.main(x)
        return x+y

class ResNetAttentionBlock_large(nn.Module):
    '''
    Basic block for resnet-50
    '''
    def __init__(self,attention_block,in_channel,out_channel,*args,**kwargs):
        super(ResNetAttentionBlock_large,self).__init__()
        self.downsample=nn.Sequential(
            ConvLayer(in_channel,in_channel,ks=(1,1)),
            ConvLayer(in_channel,in_channel),
            ConvLayer(in_channel, out_channel, stride=(2, 2), ks=(1, 1),activation='none'),
            attention_block(*args, **kwargs)
        )
        self.side=nn.Sequential(
            ConvLayer(in_channel,out_channel,stride=(2,2),ks=(1,1))
        )
        self.out=ConvBaseBlock([out_channel,in_channel,in_channel,out_channel],ks=[1,3,1],pool=-1,activate=['relu','relu','none'])

    def forward(self,x):
        y1=self.downsample(x)+self.side(x)
        y2=self.out(y1)
        return y1+y2

class ResNetAttentionBlock(UnitLayer):
    def __init__(self,factory,resblock_type,attblock_type,*args,**kwargs):
        super(ResNetAttentionBlock,self).__init__()
        if resblock_type=='small_pre':
            self.main=ResNetAttentionBlock_small_pre(factory[attblock_type],*args,**kwargs)
        elif resblock_type=='small':
            self.main=ResNetAttentionBlock_small(factory[attblock_type],*args,**kwargs)
        elif resblock_type=='large':
            self.main=ResNetAttentionBlock_large(factory[attblock_type],*args,**kwargs)
        else:
            raise ValueError(f'{resblock_type} does NOT exist!!\nPossible types are: ["small_pre","small","large"]')

class AttConv2dResNetFactory(Factory):
    def __init__(self):
        '''
        bam: BAM. Possible args include: activation,gate_channel(*),reduction_ratio,dilation_conv_num,num_layers
        cbam: CBAM. Possible args include: activation,gate_channel(*),dim
        res_att: Residual attention layer. Possible args include: nchannels(*),orig_size(*),nfold,bottom_nconv,trunk_nconv
        se: Channel-wise attention layer. Possible args include: channel(*),reduction
        '''
        super(AttConv2dResNetFactory,self).__init__()
        self.register_dict({
            'attention_conv2d_resnet':ResNetAttentionBlock,
            'se':SELayer,
            'bam':BAMBlock,
            'cbam':CBAMBlock,
            'res_att':ResAttBlock,
            'none':EmptyLayer,
        })
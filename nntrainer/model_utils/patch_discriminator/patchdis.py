import torch
import torch.nn as nn
from nntrainer.model_utils.view import View
from nntrainer.model_utils.trivial import UnitLayer

class PatchSplitMerge(UnitLayer):
    def __init__(self,main_modules,stride,psize=None,in_channels=1,mid_channels=512):
        super(PatchSplitMerge,self).__init__()
        if psize is None:
            psize=stride
        starter=nn.Conv2d(in_channels,mid_channels,psize,stride,padding=0)
        self.main=nn.Sequential(starter,main_modules,nn.AdaptiveAvgPool2d((1,1)),View([-2,-1]))

class ClassicDisBlock(UnitLayer):
    def __init__(self,shapes):
        super(ClassicDisBlock,self).__init__()
        l=[]
        for idx in range(len(shapes)-1):
            l.append(nn.Conv2d(shapes[idx],shapes[idx+1],kernel_size=1,stride=1,padding=0))
            l.append(nn.LeakyReLU(0.2,inplace=True))
        l.append(nn.Conv2d(shapes[idx],1,kernel_size=1,stride=1,padding=0))
        l.append(nn.Sigmoid())
        self.main=nn.Sequential(*l)
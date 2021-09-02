'''
SENet(Squeeze-and-Excitation Networks from CVPR 2018)
'''
import torch
import torch.nn as nn
from ..trivial import UnitLayer

class SELayer(nn.Module):
    '''
    Channel-wise attention layer.
    '''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # create channel-wise weights
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        return x * y

class SEBasicBlock(nn.Module):
    def __init__(self,in_channels,channels,ks,bn_track=True,reduction=16,is_residual=False,*args,**kwargs):
        super(SEBasicBlock,self).__init__()
        self.is_residual=is_residual
        layers=[
            nn.Conv2d(in_channels,channels,ks,1,padding=ks>>1,bias=False),
            nn.BatchNorm2d(channels,track_running_stats=bn_track),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, ks, 1, padding=ks >> 1, bias=False),
            nn.BatchNorm2d(channels,track_running_stats=bn_track),
            SELayer(channels,reduction)
        ]
        self.main=nn.Sequential(*layers)
    def forward(self,x):
        y=self.main(x)
        return x+y if self.is_residual else y

class SEBlock(UnitLayer):
    def __init__(self,nchannels,ks,pool=2,bn_track=True,reduction=16,is_residual=False,*args,**kwargs):
        super(SEBlock,self).__init__()
        layers=[]
        num_layers=len(nchannels)-1
        if not isinstance(ks,list):
            ks=[ks]*num_layers
        for i in range(num_layers):
            layers.append(SEBasicBlock(
                nchannels[i],nchannels[i+1],ks[i],bn_track,reduction,is_residual
            ))
            act = nn.ReLU(inplace=True)
            layers.append(act)
        if pool>1:
            layers.append(nn.MaxPool2d(pool,pool))
        self.main=nn.Sequential(*layers)
import torch.nn as nn
import torch

class ConvBaseBlock(nn.Module):
    def __init__(self,nchannels,ks,pool=2,activate=None,bn=True,bn_track=False,*args,**kwargs):
        super(ConvBaseBlock,self).__init__()
        layers=[]
        num_layers=len(nchannels)-1
        if activate and not isinstance(activate,list):
            activate=[activate]*num_layers
        if not isinstance(ks,list):
            ks=[ks]*num_layers
        for i in range(num_layers):
            a=activate[i] if activate else None
            layers.append(nn.Conv2d(
                nchannels[i],nchannels[i+1],ks[i],padding=ks[i]>>1
            ))
            if bn:
                layers.append(nn.BatchNorm2d(nchannels[i+1],track_running_stats=bn_track))
            if a == 'relu':
                act = nn.ReLU(inplace=True)
            elif a == 'sigmoid':
                act = nn.Sigmoid()
            elif a == 'tanh':
                act = nn.Tanh()
            else:
                act = None
            if act:
                layers.append(act)
        if pool>1:
            layers.append(nn.MaxPool2d(pool,pool))
        self.main=nn.Sequential(*layers)

    def forward(self,x):
        y=self.main(x)
        return y

class ResConvBaseBlock(nn.Module):
    def __init__(self,inchannel,reschannel,nlayer,ks,pool=2,activate=None,bn=True,bn_track=False,*args,**kwargs):
        super(ResConvBaseBlock,self).__init__()
        layers=[]
        if not isinstance(ks,list):
            ks=[ks]*(nlayer+1)
        self.pre=None if inchannel==reschannel else nn.Conv2d(
            inchannel,reschannel,ks[0],padding=ks[0]>>1
            )
        offset=1 if self.pre else 0
        num_layers=nlayer
        if activate and not isinstance(activate,list):
            activate=[activate]*num_layers
        for i in range(num_layers):
            a = activate[i] if activate else None
            layers.append(nn.Conv2d(
                reschannel,reschannel,ks[i+offset],padding=ks[i+offset]>>1
            ))
            if bn:
                layers.append(nn.BatchNorm2d(reschannel,track_running_stats=bn_track))
            if a == 'relu':
                act = nn.ReLU(inplace=True)
            elif a == 'sigmoid':
                act = nn.Sigmoid()
            elif a == 'tanh':
                act = nn.Tanh()
            else:
                act = None
            if act:
                layers.append(act)
        self.main=nn.Sequential(*layers)
        if pool>1:
            self.pool=nn.MaxPool2d(pool,pool)
        else:
            self.pool=None

    def forward(self,x):
        if self.pre:
            x=self.pre(x)
        y=self.main(x)+x
        y=self.pool(y)
        return y
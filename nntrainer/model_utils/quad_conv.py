from nntrainer.model_utils.trivial import UnitLayer
import torch
import torch.nn as nn

def get_square_indices(n):
    l=[]
    start_idx=0
    end_idx=n
    n1=n+1
    for i in range(n):
        l+=list(range(start_idx,end_idx))
        start_idx+=n1
        end_idx+=n
    return l

def calcNewLength(s,padding,kernel_size,stride=1):
    return ((s + 2 * padding - kernel_size )// stride) + 1

class ChannelQuadLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ChannelQuadLayer,self).__init__()
        n_fc_in=in_channels*(in_channels+3)
        n_fc_in>>=1
        self.idx=get_square_indices(in_channels)
        self.actual_channel=n_fc_in
        self.fc=nn.Conv2d(n_fc_in,out_channels,1,bias=False)
    def forward(self,x):
        bs,chn,width,height=x.size()
        area=width*height
        chn2=chn*chn
        a=x.permute(0,2,3,1).view(bs,area,chn)  # N(H*W)C
        x1=a.unsqueeze(-1)
        x2=a.unsqueeze(-2)
        y=torch.matmul(x1,x2).view(bs,area,chn2).permute(0,2,1).view(bs,chn2,width,height)[:,self.idx]
        y=torch.cat([x,y],1)
        out=self.fc(y)
        return out

class IntegratedChannelQuadLayer(UnitLayer):
    def __init__(self,in_channels,mid_channels,out_channels,stride=1,kernel_size=3,padding=None,bias=False,dilation=1,*args,**kwargs):
        super(IntegratedChannelQuadLayer,self).__init__()
        conv=nn.Conv2d(
            in_channels,mid_channels,
            kernel_size=kernel_size,stride=stride,
            padding=padding if padding else kernel_size>>1,
            bias=bias,dilation=dilation
        )
        channel_qconv=ChannelQuadLayer(mid_channels,out_channels)
        self.main=nn.Sequential(conv,channel_qconv)


class SpatialQuadLayer(nn.Module):
    def __init__(self,n_channels,ks=3,stride=1,padding=None,chn_share_weight=False,chn_expand=1):
        super(SpatialQuadLayer,self).__init__()
        qlen=ks*ks
        n_fc_in=qlen*(qlen+3)
        n_fc_in>>=1
        if padding is None:
            padding=ks>>1
        self.idx=get_square_indices(qlen)
        self.fc=nn.Conv1d(n_fc_in,chn_expand,kernel_size=1)
        self.qlen=qlen
        self.stride=stride
        self.padding=padding
        self.unfold=nn.Unfold(kernel_size=ks,dilation=1,padding=padding,stride=stride)
        self.ks=ks
        self.out_chn=n_channels*chn_expand
    def forward(self,x):
        bs,chn,width,height=x.size()
        area=width*height
        width_ = calcNewLength(width,self.padding,self.ks,self.stride)
        height_ = calcNewLength(height,self.padding,self.ks,self.stride)
        windows=self.unfold(x).transpose(1,2).view(bs,-1,chn,self.ks*self.ks).transpose(1,2)
        num_kernel=windows.size(2)
        x1=windows.unsqueeze(-1)
        x2=windows.unsqueeze(-2)
        y=torch.matmul(x1,x2).view(bs,chn,num_kernel,-1)[...,self.idx].permute(0,3,1,2) # N(IN)C(H*W)
        y=torch.cat([windows.permute(0,3,1,2),y],dim=1).view(bs,-1,chn*area)
        out=self.fc(y).view(bs,self.out_chn,height_,width_)
        return out
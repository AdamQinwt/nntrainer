import torch
import torch.nn as nn
from nntrainer.model_utils.trivial import ActivationLayer,UnitLayer

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

class IdenticalQuadLayer(nn.Module):
    '''
    y=fc([x*x^T,x])
    '''
    def __init__(self,in_n,out_n):
        super(IdenticalQuadLayer,self).__init__()
        n_fc_in=in_n*(in_n+3)
        n_fc_in>>=1
        self.idx=get_square_indices(in_n)
        self.fc=nn.Linear(n_fc_in,out_n,bias=False)
    def forward(self,x):
        bs=x.size(0)
        x1=x.unsqueeze(2)
        x2=x.unsqueeze(1)
        y=torch.matmul(x1,x2).view(bs,-1)[:,self.idx]
        y=torch.cat([x,y],-1)
        out=self.fc(y)
        return out

class NonIdenticalQuadLayer(nn.Module):
    '''
    DO REMEMBER to squeeze the last dimension if your out_n=1.
    '''
    def __init__(self,in_n1,in_n2=None,out_n=1):
        super(NonIdenticalQuadLayer,self).__init__()
        if in_n2 is None:
            in_n2=in_n1
        self.fc=nn.Linear(in_n1*in_n2+in_n1+in_n2,out_n,bias=False)
    def forward(self,x,y=None):
        bs=x.size(0)
        if y is None:
            y=x
        x1=x.unsqueeze(2)
        x2=y.unsqueeze(1)
        x12=torch.matmul(x1,x2).view(bs,-1)
        y=torch.cat([x,y,x12],-1)
        out=self.fc(y)
        return out

class QuadFCLayer(UnitLayer):
    def __init__(self, in_n1,in_n2=None,is_identical=True,out_n=1,bn=True,bn_track=True,activate='relu',dropout=0.0,activation_args=None,*args,**kwargs):
        super(QuadFCLayer, self).__init__()
        if activation_args is None: activation_args={}
        if in_n2 is None:
            in_n2=in_n1
        fc=IdenticalQuadLayer(in_n1,out_n) if is_identical else NonIdenticalQuadLayer(in_n1,in_n2,out_n)
        act=ActivationLayer(activate,**activation_args)
        layers=[
            fc,
            nn.BatchNorm1d(out_n,track_running_stats=bn_track)
        ] if bn else [fc]
        if act:
            layers.append(act)
        if not dropout:
            dropout=0.0
        if dropout>0:
            layers.append(nn.Dropout(dropout))
        self.main=nn.Sequential(*layers)


class IdenticalQuadFCBlock(UnitLayer):
    def __init__(self, shapes,activate=None,activate_args=None,dropout=None,*args,**kwargs):
        super(IdenticalQuadFCBlock, self).__init__()
        main=[]
        num_layers=len(shapes)-1
        if activate and not isinstance(activate,list):
            activate=[activate]*num_layers
        if dropout and not isinstance(dropout,list):
            dropout=[dropout]*num_layers
        if activate_args and not isinstance(activate_args,list):
            activate_args=[activate_args]*num_layers
        for i in range(num_layers):
            main.append(QuadFCLayer(in_n1=shapes[i],out_n=shapes[i+1],
                                activate=activate[i] if activate else None,
                                dropout=dropout[i] if dropout else None,
                                activation_args=activate_args[i] if activate_args else None,
                                *args,**kwargs))
        self.main=nn.Sequential(*main)
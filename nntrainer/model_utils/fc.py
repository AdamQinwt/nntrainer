import torch
import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, in_n,out_n,bn=True,bn_track=True,activate='relu',dropout=0.0):
        super(FCLayer, self).__init__()
        if activate=='relu':
            act=nn.ReLU(inplace=True)
        elif activate=='sigmoid':
            act=nn.Sigmoid()
        elif activate=='tanh':
            act=nn.Tanh()
        else:
            act=None
        layers=[
            nn.Linear(in_n,out_n),
            nn.BatchNorm1d(out_n,track_running_stats=bn_track)
        ] if bn else [nn.Linear(in_n,out_n)]
        if act:
            layers.append(act)
        if dropout>0:
            layers.append(nn.Dropout(dropout))
        self.main=nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class FCBlock(nn.Module):
    def __init__(self, shapes,bn=True,bn_track=True,activate='relu'):
        super(FCBlock, self).__init__()
        main=[]
        num_layers=len(shapes)-1
        for i in range(num_layers):
            main.append(FCLayer(shapes[i],shapes[i+1],bn,bn_track,activate))
        self.main=nn.Sequential(*main)

    def forward(self, x):
        return self.main(x)

class FCBlock_v2(nn.Module):
    def __init__(self, shapes,bn=False,bn_track=True,activate=None):
        super(FCBlock_v2, self).__init__()
        main=[]
        num_layers=len(shapes)-1
        if activate and not isinstance(activate,list):
            activate=[activate]*num_layers
        for i in range(num_layers):
            main.append(FCLayer(shapes[i],shapes[i+1],bn,bn_track,activate[i] if activate else None))
        self.main=nn.Sequential(*main)

    def forward(self, x):
        return self.main(x)
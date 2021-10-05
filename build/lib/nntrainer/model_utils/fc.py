import torch
import torch.nn as nn
from nntrainer.model_utils.trivial import ActivationLayer,UnitLayer

class FCLayer(UnitLayer):
    def __init__(self, in_n,out_n,bn=True,bn_track=True,activate='relu',dropout=0.0,activation_args=None,*args,**kwargs):
        super(FCLayer, self).__init__()
        if activation_args is None: activation_args={}
        act=ActivationLayer(activate,**activation_args)
        layers=[
            nn.Linear(in_n,out_n),
            nn.BatchNorm1d(out_n,track_running_stats=bn_track)
        ] if bn else [nn.Linear(in_n,out_n)]
        if act:
            layers.append(act)
        if not dropout:
            dropout=0.0
        if dropout>0:
            layers.append(nn.Dropout(dropout))
        self.main=nn.Sequential(*layers)

class FCBlock_v2(UnitLayer):
    def __init__(self, shapes,activate=None,activate_args=None,dropout=None,*args,**kwargs):
        super(FCBlock_v2, self).__init__()
        main=[]
        num_layers=len(shapes)-1
        if activate and not isinstance(activate,list):
            activate=[activate]*num_layers
        if dropout and not isinstance(dropout,list):
            dropout=[dropout]*num_layers
        if activate_args and not isinstance(activate_args,list):
            activate_args=[activate_args]*num_layers
        for i in range(num_layers):
            main.append(FCLayer(shapes[i],shapes[i+1],
                                activate=activate[i] if activate else None,
                                dropout=dropout[i] if dropout else None,
                                activation_args=activate_args[i] if activate_args else None,
                                *args,**kwargs))
        self.main=nn.Sequential(*main)
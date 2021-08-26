import torch
import torch.nn as nn

class EmptyLayer(nn.Module):
    """Placeholder"""
    def __init__(self,*args,**kwargs):
        super(EmptyLayer, self).__init__()
        self.infolist=args
        self.infodict=kwargs
        for k,v in kwargs.items():
            self.__setattr__(k,v)

class UnitLayer(nn.Module):
    '''
    models that only need one main part
    could be a sequential or unit block
    '''
    def __init__(self):
        super(UnitLayer,self).__init__()
    def forward(self,*args,**kwargs):
        return self.main(*args,**kwargs)
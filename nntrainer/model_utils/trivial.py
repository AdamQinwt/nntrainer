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

class ActivationLayer(UnitLayer):
    '''
    activation layer from name
    '''
    def __init__(self,act,*args,**kwargs):
        super(ActivationLayer,self).__init__()
        act_dict={'none':EmptyLayer,'sigmoid':nn.Sigmoid,'relu':nn.ReLU,'tanh':nn.Tanh,'lrelu':nn.LeakyReLU}
        if act in act_dict.keys():
            self.main=act_dict[act](*args,**kwargs)
        else:
            self.main=eval(f'nn.{act}')(*args,**kwargs)
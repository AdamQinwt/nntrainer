import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape,*args,**kwargs):
        super(View, self).__init__()
        self.shape=shape

    def forward(self, x):
        bs=x.size(0)
        shape=[]
        for s in self.shape:
            if s==-2:
                shape.append(bs)
            else:
                shape.append(s)
        return x.view(*shape)

class Flatten(View):
    def __init__(self,*args,**kwargs):
        super(Flatten, self).__init__([-2,-1])

class Cat(nn.Module):
    def __init__(self, dim=-1,*args,**kwargs):
        super(Cat, self).__init__()
        self.dim=dim

    def forward(self, x):
        return torch.cat(x,dim=self.dim)

class Squeeze(nn.Module):
    '''
    squeeze or unsqueeze
    '''
    def __init__(self, dim=-1,direction=True,*args,**kwargs):
        '''
        squeeze or unsqueeze
        :param dim: dim list
        :param direction: true for squeeze; false for unsqueeze
        :param args:
        :param kwargs:
        '''
        super(Squeeze, self).__init__()
        self.dim=dim
        self.dir=direction

    def forward(self, x):
        dim=self.dim
        direction=self.dir
        if isinstance(dim,list):
            for d in dim:
                x=x.squeeze(d) if direction else x.unsqueeze(d)
        else:
            x = x.squeeze(dim) if direction else x.unsqueeze(dim)
        return x
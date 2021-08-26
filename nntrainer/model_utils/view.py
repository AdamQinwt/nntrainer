import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
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

class Cat(nn.Module):
    def __init__(self, dim=-1):
        super(Cat, self).__init__()
        self.dim=dim

    def forward(self, x):
        return torch.cat(x,dim=self.dim)
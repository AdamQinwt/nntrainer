from . import odesolver_adjoint as odesolver
import torch.nn as nn

class ODEBlock(nn.Module):
    '''
    An ode instance should be defined as ODEBlock(ODEFunction(params)),
    where ODEFunction has a forward function like forward(self,t,x) with t as the timestamp and x as the input.
    options should contain an Nt and method(from ['Euler', 'RK2', 'RK4'])
    '''
    def __init__(self, odefunc,options=None):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options=options if options else {'Nt':2,'method':'Euler'}

    def forward(self, x):
        out = odesolver(self.odefunc, x, self.options)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
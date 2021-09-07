'''
SENet(Squeeze-and-Excitation Networks from CVPR 2018)
'''
import torch
import torch.nn as nn
from ..trivial import UnitLayer
class SELayer(nn.Module):
    '''
    Channel-wise attention layer.
    Possible args include: channel(*),reduction
    '''
    def __init__(self, channel, reduction=16,*args,**kwargs):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # create channel-wise weights
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        return x * y
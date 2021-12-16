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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowPartition(nn.Module):
    def __init__(self,window_size):
        # window_size (int): window size
        super(WindowPartition,self).__init__()
        self.ws=window_size
    def forward(self,x):
        """
        Args:
            x: (B, H, W, C)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        window_size=self.ws
        B, H, W, C = x.size()
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

class WindowPartitionReverse(nn.Module):
    def __init__(self,window_size):
        # window_size (int): window size
        super(WindowPartitionReverse,self).__init__()
        self.ws=window_size
    def forward(self,windows,H,W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        window_size=self.ws
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
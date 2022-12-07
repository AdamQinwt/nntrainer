import numpy as np
import torch

def one_hot(idx,n=None):
    if isinstance(idx,np.ndarray):
        if n is None:
            n=np.max(idx)
        a=np.zeros([len(idx),n],dtype=np.float32)
        a[:,idx]=1.0
    elif isinstance(idx,torch.Tensor):
        if n is None:
            n=torch.max(idx)
        a=torch.zeros([len(idx),n],dtype=torch.float32,device=idx.device)
        a[:,idx]=1.0
    return a
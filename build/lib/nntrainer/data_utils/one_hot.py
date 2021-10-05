import numpy as np

def one_hot(idx,n):
    a=np.zeros([n],dtype=np.float32)
    a[idx]=1.0
    return a
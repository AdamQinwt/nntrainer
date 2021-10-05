from torch.utils.data import Dataset
import numpy as np

def to_bin(x,n):
    l=np.zeros([n],dtype=np.float32)
    i=n-1
    while x>0:
        l[i]=x&1
        x>>=1
        i-=1
    return l

class Xor(Dataset):
    '''
    n-bit xor dataset
    '''
    def __init__(self, nbit=1):
        self.nbit=nbit
        num=1<<nbit
        self.num=num
        self.x=np.stack([to_bin(x,nbit) for x in range(num)],axis=0)
        y=np.sum(self.x,axis=-1)
        self.y=np.array(np.array(y,dtype=int)&1,dtype=np.float32)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.x[index],self.y[index]
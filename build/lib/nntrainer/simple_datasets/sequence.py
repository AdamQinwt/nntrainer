from torch.utils.data import Dataset
import numpy as np
import h5py

class SequenceDataset(Dataset):
    '''
    y=sin(w*x) dataset
    '''
    def __init__(self, path,xlen,ylen):
        self.xlen=xlen
        self.ylen=ylen
        with h5py.File(path,'r') as fin:
            x=np.array(fin['x'],dtype=np.float32)
        self.x=x
        self.xylen=xlen+ylen
        self.n=len(x)-self.xylen

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.x[index:index+self.xlen],self.y[index+self.xlen:index+self.xylen]
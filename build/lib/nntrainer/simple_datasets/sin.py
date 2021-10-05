from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class SinDataset(Dataset):
    '''
    y=sin(w*x) dataset
    '''
    def __init__(self, w=1.0,n=1024,x_min=-1.0,x_max=1.0):
        self.w=w    # frequency
        self.n=n    # number of points sampled
        x=np.linspace(x_min,x_max,n,dtype=np.float32)
        y=np.sin(w*x)
        self.x=x
        self.y=y

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.x[index],self.y[index]

class SummedSinDataset(Dataset):
    '''
    y=x0+/sigma sin(w*x) dataset
    '''
    def __init__(self, w,a,x0=0.0,n=1024,x_min=-1.0,x_max=1.0):
        self.w=w    # frequency
        self.a=a    # amplitude
        self.n=n    # number of points sampled
        x=np.linspace(x_min,x_max,n,dtype=np.float32)
        y=sum([a[i]*np.sin(w[i]*x) for i in range(len(w))])+x0
        self.x=x
        self.y=y

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.x[index],self.y[index]

class SequenceSinDataset(Dataset):
    '''
    dataset for predicting y(0) from y(-t),...,y(-1)
    y=sin(x)
    '''
    def __init__(self, w=1.0,n=1024,t=32,x_min=-1.0,x_max=1.0):
        self.w=w    # frequency
        self.n=n    # number of points sampled
        self.t=t    # time span required
        x=np.linspace(x_min,x_max,n,dtype=np.float32)
        y=np.sin(w*x)
        self.x=x
        self.y=y

    def __len__(self):
        return self.n-self.t

    def __getitem__(self, index):
        t=self.t
        return self.y[index:index+t],self.y[index+t]

class SequenceSummedSinDataset(Dataset):
    '''
    dataset for predicting y(0) from y(-t),...,y(-1)
    y=x0+/sigma sin(w*x)
    '''
    def __init__(self, w,a,x0=0.0,n=1024,t=32,x_min=-1.0,x_max=1.0):
        self.w=w    # frequency
        self.a=a    # amplitude
        self.n=n    # number of points sampled
        self.t=t    # time span required
        x=np.linspace(x_min,x_max,n,dtype=np.float32)
        y=x0+sum([a[i]*np.sin(w[i]*x) for i in range(len(w))])
        self.x=x
        self.y=y

    def __len__(self):
        return self.n-self.t-1

    def __getitem__(self, index):
        t=self.t
        return self.y[index:index+t],self.y[index+t]

if __name__=='__main__':
    s=SequenceSummedSinDataset(w=[3.14159,5],a=[-1,2],x0=-9)
    for x,y in s:
        print(x)
        print(y)
        break
    x=s.x
    y=s.y
    plt.plot(x,y)
    plt.show()
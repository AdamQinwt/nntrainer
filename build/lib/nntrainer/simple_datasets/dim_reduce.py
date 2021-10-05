from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import random
from .one_hot import one_hot
from .random_generator import default_random_generators

class BaseDimReduceDataset(Dataset):
    def __init__(self,generators,generator_params,ns,dim=2,generator_dict=None):
        if not generator_dict:
            generator_dict=default_random_generators
        self.dim=dim
        n_classes=len(ns)
        self.n_classes=n_classes
        self.ns=ns
        if not isinstance(generators,list):
            generators=[generators]*n_classes
            generator_params=[generator_params]*n_classes
        generators=[default_random_generators[f'{x}'] for x in generators]
        x=[]
        y=[]
        y_cls=[]
        idx=[[] for i in range(n_classes)]
        for i in range(n_classes):
            for j in range(ns[i]):
                x.append(np.array(generators[i](dim,**generator_params[i]),dtype=np.float32))
                y.append(one_hot(i,n_classes))
                y_cls.append(i)
        self.x=np.stack(x)
        self.y=np.stack(y)
        self.y_cls=np.array(y_cls,dtype=np.int)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index],self.y[index],self.y_cls[index]
    def plot(self):
        colors=['red','blue','yellow','green']
        for i,x in enumerate(self.x):
            plt.scatter(*list(x),c=colors[self.y_cls[i]])

if __name__=='__main__':
    ds=BaseDimReduceDataset(
        ['uniform','uniform','norm'],
        [{'a':[-3,2],'b':[-1,4]},{'a':[-1,-1],'b':[3,8]},{'mean':[5,6],'cov':[1,3]}],
        [10,20,50],
        2,
        default_random_generators
    )
    ds.plot()
    plt.show()
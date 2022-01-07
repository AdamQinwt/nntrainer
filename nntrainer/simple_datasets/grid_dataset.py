from torch.utils.data import Dataset
import numpy as np

class Grid2DDataset(Dataset):
    '''
    2-dim grid dataset
    '''
    def __init__(self, data_ranges,data_stride=.1):
        self.ndim=2
        if isinstance(data_ranges,float) or isinstance(data_ranges,int):
            data_ranges=[[-data_ranges,data_ranges],[-data_ranges,data_ranges]]
        elif isinstance(data_ranges[0],float) or isinstance(data_ranges[0],int):
            data_ranges=[data_ranges,data_ranges]
        if isinstance(data_stride,float) or isinstance(data_stride,int):
            data_stride=[data_stride,data_stride]
        num=[]
        for i in range(self.ndim):
            dr=data_ranges[i]
            dr=dr[1]-dr[0]
            ds=data_stride[i]
            num.append(int(dr//ds+1))
        self.num=num
        self.data_range=data_ranges
        self.data_stride=data_stride
        # print(len(self))

    def __len__(self):
        return self.num[0]*self.num[1]

    def __getitem__(self, index):
        vi=index//self.num[0]
        hi=index-self.num[0]*vi
        d=[self.data_range[0][0]+vi*self.data_stride[0],self.data_range[1][0]+hi*self.data_stride[1]]
        return np.array(d,dtype=np.float32),vi,hi
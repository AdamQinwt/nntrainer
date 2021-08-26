from torchvision.datasets import MNIST
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from .one_hot import one_hot
import numpy as np
from PIL import Image
from random import randint

def get_transform(is_train=True):
    """ get transformation for the dataset
    :param is_train: training set or valid set
    :return transforms
    """
    transforms = []
    #mean, std = get_mean_std(args)
    norm=T.Normalize((0.5,), (0.5,))
    if is_train:
        transforms.extend([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            norm
        ])
    else:
        transforms.extend([
            T.ToTensor(),
            norm
        ])
    return T.Compose(transforms)

def load_mnist(root_path,train_bs=256,valid_bs=512):
    train_dataset=MNIST(f'{root_path}/train',train=True,transform=get_transform(True),download=False)
    valid_dataset=MNIST(f'{root_path}/valid',train=False,transform=get_transform(False),download=False)
    train_loader=DataLoader(train_dataset,train_bs,shuffle=True)
    valid_loader=DataLoader(valid_dataset,valid_bs,shuffle=False)
    return train_loader,valid_loader

class SubsetMnist(Dataset):
    def __init__(self,raw_dataset,num_list=None,transform=None,need_onehot=False):
        idx=[]
        remap={x:i for i,x in enumerate(num_list)}
        for i, y in enumerate(raw_dataset.targets):
            if y in num_list:
                idx.append(i)
        self.data=raw_dataset.data[idx]
        targets=raw_dataset.targets[idx]
        self.targets=np.array([remap[t.item()] for t in targets],dtype=np.float32)
        if need_onehot:
            self.onehot=np.stack([one_hot(remap[t.item()],len(num_list)) for t in targets])
        else:
            self.onehot=None
        self.remap=remap
        self.num_list=num_list
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index] if self.transform is None else self.transform(Image.fromarray(self.data[index].numpy(), mode='L')),\
               self.targets[index],self.onehot[index] if self.onehot else self.targets[index]
    def sample(self):
        index=randint(0,len(self.data))
        return self[index]


def load_mnist_subset(root_path,num_list=None,train_bs=256,valid_bs=512,need_onehot=False):
    '''
    create subset dataloaders of a mnist subset
    :param root_path: root path to mnist
    :param num_list: list of digits needed. None for 'all'
    :param train_bs: training batch size
    :param valid_bs: validation batch size
    :param need_onehot: whether one-hot vectors are required
    :return: train_loader,valid_loader,train_dataset,valid_dataset
    '''
    train_dataset=MNIST(f'{root_path}/train',train=True,transform=get_transform(True),download=False)
    train_dataset=SubsetMnist(train_dataset,num_list,get_transform(True),need_onehot)
    valid_dataset=MNIST(f'{root_path}/valid',train=False,transform=get_transform(False),download=False)
    valid_dataset = SubsetMnist(valid_dataset, num_list,get_transform(False), need_onehot)
    train_loader=DataLoader(train_dataset,train_bs,shuffle=True)
    valid_loader=DataLoader(valid_dataset,valid_bs,shuffle=False)
    return train_loader,valid_loader,train_dataset,valid_dataset
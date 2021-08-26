import numpy as np
from torch.utils.data import DataLoader

def load_dataset(dataset,bs,num_workers=16,*args,**kwargs):
    loaders = DataLoader(
        dataset=dataset,
        batch_size=bs,
        num_workers=num_workers,
        **kwargs
    )
    return loaders

def load_train_dataset(dataset,bs,num_workers=16):
    loaders = DataLoader(
        dataset=dataset,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    return loaders

def load_valid_dataset(dataset,bs,num_workers=16):
    loaders = DataLoader(
        dataset=dataset,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    return loaders
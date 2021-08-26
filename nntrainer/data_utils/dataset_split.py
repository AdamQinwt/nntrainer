import numpy as np
from torch.utils.data import DataLoader,SubsetRandomSampler

def split_dataset(dataset,rsplit):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(rsplit * dataset_size))
    np.random.shuffle(indices)
    indices= [indices[split:], indices[:split]]
    samplers=[SubsetRandomSampler(index) for index in indices]
    return samplers

def load_split_dataset(dataset,rsplit,bs=None,num_workers=16):
    samplers=split_dataset(dataset,rsplit)
    loaders = [DataLoader(
        dataset=dataset,
        batch_size=bs[i],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        sampler=sampler
    ) for i,sampler in enumerate(samplers)]

    return loaders
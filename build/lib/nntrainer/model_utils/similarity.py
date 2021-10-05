import torch
import torch.nn as nn
import torch.nn.functional as F
class CosSimilarity(nn.Module):
    def __init__(self):
        '''
        Cos similarity.
        scale invariant.
        0-1
        '''
        super(CosSimilarity, self).__init__()

    def forward(self, a,b,dim=-1):
        r=F.cosine_similarity(a,b,dim)+1
        return r/2

class EuclideanDistance(nn.Module):
    def __init__(self):
        '''
        Euclidean Distance
        scale variant.
        '''
        super(EuclideanDistance, self).__init__()

    def forward(self, a,b,dim=-1):
        c=a-b
        c=c.norm(p=2,dim=dim)/c.size(dim)
        return c

class MeanDistance(nn.Module):
    def __init__(self):
        '''
        Mean Distance
        scale variant.
        '''
        super(MeanDistance, self).__init__()

    def forward(self, a,b,dim=-1):
        c=a-b
        c=(c**2).mean(dim=dim).sqrt()
        return c

class PearsonCorrelation(nn.Module):
    def __init__(self):
        '''
        Pearson similarity.
        scale invariant.
        0-1
        '''
        super(PearsonCorrelation, self).__init__()

    def forward(self, a,b,dim=-1):
        a=a-a.mean(dim)
        b=b-b.mean(dim)
        r = F.cosine_similarity(a, b, dim) + 1
        return r/2

if __name__=='__main__':
    a=torch.tensor([[1,-1],[1,1]],dtype=torch.float32)
    b=torch.tensor([[-1,1],[0,-1]],dtype=torch.float32)
    sim=MeanDistance()
    r=sim(a,b)
    print(r)

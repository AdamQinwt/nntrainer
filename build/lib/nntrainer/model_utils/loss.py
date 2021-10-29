import torch
import torch.nn as nn

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, pred,ans):
        diff=pred-ans
        diff=diff.pow(2).mean()
        return diff

class RMSEFromX(RMSE):
    def __init__(self,x):
        super(RMSEFromX,self).__init__()
        self.x=x
    def forward(self,y):
        return super(RMSEFromX,self).forward(y,self.x)

class RMSEFrom0(RMSEFromX):
    def __init__(self):
        super(RMSEFrom0,self).__init__(0)
class RMSEFrom1(RMSEFromX):
    def __init__(self):
        super(RMSEFrom1,self).__init__(1)

class SupervisedClusterLoss(nn.Module):
    '''
    A supervised clustering loss made up of two parts:
    cov: embedding results if one class should be close
    mean distance: embedding results if different classes should be distant
    the resulting loss is a*cov+b/mean distance
    '''
    def __init__(self,n_classes,a=1.0,b=.1):
        super(SupervisedClusterLoss, self).__init__()
        self.a=a
        self.n_classes=n_classes
        self.b=b

    def forward(self, pred,group_idx):
        '''
        A supervised clustering loss made up of two parts:
            cov: embedding results if one class should be close
            mean distance: embedding results if different classes should be distant
        the resulting loss is a*cov+b/mean distance
        :param pred: embedding results
        :param group_idx: ground truth group indices
        :return: loss,mean_distance,cov
        '''
        n_classes=self.n_classes
        groups = [pred[i == group_idx] for i in range(n_classes)]
        mean = [x_grouped.mean(dim=0) for x_grouped in groups]
        pairwise_distance = [
            ((mean[i] - mean[j]) ** 2).sum() for i in range(n_classes) for j in range(i)
        ]
        std = [x_grouped.std(dim=0) for x_grouped in groups]
        cov = ((sum(std) ** 2).sum()) / n_classes
        mean_distance = (sum(pairwise_distance)) / (n_classes*(n_classes-1)/2)
        loss = self.a*cov + self.b / mean_distance
        return loss,mean_distance,cov

class SupervisedClusterLoss2(nn.Module):
    '''
    A supervised clustering loss made up of two parts:
    cov: embedding results if one class should be close
    mean distance: embedding results if different classes should be distant
    the resulting loss is a*cov+b*(max possible distance-mean distance)**2+c*(max possible distance-min distance)**2
    '''
    def __init__(self,n_classes,max_possible=8,a=1.0,b=.1,c=0.0):
        super(SupervisedClusterLoss2, self).__init__()
        self.a=a
        self.n_classes=n_classes
        self.b=b
        self.c=c
        self.max_possible=max_possible

    def forward(self, pred,group_idx):
        '''
        A supervised clustering loss made up of two parts:
            cov: embedding results if one class should be close
            mean distance: embedding results if different classes should be distant
        the resulting loss is a*cov+b*(max possible distance-mean distance)**2+c*(max possible distance-min distance)**2
        :param pred: embedding results
        :param group_idx: ground truth group indices
        :return: loss,mean_distance,min_distance,cov
        '''
        n_classes=self.n_classes
        groups = [pred[i == group_idx] for i in range(n_classes)]
        mean = [x_grouped.mean(dim=0) for x_grouped in groups]
        pairwise_distance = [
            ((mean[i] - mean[j]) ** 2).sum() for i in range(n_classes) for j in range(i)
        ]
        std = [x_grouped.std(dim=0) for x_grouped in groups]
        cov = ((sum(std) ** 2).sum()) / n_classes
        mean_distance = (sum(pairwise_distance)) / (n_classes*(n_classes-1)/2)
        min_dist=min(pairwise_distance)
        loss = self.a*cov + self.b * ((self.max_possible-mean_distance)**2)+self.c*((self.max_possible-min_dist)**2)
        return loss,mean_distance,min_dist,cov
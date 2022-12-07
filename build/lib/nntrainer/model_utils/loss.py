import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred,ans):
        diff=pred-ans
        diff=diff.pow(2).mean()
        return diff

class MSEFromX(MSE):
    def __init__(self,x):
        super(MSEFromX,self).__init__()
        self.x=x
    def forward(self,y):
        return super(MSEFromX,self).forward(y,self.x)

class MSEFrom0(MSEFromX):
    def __init__(self):
        super(MSEFrom0,self).__init__(0)
class MSEFrom1(MSEFromX):
    def __init__(self):
        super(MSEFrom1,self).__init__(1)

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR,self).__init__()
    def forward(self,img1, img2):
        data_range=max(img1.max(),img2.max())
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(data_range / torch.sqrt(mse))

class FixedIDXFunc:
    def __init__(self,func,idx):
        self.func=func
        self.idx=idx
    def __call__(self,*args):
        return self.func(*[args[idx] for idx in self.idx])

class FixedKEYFunc:
    def __init__(self,func,idx):
        self.func=func
        self.idx=idx
    def __call__(self,**kwargs):
        return self.func(**{idx:kwargs[idx] for idx in self.idx})

class WeightedSumLoss(nn.Module):
    def __init__(self,loss_names,loss_weights,*args):
        super(WeightedSumLoss,self).__init__()
        names=[]
        weights=[]
        modules=[]
        for i in range(len(loss_names)):
            if loss_weights[i]==0: continue
            names.append(loss_names[i])
            weights.append(loss_weights[i])
            modules.append(args[i])
        self.loss_names=names
        self.loss_weights=weights
        self.loss_modules=modules
    def forward(self,*args,**kwargs):
        losses={}
        totals=0
        for idx,name in enumerate(self.loss_names):
            tmp=self.loss_modules[idx](*args,**kwargs)
            losses[name]=tmp
            totals=totals+self.loss_weights[idx]*tmp
        return losses,totals
    def update_amgrid(self,amg,column,loss,totals,name='loss',bs=1):
        for k,v in loss.items():
            l=amg+[v.detach().cpu().item(),bs,k,column]
        l=amg+[totals.detach().cpu().item(),bs,name,column]
        return l

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss,self).__init__()
    def forward(self,y,label):
        bs=y.size(0)
        similarity=torch.cosine_similarity(y.view(bs,-1),label.view(bs,-1))
        return (1-similarity).mean()

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
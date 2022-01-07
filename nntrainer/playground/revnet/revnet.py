import random
from numpy.core.fromnumeric import ptp, shape
from numpy.core.numeric import indices
import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as nl
import numpy.random as npr
import nntrainer.trainer as tr
import tqdm

import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import nntrainer.model_utils as mu
import nntrainer.data_utils as du
import nntrainer.simple_datasets as sd

def solve_x_rand_(A,a,b=0,scale=True):
    '''
    solve Ax=b where A: M*N, x: N*1, b: M*1
    a: (N-M)*1 values of (N-M) xs
    '''
    # print('solving...',A.shape)
    if scale:
        scale=np.abs(A).mean()
        # print(scale)
        # print(b)
        A=A/scale
        b=b/scale
    M,N=A.shape
    r=np.zeros(shape=[N])
    if len(a.shape)==1:
        a=a[...,np.newaxis]
    if N>M:
        indices=random.sample(range(N),N-M)
        i1=np.sort(indices)
        i2=[i for i in range(N) if i not in i1]
        # print('i',i1,i2)
        r[i1]=a[:,0]
        A1=A[:,i1]
        A2=A[:,i2]
        b=b-A1@a
    else:
        A2=A
        i2=list(range(N))
    # print(A2.shape,b.shape)
    # print(A2)
    # print('i2',i2,r[i2],'N=',N)
    try:
        r[i2]=nl.solve(A2,b[:,0])
        # print(r)
    except Exception as exc:
        # print('err',exc)
        d=np.sqrt((A2**2).sum(-1,keepdims=True))+.0001
        A2=A2/d
        corr=A2@A2.T
        active=[True]*M
        for i in range(M):
            if not active[i]: continue
            if d[i]<.001: active[i]=False
            for j in range(i+1,M):
                if corr[i,j]>.999:
                    active[j]=False
        active_cnt=sum([1 for act in active if act])
        nonactive=[not act for act in active]
        nonactive_cnt=M-active_cnt
        a0=npr.normal(size=[nonactive_cnt])
        r0=np.zeros(shape=[M])
        tmp=solve_x_rand_(A2[active][:,active],a0,b[active])
        # print('tmp',tmp.shape,active,M)
        r0[active]=tmp
        # print(r0,a0.shape)
        r0[nonactive]=a0
        # print('success2')
        r[i2]=r0
    return r

def norm(a):
    d=np.sqrt((a**2).sum(),dtype=np.float32)
    return a/d

class RevNet(nn.Module):
    def __init__(self,child_net,input_shape,output_size,x=None,step_size=.001,load_child_path=None):
        '''
        childnet: f(.) to be solved
        input_shape: shape of the input x
        output_size: total number of float/double numbers in y
        x: initial x. default=random
        step_size: x change in each iteration. default=0.001
        load_child_path: pretrained child_net path. default=None
        '''
        super(RevNet,self).__init__()
        self.child_net=child_net
        self.child_net.eval()
        if load_child_path:
            try:
                tr.load(self.child_net,load_child_path)
            except Exception as exc:
                print(exc)
        self.step_size=step_size
        self.input_shape=input_shape
        self.output_size=output_size
        if x is None:
            x=torch.randn(input_shape,dtype=torch.float32)
        else:
            x=torch.tensor(x,dtype=torch.float32)
        self.x=nn.Parameter(x.unsqueeze(0),requires_grad=True)
        self.x_grad=None
    def _clear_grad(self):
        for p in self.child_net.parameters():
            p.grad=None
        self.x.grad=None
    def forward(self,x=None):
        if x is None:
            x=self.x
        return self.child_net(x)
    def adjust_initx(self,tgt,opt_dict,opt='sgd',need_reset=False,num_epoch=2000):
        '''
        Adjust x with gradient descent.(Step 1)
        tgt: target y
        opt_dict, opt: optimizer definition, similar to that in nntrainer.
        need_reset: whether or not to re-randomize x as initalization.
        num_epoch: number of epochs required in this step.
        '''
        if need_reset:
            x=torch.randn([1]+self.input_shape,dtype=torch.float32)
            self.x=nn.Parameter(x,requires_grad=True)
        opt=tr.get_optimizer([self.x],opt,opt_dict)
        self.train()
        tbar=tqdm.tqdm(range(num_epoch),desc='Training',total=num_epoch)
        tgt=torch.tensor(tgt,dtype=torch.float32,requires_grad=False)
        for epoch in tbar:
            opt.zero_grad()
            y=self.forward()[0].view(-1)
            loss=((y-tgt)**2).mean()
            loss.backward()
            opt.step()
            tbar.set_postfix({'mse':loss.detach().item()})
    def train_child(self,dl_train,dl_valid,opt_dict,opt='sgd',num_epoch=2000,save_name=None):
        '''
        Train child_net
        Only standard CLS training in this function.
        For other training schemes, you can either override this function or pretrain the network somewhere else.
        '''
        opt=tr.get_optimizer(self.child_net.parameters(),opt,opt_dict)
        tbar=tqdm.tqdm(range(num_epoch),desc='Training',total=num_epoch)
        loss_func=nn.CrossEntropyLoss()
        am=tr.AverageMeter()
        for epoch in tbar:
            self.child_net.train()
            for x,y_ in dl_train:
                opt.zero_grad()
                y=self.forward(x)
                loss=loss_func(y,y_)
                loss.backward()
                opt.step()
            self.child_net.eval()
            am.reset()
            with torch.no_grad():
                for x,y_ in dl_valid:
                    bs=y_.size(0)
                    y=self.forward(x)
                    top1=tr.accuracy(y,y_)[0]
                    l=am+(top1,bs)
            tbar.set_postfix({'acc':am.avg})
        if save_name:
            tr.save(self.child_net,f'models/{save_name}.pth')
    def calc_xgrad(self):
        xgrad=[]
        y=self.forward()[0].view(-1) # 1 item in batch
        for i in range(self.output_size):
            self._clear_grad()
            y[i].backward(retain_graph=True)
            xgrad.append(self.x.grad.view(-1).numpy())
        # print(xgrad)
        self.x_grad=np.array(xgrad)
    def move_eq(self,a,clip=None):
        '''
        Step 2 of RevNet.
        Moves x to x1 so that f(x)=f(x1).
        a: random vector as the major direction indicator(required since the equations have infinite solutions)
        clip: x clipping to avoid unreasonable values. default: None indicating no need for clipping
        '''
        self.calc_xgrad()
        # print(self.x_grad)
        for i in range(1):
            try:
                # print(a)
                dx=solve_x_rand_(self.x_grad,np.array(a,dtype=np.float32)).reshape(self.x.shape)
                break
            except Exception as exc:
                print(exc)
                a+=self.step_size*npr.normal(0,1,size=a.shape)
        dx=norm(dx).astype(np.float32)
        # print(dx.dtype)
        self.x.data+=self.step_size*dx
        if clip:
            self.x.data=torch.clip(self.x.data,clip[0],clip[1])
    def get_x(self):
        '''
        Returns numpy version of the calculated x.
        '''
        return self.x.data.numpy()

class LeNet5(mu.UnitLayer):
    def __init__(self,nclass=10):
        super(LeNet5,self).__init__()
        self.main=nn.Sequential(
            # Squeeze(1,False),
            mu.ConvBaseBlock([1,6],ks=[5],activate='relu',bn=False),
            nn.Conv2d(6,16,5,padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            mu.Flatten(), # .View([-2,-1]),
            mu.FCBlock_v2([400,120,84,nclass],activate=['relu','relu','relu','none'],bn=False),
            nn.Softmax(-1)
        )

if __name__=='__main__':
    '''
    Example: LeNet5 analysis and visualization.
    '''
    model=LeNet5()
    name='lenet5_mnist'
    # dl_train,dl_valid=sd.load_mnist('e:/data/mnist')
    x8=cv2.imread('mnist_pic/mnist_13.bmp',0)/255.0
    rev_model=RevNet(model,[1,28,28],10,step_size=1,load_child_path=f'models/{name}.pth',x=x8.reshape(1,28,28))
    # rev_model.train_child(dl_train,dl_valid,opt='adamw',opt_dict={'lr':.001,'betas':(.5,.9)},num_epoch=20,save_name=name)
    # rev_model.adjust_initx(tgt=[1]+[0 for i in range(9)],opt='adamw',num_epoch=200,opt_dict={'lr':.005,'betas':(.5,.9)})
    '''for x,y_ in dl_valid:
        x_vis=x[0].numpy()
        break'''
    # print(x_vis)
    x=rev_model.get_x()
    # print(x)

    rev_model.step_size=1
    NUM=49
    cols=10
    rows=NUM//cols+1

    fig=plt.figure(figsize=(cols<<2,rows<<2))   
    ax=fig.add_subplot(rows,cols,1)
    ax.imshow(x[0,0],cmap='gray')
    print(rev_model(torch.tensor(x)))
    for i in range(NUM):
        a=npr.uniform(high=2.0,low=-2.0,size=[28*28-10])
        rev_model.move_eq(a)
        x=rev_model.get_x()
        ax=fig.add_subplot(rows,cols,i+2)
        ax.imshow(x[0,0],cmap='gray')
        print(rev_model(torch.tensor(x,dtype=torch.float32)))
    plt.show()
    
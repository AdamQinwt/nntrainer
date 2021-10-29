def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

from random import randint

def random_select(x,p=.5):
    p=p*100
    for itm in x:
        if randint(0,100)<p:
            return itm
    return itm
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
import torch

def plot_vis_img(buf):
    plt.savefig(buf,format='png')
    buf.seek(0)
    im=imread(buf)
    buf.seek(0)
    return torch.from_numpy(im).type(torch.float32).permute(2,0,1)/255

def plot_gray(images,buf,titles=None):
    cols=len(images)
    rows=1
    fig=plt.figure(figsize=(cols<<2,4))
    plt.subplots_adjust(wspace=.05,hspace=.05)
    for idx in range(cols):
        ax=fig.add_subplot(rows,cols,idx+1)
        if titles:
            ax.imshow(images[idx],cmap='gray')
            ax.set_title(titles[idx])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_axis_off()
    im=plot_vis_img(buf)
    plt.close()
    return im
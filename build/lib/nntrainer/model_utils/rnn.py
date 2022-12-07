import torch
import torch.nn as nn

if __name__=='__main__':
    # batch,length,channel
    x = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(2, 3, 1)
    model=nn.RNN(1, 3, 2, batch_first=True)
    for  p in model.parameters():
        print(p.size())
    y,h=model(x)
    print(x.numpy())
    print(y.detach().numpy().shape)
    print(h.detach().numpy().shape)
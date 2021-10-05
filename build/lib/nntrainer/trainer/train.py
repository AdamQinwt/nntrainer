from trainer.am import AverageMeter

def train(model,criterion,optim,loader,*args,**kwargs):
    model.train()
    train_loss = AverageMeter()
    for x,y_ in loader:
        optim.zero_grad()
        y=model(x,*args,**kwargs)
        loss=criterion(y,y_)
        loss.backward()
        optim.step()
        avg = train_loss + (loss.detach().item(), x[0].size(0))
    return train_loss.avg

def train_cuda(model,criterion,optim,loader,*args,**kwargs):
    model.train()
    train_loss = AverageMeter()
    for x,y_ in loader:
        x = x.cuda()
        y_ = y_.cuda()
        optim.zero_grad()
        y=model(x,*args,**kwargs)
        loss=criterion(y,y_)
        loss.backward()
        optim.step()
        avg = train_loss + (loss.detach().cpu().item(), x[0].size(0))
    return train_loss.avg
from trainer.am import AverageMeter

def valid(model,criterion,loader,*args,**kwargs):
    model.eval()
    valid_loss = AverageMeter()
    for x,y_ in loader:
        y=model(x,*args,**kwargs)
        loss=criterion(y,y_)
        avg = valid_loss + (loss.detach().item(), x[0].size(0))
    return valid_loss.avg

def valid_cls(model,loader,*args,**kwargs):
    model.eval()
    correct,total=0.0,0.0
    for x,y_ in loader:
        y=model(x,*args,**kwargs).detach()
        y_=y_.argmax(1,keepdim=True)
        y=y.argmax(1,keepdim=True)
        c=(y==y_).sum()
        # print(y,y_,y==y_)
        correct+=c
        total+=x.size(0)
    # print(correct,total)
    return correct/total

def valid_cuda(model,criterion,loader,*args,**kwargs):
    model.eval()
    valid_loss = AverageMeter()
    for x,y_ in loader:
        x=x.cuda()
        y_=y_.cuda()
        y=model(x,*args,**kwargs)
        loss=criterion(y,y_)
        avg = valid_loss + (loss.detach().cpu().item(), x[0].size(0))
    return valid_loss.avg

def valid_clscuda(model,loader,*args,**kwargs):
    model.eval()
    correct,total=0.0,0.0
    for x,y_ in loader:
        x = x.cuda()
        y_ = y_.cuda()
        y=model(x,*args,**kwargs).detach().cpu()
        y_=y_.argmax(1,keepdim=True)
        y=y.argmax(1,keepdim=True)
        c=(y==y_).sum()
        # print(y,y_,y==y_)
        correct+=c
        total+=x.size(0)
    # print(correct,total)
    return correct/total
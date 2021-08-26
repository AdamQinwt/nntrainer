def predict(model,loader,*args,**kwargs):
    model.eval()
    for x,y_ in loader:
        y=model(x,*args,**kwargs).detach()
        yield x.numpy(),y.numpy()

def predict_cls(model,loader,*args,**kwargs):
    model.eval()
    for x,y_ in loader:
        y=model(x,*args,**kwargs).detach().argmax(1,keepdim=True)
        yield x.numpy(),y.numpy()

def predict_cuda(model,loader,*args,**kwargs):
    model.eval()
    for x,y_ in loader:
        y=model(x,*args,**kwargs).detach().cpu()
        yield x.numpy(),y.numpy()

def predict_cls_cuda(model,loader,*args,**kwargs):
    model.eval()
    for x,y_ in loader:
        y=model(x,*args,**kwargs).detach().argmax(1,keepdim=True).cpu()
        yield x.numpy(),y.numpy()
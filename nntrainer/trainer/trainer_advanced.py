from trainer.am import AverageMeter

class Trainer:
    def __init__(self,
                 data_preprocess_func,model_forward_func,loss_forward_func):
        self.data_preprocess_func=data_preprocess_func
        self.model_forward_func=model_forward_func
        self.loss_forward_func=loss_forward_func

    def train(self,model, criterion, optim, loader, *args, **kwargs):
        model.train()
        train_loss = AverageMeter()
        for x, y_ in loader:
            optim.zero_grad()
            x,y_=self.data_preprocess_func(x,y_,*args,**kwargs)
            y = self.model_forward_func(model,x, *args, **kwargs)
            loss = self.loss_forward_func(criterion,y, y_)
            loss.backward()
            optim.step()
            avg = train_loss + (loss.detach().item(), x[0].size(0))
        return train_loss.avg

    def valid(self,model,criterion,loader,*args,**kwargs):
        model.eval()
        valid_loss = AverageMeter()
        for x,y_ in loader:
            x, y_ = self.data_preprocess_func(x, y_, *args, **kwargs)
            y = self.model_forward_func(model, x, *args, **kwargs)
            loss = self.loss_forward_func(criterion, y, y_)
            avg = valid_loss + (loss.detach().item(), x[0].size(0))
        return valid_loss.avg

    def predict(self,model, loader, *args, **kwargs):
        model.eval()
        for x, y_ in loader:
            x, y_ = self.data_preprocess_func(x, y_, *args, **kwargs)
            y = self.model_forward_func(model, x, *args, **kwargs).detach()
            yield x.numpy(), y.numpy()

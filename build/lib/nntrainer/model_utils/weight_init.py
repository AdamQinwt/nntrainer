import torch
from nntrainer.trainer.am import AMGroup

def model_param_stat(model):
    cnt=0
    total=0
    for param in model.parameters():
        size=1
        for i in param.size():
            size*=i
        total+=size
        cnt+=1
    return cnt,total

def model_norm_stat(model,p):
    if not isinstance(p,list):
        p=[p]
    am=AMGroup([str(x) for x in p])
    am.reset()
    for param in model.parameters():
        size=1
        for i in param.size():
            size*=i
        for x in p:
            l=am[str(x)]+(param.norm(x).detach().cpu().item()/size,size)
    return [v for k,v in am.t_avg()]

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
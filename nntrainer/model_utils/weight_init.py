import torch

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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
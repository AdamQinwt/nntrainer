import torch

def save(model,path):
    torch.save(model.state_dict(),path)

def remove_state_dict_prefix(state_dict,prefix='module.'):
    l=len(prefix)
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k[l:]] = v
        else:
            new_sd[k] = v
    return new_sd

def add_state_dict_prefix(state_dict,prefix='module.'):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k] = v
        else:
            new_sd[prefix+k] = v
    return new_sd

def load(model,path,remove_prefix=None,add_prefix=None):
    state_dict = torch.load(path)
    if remove_prefix is not None:
        state_dict = remove_state_dict_prefix(state_dict,remove_prefix)
    if add_prefix is not None:
        state_dict = add_state_dict_prefix(state_dict,add_prefix)
    model.load_state_dict(state_dict, strict=True)
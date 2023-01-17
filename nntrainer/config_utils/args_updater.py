import os
import os.path as osp
import shutil
import yaml
import numpy as np
import argparse
from easydict import EasyDict as edict
import torch

def ed():
    return edict()

def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    # torch.cuda.set_device(gpu)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argument_gpu():
    return ('--gpu',{'help':'Specify the GPUs to be used.','default':'0'})
def argument_name():
    return ('--name',{'help':'A unique name for the run.','default':''})

def parse_args(introduction='',arguments=None):
    parser = argparse.ArgumentParser(introduction)
    parser.add_argument('--cfg', help="Specify the path of the path of the config(*.yaml)",
                        default=None)
    if arguments is not None:
        for a,d in arguments:
            parser.add_argument(a,**d)
    args = parser.parse_args()
    return args

def get_config(raw_args,config,*args,**kwargs):
    if config is None:
        config=edict()
        raw_args.CFG = ''
    if raw_args.cfg:
        config.CFG=raw_args.cfg
        config=update_config(config,raw_args.cfg,*args,**kwargs)
        print('Using '+config.CFG)
    else:
        print("Using default config...")
    return config

def get_args(raw_config,reset_config=None,introduction='',arguments=None,allow_extra=True):
    args=parse_args(introduction=introduction,arguments=arguments)
    config=get_config(args,raw_config,allow_extra=allow_extra)
    return reset_config(config,args) if reset_config is not None else config

def _update_dict(config,k, v,allow_extra,*args,**kwargs):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            if not allow_extra:
                raise ValueError("{}.{} not exist in config.py".format(k, vk))
    return config


def update_config(config,config_file,allow_extra,*args,**kwargs):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f,Loader=yaml.FullLoader))
    for k, v in exp_config.items():
        if k in config:
            if isinstance(v, dict):
                config=_update_dict(config,k, v,allow_extra,*args,**kwargs)
            else:
                config[k] = v
        else:
            if not allow_extra:
                raise ValueError("{} not exist in config.py".format(k))
            else:
                if isinstance(v, dict):
                    config=_update_dict(config,k, v,allow_extra,*args,**kwargs)
                else:
                    config[k] = v
    return config
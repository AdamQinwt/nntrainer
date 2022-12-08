import torch
import torch.nn as nn
from ..config_utils import get_device
from ..data_utils import load_dataset
from torch.utils.data import DataLoader,Dataset
from ..model_utils import CascadedModels,default_factories,WeightedSumLoss
from .model_saveload import save,load
from .optimizer import get_optimizer_sheduler_v2
from .am import AMGrid
from tqdm import tqdm
import os

def dictcpy(src=None,dst=None):
    if src is None: return dst
    for k,v in src.items(): dst[k]=v
    return dst

class ModelGroup:
    def __init__(
        self,
        args,
        device,
        models=None,
        factory=None,
        global_model_params=None,
        model_params=None,
        donnot_load=None,
        ):
        if models is None: models=['model']
        elif isinstance(models,str): models=[models]
        if factory is None: factory=default_factories
        if donnot_load is None: donnot_load=[]
        m={}
        for model_name in models:
            param_dict={}
            param_dict=dictcpy(global_model_params,param_dict)
            if model_params is not None:
                param_dict=dictcpy(model_params[model_name],param_dict)
            m0=CascadedModels(args[model_name].model_def,factory,param_dict).to(device)
            if model_name not in donnot_load:
                if args[model_name].pretrained is not None: load(m0,args[model_name].pretrained)
            m[model_name]=m0
            self.__setattr__('model_name',m0)
        self.m=m
    def __getitem__(self,idx):
        return self.m[idx]
    def parameters(self,model_names):
        if isinstance(model_names,str): return self[model_names].parameters()
        e_d_paramlist=[]
        for mn in model_names:
            for p in self[mn].parameters():
                e_d_paramlist.append(p)
        return p
    def create_optimizers(self,args):
        opt,sch={},{}
        for opt_k,opt_params in args.items():
            opt0,sch0=get_optimizer_sheduler_v2(
                self.parameters(opt_params['model']),
                opt_params['type'],
                opt_params['scheduler'],
                opt_params['params'],
                opt_params['scheduler_params'],
            )
            opt[opt_k]=opt0
            sch[opt_k]=sch0
        return opt,sch
    def train(self):
        for k,v in self.m.items(): v.train()
    def valid(self):
        for k,v in self.m.items(): v.eval()
    def save(self,root_dir):
        for k,v in self.m.items(): save(v,f'{root_dir}/{k}.pth')

class Trainer:
    def __init__(
        self,args,
        data,
        models=None,factory=None,global_model_params=None,model_params=None,donnot_load=None,
        loss_names=None,loss_weights=None,loss_funcs=None,
        device=None
        ):
        if device is None: device=get_device()
        d={}
        for dk,dv in data.items():
            if not isinstance(dv,DataLoader): dv=load_dataset(dv,**args.data[dk].loader)
            d[dk]=dv
        if models is not None:
            models=ModelGroup(
                args,device,models,factory,global_model_params,model_params,donnot_load
            )
            if 'optimizers' in args.keys():
                opt,sch=models.create_optimizers(args.optimizers)
            else:
                opt=None
                sch=None
        else:
            models=None
            opt=None
            sch=None
        if loss_names is None: loss_names=[]
        if loss_weights is None: loss_weights=[]
        loss_func=WeightedSumLoss(loss_names,loss_weights,*loss_funcs).to(device)
        am=AMGrid(loss_func.loss_names+['loss'],list(data.keys()))
        am.reset_active()

        root_dir=f'runs/{args.name}'
        os.makedirs(root_dir,exist_ok=True)
        args.root_dir=root_dir

        self.device=device
        self.args=args
        self.data=d
        self.model_names=list(models.m.keys())
        self.opt_names=list(opt.m.keys())
        self.models=models
        self.opt=opt
        self.sch=sch
        self.loss_func=loss_func
        self.am=am
        self.stages=list(data.keys())
    def run(
        self,nepoch,
        forward_func=None,
        before_iter='normal',
        start_iter='normal',
        end_iter='normal',
        after_iter='normal',
        ):

        if before_iter is None: before_iter=DoNothing
        elif before_iter=='normal': before_iter=PrepareStageNormal
        if start_iter is None: start_iter=DoNothing
        elif start_iter=='normal': start_iter=StartIterNormal
        if end_iter is None: end_iter=DoNothing
        elif end_iter=='normal': end_iter=EndIterNormal
        if after_iter is None: after_iter=DoNothing
        elif after_iter=='normal': after_iter=AfterIterNormal
        opt=self.opt
        sch=self.sch
        models=self.models
        loss_func=self.loss_func
        am=self.am
        args=self.args
        device=self.device

        for epoch in range(nepoch):
            for stage in self.stages:
                before_iter(stage,opt,sch,models,args,device)
                tbar=tqdm.tqdm(self.data[stage],total=64,desc=f'Ep.{stage} {epoch}')
                for idx, d in enumerate(tbar):
                    start_iter(stage,opt,sch,models,args,device)
                    outcomes,bs=forward_func(models,d,device)
                    losses,total_losses=loss_func(outcomes,d,models)
                    total_losses.backward()
                    end_iter(stage,d,opt,sch,models,args,device)
                    l=loss_func.update_amgrid(am,stage,losses,total_losses,bs=bs)
                    tbar.set_postfix({'loss':l})
                after_iter(stage,opt,sch,models,args)
            print(str(am))

class DoNothing:
    def __call__(self, *args,**kwargs):
        pass
class StageFunc(DoNothing):
    def train(self,*args,**kwargs):
        pass
    def valid(self,*args,**kwargs):
        pass
    def __call__(self, stage,*args, **kwargs):
        eval(f'self.{stage}')(*args,**kwargs)
class PrepareStageNormal(StageFunc):
    def train(self, opt, sch, models,*args,**kwargs):
        models.train()
    def valid(self, opt, sch, models,*args,**kwargs):
        models.valid()
class StartIterNormal(StageFunc):
    def train(self, opt, sch, models,*args,**kwargs):
        for k,v in opt.items(): v.zero_grad()
class EndIterNormal(StageFunc):
    def train(self, opt, sch, models,*args,**kwargs):
        for k,v in opt.items(): v.step()
class AfterIterNormal(StageFunc):
    def train(self, opt, sch, models,*args,**kwargs):
        for k,v in sch.items(): v.step()
    def valid(self, opt, sch, models,config,*args,**kwargs):
        models.save(config.root_dir)

class CalcLoss:
    def __call__(self,outcomes,d,models):
        raise NotImplementedError

class ForwardFunction:
    def __call__(self,models,d,device):
        raise NotImplementedError
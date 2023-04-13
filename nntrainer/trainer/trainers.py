import torch
import torch.nn as nn
from ..config_utils import get_device
from ..data_utils import load_dataset
from torch.utils.data import DataLoader,Dataset
from ..model_utils import CascadedModels,default_factories,WeightedSumLoss,model_param_stat,weights_init_normal
from .model_saveload import save,load
from .optimizer import get_optimizer_sheduler_v2
from .am import AMGrid
import tqdm
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
            # weights_init_normal(m0)
            if model_name not in donnot_load:
                if args[model_name].pretrained is not None: load(m0,args[model_name].pretrained)
            m[model_name]=m0
            self.__setattr__(model_name,m0)
        self.m=m
    def __getitem__(self,idx):
        return self.m[idx]
    def parameters(self,model_names):
        if isinstance(model_names,str): return self[model_names].parameters()
        e_d_paramlist=[]
        for mn in model_names:
            for p in self[mn].parameters():
                e_d_paramlist.append(p)
        return e_d_paramlist
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
    def train(self,need=None):
        for k,v in self.m.items():
            if need is not None:
                if k not in need:
                    continue
            v.train()
    def valid(self,need=None):
        for k,v in self.m.items():
            if need is not None:
                if k not in need:
                    continue
            v.eval()
    def weight_init(self,need=None):
        for k,v in self.m.items():
            if need is not None:
                if k not in need:
                    continue
            weights_init_normal(v)
    def train_valid(self,need_train=None,need_eval=None):
        if need_train is not None: self.train(need_train)
        if need_eval is not None: self.valid(need_eval)
    def save(self,root_dir,need_save=None):
        for k,v in self.m.items(): 
            if need_save is None:
                save(v,f'{root_dir}/{k}.pth')
            elif k in need_save:
                save(v,f'{root_dir}/{k}.pth')
    def model_param_stat(self):
        mps={}
        for k,v in self.m.items():
            mps[k]=model_param_stat(v)
        return mps
    @staticmethod
    def step(x,need=None):
        # step opt or sch
        for k,v in x.items():
            if need is not None:
                if k not in need:
                    continue
            if v is not None: v.step()
    @staticmethod
    def train_batch(data_in,models,opt,forward_func,loss_func,loss_keys,need_train=None,need_eval=None,**extra_args):
        models.train_valid(need_train,need_eval)
        opt.zero_grad()
        rdict=forward_func(data_in,models,**extra_args)
        ldict={k:v for k,v in rdict.items() if k in loss_keys}
        if 'data' in loss_keys: ldict['data']=data_in
        if 'models' in loss_keys: ldict['models']=models
        for k,v in extra_args.items():
            if k in loss_keys:
                ldict[k]=v
        losses,loss_totals=loss_func(**ldict)
        loss_totals.backward()
        rdict['losses']=losses
        rdict['loss_totals']=loss_totals
        opt.step()
        return rdict
    @staticmethod
    def valid_batch(data_in,models,forward_func,loss_func,loss_keys,need_eval=None,**extra_args):
        models.train_valid(None,need_eval)
        rdict=forward_func(data_in,models,**extra_args)
        ldict={k:v for k,v in rdict.items() if k in loss_keys}
        if 'data' in loss_keys: ldict['data']=data_in
        if 'models' in loss_keys: ldict['models']=models
        for k,v in extra_args.items():
            if k in loss_keys:
                ldict[k]=v
        losses,loss_totals=loss_func(**ldict)
        rdict['losses']=losses
        rdict['loss_totals']=loss_totals
        return rdict

class StageLoss:
    def __init__(self,weighted_loss_function):
        self.wsl=weighted_loss_function
    def update_amgrid(self,*args,**kwargs): return self.wsl.update_amgrid(*args,**kwargs)
    def __call__(self,**kwargs):
        al=[v for k,v in kwargs.items()]
        return self.wsl(*al)

class StageTrainer:
    def __init__(
            self,
            models,
            opt,sch,
            **extra_args
    ):
        self.models=models
        self.opt=opt
        self.sch=sch
        self.extra_args=extra_args
        self.stages={}
    def add_stage(self,stage_type,stage_name,forward_function=None,loss_function=None,loss_keys=None,loss_names=None,total_loss_name=None,opt=None,opt_name=None,need_train=None,need_valid=None,**extra_args):
        self.stages[stage_name]={
            'stage_type': stage_type,
            'forward_func': forward_function,
            'loss_func': loss_function,
            'loss_keys': loss_keys,
            'loss_names': loss_names,
            'total_loss_name': total_loss_name,
            'need_train': need_train,
            'need_valid': need_valid,
            'extra_args': extra_args,
        }
        if opt_name is None: opt_name=stage_name
        if opt: self.stages[stage_name]['optimizer']=opt[opt_name]
    def collect_names(self):
        rows=[]
        cols=[]
        for k,v in self.stages.items():
            cols.append(k)
            for loss_name in v['loss_names']:
                if loss_name in rows: continue
                rows.append(loss_name)
            if v['total_loss_name'] in rows: continue
            rows.append(v['total_loss_name'])
        return rows,cols
    def deactivate_amgrid_entry(self,amgrid):
        for stage_name,stage_info in self.stages.items():
            total_loss_name=stage_info['total_loss_name']
            loss_names=stage_info['loss_names']
            for loss_item in amgrid.rows:
                if loss_item not in loss_names and not loss_item == total_loss_name:
                    amgrid.deactivate(loss_item,stage_name)
    def activate_amgrid_for_stages(self,amgrid,row_name,stages=None):
        if isinstance(row_name,str): row_name=[row_name]
        if stages is None: stages=self.stages.keys()
        for rn in row_name:
            for s in stages: amgrid.activate(rn,s)
    def __call__(self,stage_name,data_in,amgrid=None,**extra_args):
        stage=self.stages[stage_name]
        ea={}
        for k,v in extra_args.items(): ea[k]=v
        for k,v in stage['extra_args'].items(): ea[k]=v
        stage_type=stage['stage_type']
        if stage_type=='train':
            rdict=ModelGroup.train_batch(
                data_in,
                self.models,
                stage['optimizer'],
                stage['forward_func'],
                stage['loss_func'],
                stage['loss_keys'],
                stage['need_train'],
                stage['need_valid'],
                **ea
            )
        elif stage_type=='valid':
            rdict=ModelGroup.valid_batch(
                data_in,
                self.models,
                stage['forward_func'],
                stage['loss_func'],
                stage['loss_keys'],
                stage['need_valid'],
                **ea
            )
        else: raise ValueError
        if amgrid is not None:
            # print(rdict)
            l=stage['loss_func'].update_amgrid(amgrid,stage_name,rdict['losses'],rdict['loss_totals'],stage['total_loss_name'],bs=rdict['bs'])
            # print(l)
        else: l=None
        return rdict,l

class Stage:
    def __init__(
            self,
            stage_type,
            stage_name,
            forward_function,
            need_train=None,
            need_valid=None,
            **extra_args,
        ):
        self.info={
            'stage_type': stage_type,
            'stage_name': stage_name,
            'forward_function': forward_function,
            'need_train': need_train,
            'need_valid': need_valid,
        }
        for k,v in extra_args.items():
            self.info[k]=v
    def set_opt(self,opt=None,opt_name=None):
        self.info['opt']=opt
        self.info['opt_name']=opt_name
    def set_loss(self,total_loss_name,loss_keys,loss_names,loss_weights,*args):
        self.info['loss_keys']=loss_keys
        self.info['loss_names']=loss_names
        self.info['loss_function']=StageLoss(WeightedSumLoss(loss_names,loss_weights,*args))
        self.info['total_loss_name']=total_loss_name
    def set(self,k,v): self.info[k]=v
    def add_to_trainer(self,trainer):
        trainer.add_stage(**self.info)

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
        self.opt_names=list(opt.keys())
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
        elif before_iter=='normal': before_iter=PrepareStageNormal()
        if start_iter is None: start_iter=DoNothing
        elif start_iter=='normal': start_iter=StartIterNormal()
        if end_iter is None: end_iter=DoNothing
        elif end_iter=='normal': end_iter=EndIterNormal()
        if after_iter is None: after_iter=DoNothing
        elif after_iter=='normal': after_iter=AfterIterNormal()
        opt=self.opt
        sch=self.sch
        models=self.models
        loss_func=self.loss_func
        am=self.am
        args=self.args
        device=self.device

        for epoch in range(nepoch):
            am.reset()
            for stage in self.stages:
                before_iter(stage,opt,sch,models,args,device)
                tbar=tqdm.tqdm(self.data[stage],total=len(self.data[stage]),desc=f'Ep.{stage} {epoch}')
                for idx, d in enumerate(tbar):
                    batch_data=start_iter(stage,d,opt,sch,models,args,device)
                    outcomes,bs=forward_func(models,batch_data,device)
                    losses,total_losses=loss_func(outcomes,batch_data,models,device)
                    total_losses.backward()
                    end_iter(stage,batch_data,opt,sch,models,args,device)
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
        return eval(f'self.{stage}')(*args,**kwargs)
class PrepareStageNormal(StageFunc):
    def __init__(self,need_tune=None):
        self.need_tune=None
    def train(self, opt, sch, models,*args,**kwargs):
        models.train(self.need_tune)
    def valid(self, opt, sch, models,*args,**kwargs):
        models.valid(self.need_tune)
class StartIterNormal(StageFunc):
    def __init__(self,need=None):
        self.need=need
    def train(self, d,opt, sch, models,cfg,device,*args,**kwargs):
        if self.need is None:
            for k,v in opt.items(): v.zero_grad()
        else:
            for ne in self.need:
                opt[ne].zero_grad()
        return [dd.to(device) for dd in d]
    def valid(self, d,opt, sch, models,cfg,device,*args,**kwargs):
        return [dd.to(device) for dd in d]
class EndIterNormal(StageFunc):
    def __init__(self,need=None):
        self.need=need
    def train(self, d,opt, sch, models,*args,**kwargs):
        if self.need is None:
            for k,v in opt.items(): v.step()
        else:
            for ne in self.need:
                opt[ne].step()
class AfterIterNormal(StageFunc):
    def __init__(self,need_save=None):
        super(StageFunc,self).__init__()
        self.need_save=need_save
    def train(self, opt, sch, models,*args,**kwargs):
        for k,v in sch.items(): v.step()
    def valid(self, opt, sch, models,config,*args,**kwargs):
        models.save(config.root_dir,self.need_save)

class CalcLoss:
    def __call__(self,outcomes,d,models):
        raise NotImplementedError

class ForwardFunction:
    def __call__(self,models,d,device):
        raise NotImplementedError
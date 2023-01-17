import torch
import torch.optim as optim

def zipdict(keys,values):
    '''
    zip key-value lists into dictionary
    :param keys: keys
    :param values: values
    :return: dictionary
    '''
    r={keys[i]:values[i] for i in range(len(keys))}
    return r

def get_optimizer(parameters,type,param_dict,*args,**kwargs):
    '''
    get an optimzer
    :param parameters: optimizer parameters
    :param type: optimizer class. sgd, adam, adamw supported
    :param args: None
    :param param_dict: specify optimizer parameters. in key-value form.
    :return:
    '''
    p=param_dict
    if type=='sgd':
        optimizer=optim.SGD(
            parameters,**p
        )
    elif type=='adam':
        optimizer=optim.Adam(
            parameters,**p
        )
    elif type == 'adamw':
        optimizer = optim.AdamW(
            parameters,**p
        )
    else:
        raise ValueError
    return optimizer

def get_scheduler(optimizer,type='multi_step',gamma=.1,*args,**kwargs):
    if type=='multi_step':
        scheduler=optim.lr_scheduler.MultiStepLR(
            optimizer,gamma=gamma,**kwargs
        )
    elif type=='step':
        scheduler=optim.lr_scheduler.StepLR(
            optimizer,gamma=gamma,**kwargs
        )
    elif type=='none':
        scheduler=None
    else:
        raise ValueError
    return scheduler

def get_optimizer_sheduler(parameters,type_opt,type_sch,param_dict_optimizer,*args,**kwargs):
    '''
    get an optimzer and its scheduler
    '''
    try:
        optimizer=get_optimizer(parameters,type_opt,param_dict_optimizer)
        scheduler=get_scheduler(optimizer,type_sch,**kwargs)
    except Exception as exp:
        print(exp)
        raise ValueError
    return optimizer,scheduler

def get_optimizer_sheduler_v2(parameters,type_opt,type_sch,param_dict_optimizer,param_dict_scheduler):
    '''
    get an optimzer and its scheduler
    '''
    try:
        optimizer=get_optimizer(parameters,type_opt,param_dict_optimizer)
        scheduler=get_scheduler(optimizer,type_sch,**param_dict_scheduler)
    except:
        raise ValueError
    return optimizer,scheduler
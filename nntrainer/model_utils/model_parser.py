import torch
import torch.nn as nn
import yaml
from copy import deepcopy
from nntrainer.model_utils.fc import FCBlock_v2
from nntrainer.model_utils.convbase import ConvBaseBlock,ResConvBaseBlock
from nntrainer.model_utils.anode.ode_block import ODEBlock
from nntrainer.model_utils.view import Cat,View
def str_replacer(s,params):
    if '$' in s:
        keys = []
        current = ''
        current_stack = 0
        for c in s:
            if c == '$':
                if current_stack > 0:
                    current_stack -= 1
                    if current_stack == 0:
                        keys.append(current)
                else:
                    current_stack += 1
            elif current_stack > 0:
                current += c
        for k in keys:
            if k in params.keys():
                # print('f{Final Replacing {s} with {k}}')
                s = s.replace(f'${k}$', str(params[k]))
            else:
                # print(f'{k} is not found in the parameter dict. Setting to default(1024)')
                # s = s.replace(f'${k}$', '1024')
                pass
    try:
        return eval(s)
    except:
        return s

def item_replacer(entry,params):
    if isinstance(entry,dict):
        return {k:item_replacer(v,params) for k,v in entry.items()}
    elif isinstance(entry,list):
        return [item_replacer(x,params) for x in entry]
    elif isinstance(entry,str):
        return str_replacer(entry,params)
    else:
        return entry

class Factory:
    def __init__(self):
        self.reset()
    def reset(self):
        self.factory_dict = {}
    def register_item(self,k,v):
        self.factory_dict[k]=v
    def register_dict(self,d):
        for k,v in d.items():
            self.factory_dict[k]=v
    def create(self,modules):
        m = []
        d=self.factory_dict
        for mod in modules:
            type=mod['type']
            del mod['type']
            if type in d.keys():
                m.append(d[type](**mod))
            else:
                try:
                    m.append(eval(f'nn.{type}')(**mod))
                except:
                    print(f'{type} not found!')
                    raise ValueError
        return nn.Sequential(*m)

class DefaultNNFactory(Factory):
    def __init__(self):
        super(DefaultNNFactory,self).__init__()
        self.register_dict({
            'fc':FCBlock_v2,
            'conv':ConvBaseBlock,
            'res_conv':ResConvBaseBlock,
            'anode':ODEBlock,
            'cat':Cat,
            'view':View,
        })

class NetworkFromFactory(nn.Module):
    def __init__(self,factory,modules):
        self.factory=factory
        super(NetworkFromFactory,self).__init__()
        self.main=factory.create(modules)
    def forward(self,x):
        return self.main(x)

def parse_model(fnames,factory,params=None):
    ''' parse model from yaml
    :param fnames: file name list
    :param factory: A dictionary of factories like {'fm':feature_map_factory,'fc':fc_factory...}
    :param params: A dictionary of params like {'HIDDEN':256,'ACT':relu...}
    :return:a list of models
    '''
    if not params:
        params={}
    if not isinstance(fnames,list):
        fnames=[fnames]
    models=[]
    for fname in fnames:
        if isinstance(fname,list):
            models.append([parse_model(fname,factory,deepcopy(params))])
        else:
            with open(fname,'r') as f:
                y=yaml.load(f)
            if 'params' in y.keys():
                for k,v in y['params'].items():
                    params[k]=v
            # print(f'Parsing {y}')
            typename = y['type']
            module_dict=[item_replacer(x,params) for x in y['modules']]
            print(module_dict)
            if typename in factory.keys():
                model = factory[typename].create(module_dict)
            else:
                raise ValueError
            models.append(model)
    return models

class CascadedModels(nn.Module):
    def __init__(self,fnames,factory,params=None):
        super(CascadedModels,self).__init__()
        dis=parse_model(fnames,factory,params)
        self.main=nn.Sequential(*dis)
    def forward(self,x):
        y=self.main(x)
        return y

if __name__=='__main__':
    f=Factory()
    f.register_item('fc',FCBlock_v2)
    # model=parse_model(['example.yaml','example2.yaml'],{'fc':f})
    model=CascadedModels(['example.yaml','example2.yaml'],{'fc':f})
    # print(len(model))
    print(model)
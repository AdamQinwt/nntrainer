import torch
import torch.nn as nn

class EmptyLayer(nn.Module):
    """Placeholder"""
    def __init__(self,*args,**kwargs):
        super(EmptyLayer, self).__init__()
        self.infolist=args
        self.infodict=kwargs
        for k,v in kwargs.items():
            self.__setattr__(k,v)

class UnitLayer(nn.Module):
    '''
    models that only need one main part
    could be a sequential or unit block
    '''
    def __init__(self):
        super(UnitLayer,self).__init__()
    def forward(self,*args,**kwargs):
        return self.main(*args,**kwargs)

class ActivationLayer(UnitLayer):
    '''
    activation layer from name
    '''
    def __init__(self,act,*args,**kwargs):
        super(ActivationLayer,self).__init__()
        act_dict={'none':EmptyLayer,'sigmoid':nn.Sigmoid,'relu':nn.ReLU,'tanh':nn.Tanh,'lrelu':nn.LeakyReLU}
        if act in act_dict.keys():
            self.main=act_dict[act](*args,**kwargs)
        else:
            self.main=eval(f'nn.{act}')(*args,**kwargs)

class ChainLayer(UnitLayer):
    def __init__(self,block,chain_args,itm_args=None,common_args=None,*args,**kwargs):
        '''
        w
        :param block: Chained block function.
        :param chain_args: Chained args. The next layer uses part of the data of the previous one.[{'name':[in_channel,out_channel],'value':[3,64,64,128]}]
        :param itm_args: Item args. arguments for each layer.
        :param common_args: Shared args. Dict.
        '''
        super(ChainLayer,self).__init__()
        layers=[]
        n=len(chain_args)-1
        if itm_args is not None and not isinstance(itm_args,list):
            print('itm_args is not a list. common_args recommended.')
            itm_args=[itm_args for i in range(n)]
        if not isinstance(chain_args,list):
            chain_args=[chain_args]
        for i in range(n):
            chain_arg={}
            for a in chain_args:
                name1,name2=a['name']
                value1,value2=a['value'][i],a['value'][i+1]
                chain_args[name1]=value1
                chain_args[name2]=value2
            layers.append(block(**chain_arg,**itm_args[i],**common_args))
        self.main=nn.Sequential(*layers)

class ChainBlock(ChainLayer):
    def __init__(self,factory,block_type,*args,**kwargs):
        super(ChainBlock,self).__init__(factory[block_type],*args,**kwargs)

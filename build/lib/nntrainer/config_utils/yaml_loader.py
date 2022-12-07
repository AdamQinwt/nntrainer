from yaml import load,dump

def load_yaml(fname,*args,**kwargs):
    with open(fname,'r') as fin:
        res=load(fin,*args,**kwargs)
    return res

def dump_yaml(data,fname,*args,**kwargs):
    with open(fname,'w') as fout:
        dump(data,fout,*args,**kwargs)
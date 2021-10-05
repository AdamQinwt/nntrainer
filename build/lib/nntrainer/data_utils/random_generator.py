import random

def repeat_if_notlist(x,n):
    return x if isinstance(x,list) else [x]*n
def rand_uniform(dim,a,b):
    a=repeat_if_notlist(a,dim)
    b=repeat_if_notlist(b,dim)
    return [random.uniform(a[i],b[i]) for i in range(dim)]

def rand_norm(dim,mean,cov):
    mean = repeat_if_notlist(mean, dim)
    cov = repeat_if_notlist(cov, dim)
    return [random.normalvariate(mean[i],cov[i]) for i in range(dim)]

default_random_generators={'uniform':rand_uniform,'norm':rand_norm}

if __name__=='__main__':
    a=rand_norm(3,[0,1,-4],[1,3,5])
    print(a)
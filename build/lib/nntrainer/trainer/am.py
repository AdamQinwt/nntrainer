class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total=0.0
        self.avg=0.0
        self.n=0
    def __add__(self, other):
        self.total+=other[0]*other[1]
        self.n+=other[1]
        self.avg=self.total/self.n
        return self.avg
    def __str__(self):
        return f'Total\t{self.total}\nN\t\t{self.n}\nAverage\t{self.avg}'

class AMGroup:
    '''
    A group of Average Meters
    '''
    def __init__(self,keys):
        self.am={k:AverageMeter() for k in keys}
        for k,v in self.am.items():
            self.__setattr__(k,v)
    def reset(self):
        for k, v in self.am.items():
            v.reset()
    def __add__(self, other):
        '''
        update
        :param other: (value,number,key)
        :return: new average value
        '''
        return self.am[other[2]]+(other[0],other[1])
    def __getitem__(self, item):
        return self.am[item]
    def __len__(self):
        return len(self.am)
    def t(self):
        '''
        Traverse
        :return: k,vs
        '''
        for k, v in self.am.items():
            yield k,v
    def t_avg(self):
        '''
        Traverse
        :return: k,vs
        '''
        for k, v in self.am.items():
            yield k,v.avg
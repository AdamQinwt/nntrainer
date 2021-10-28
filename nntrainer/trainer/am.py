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

from terminaltables import AsciiTable

class AMGrid:
    '''
    A grid of average meters
    '''
    def __init__(self,rows,columns):
        self.am={(k1,k2):AverageMeter() for k2 in columns for k1 in rows}
        self.active_mask={}
        self.active_cnt=0
        self.rows=rows
        self.cols=columns
    def reset(self):
        self.active_cnt=0
        for k, v in self.am.items():
            v.reset()
            self.active_mask[k]=True
            self.active_cnt+=1
    def update_active_cnt(self):
        self.active_cnt=0
        for k, v in self.am.items():
            if self.active_mask[k]:
                self.active_cnt+=1
    def __add__(self, other):
        '''
        update
        :param other: (value,number,key1,key2)
        :return: new average value
        '''
        return self.am[other[2],other[3]]+(other[0],other[1])
    def __getitem__(self, item):
        return self.am[tuple(item)]
    def __len__(self):
        return self.active_cnt
    def deactivate(self,*args):
        if len(args)==1:
            self.active_mask[tuple(args[0])]=False
        else:
            self.active_mask[(args[0],args[1])]=False
        self.update_active_cnt()
    def activate(self,*args):
        if len(args)==1:
            self.active_mask[tuple(args[0])]=True
        else:
            self.active_mask[(args[0],args[1])]=True
        self.update_active_cnt()
    def t(self):
        '''
        Traverse
        :return: k,vs
        '''
        for k, v in self.am.items():
            if self.active_mask[k]:
                yield f'{k[1]}-{k[0]}',v
    def t_avg(self):
        '''
        Traverse
        :return: k,vs
        '''
        for k, v in self.am.items():
            if self.active_mask[k]:
                yield f'{k[1]}-{k[0]}',v.avg
    def __str__(self):
        table=[['']+self.cols]
        for r in self.rows:
            entry=[r]
            for c in self.cols:
                if self.active_mask[(r,c)]:
                    entry+=f'{[self[r,c].avg]}'
                else:
                    entry+='-'
            table.append(entry)
        return AsciiTable(table).table


class AMGridClassification(AMGrid):
    def __init__(self, topk):
        rows=['loss']+[f'top_{k}' for k in topk]
        self.topk=topk
        cols=['train','valid']
        super(AMGridClassification,self).__init__(rows,cols)
    def deactivate_train_accuracy(self):
        for k in self.topk:
            self.active_mask[f'top_{k}','train']=False
        self.update_active_cnt()

class AMGridRegression(AMGrid):
    def __init__(self):
        rows=['loss']
        cols=['train','valid']
        super(AMGridRegression,self).__init__(rows,cols)
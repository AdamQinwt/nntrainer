from abc import   ABCMeta, abstractmethod

class CompareRecord:
    def __init__(self):
        self.value=None
    @abstractmethod
    def _need_record(self,value):
        return None
    def update(self,value,*args,**kwargs):
        if (self.value is not None) and (not self._need_record(value)):
            return False
        self.value=value
        self.info=kwargs
        return True
    def get(self):
        return self.value,self.info

class MaxRecord(CompareRecord):
    def _need_record(self,value):
        return value>self.value
class MinRecord(CompareRecord):
    def _need_record(self,value):
        return value<self.value
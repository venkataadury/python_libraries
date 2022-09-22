#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


class ObjectWrapper(object):
    def __init__(self,obj):
        self.obj=obj
    
    def __str__(self): return self.obj.__str__()
    
    def assign(self,obj): self.obj=obj
    def assignAndGet(self,obj):
        self.obj=obj
        return self.obj
    
    def get(self): return self.obj
    def __len__(self): return len(self.obj)


# In[ ]:


print("Loaded General Python Libary")


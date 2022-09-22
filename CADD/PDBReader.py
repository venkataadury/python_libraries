#!/usr/bin/env python
# coding: utf-8

# In[ ]:


GLOBAL_LOADED_PDBREADER=True


# In[2]:


import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from CADD.General import *


# In[3]:


atomSeqParser=lambda at,mol,empty=None: at.GetPDBResidueInfo().GetName().strip()
class PDBAtomParser:
    def __init__(self,pdbfile,parsefn=atomSeqParser,preparse=None,removeHs=False):
        self.pdbfile=pdbfile
        self.atomfn=parsefn
        self.preparse=preparse
        self.removeHs=removeHs
    
    def parse(self,**kwargs):
        mol=Chem.MolFromPDBFile(self.pdbfile,removeHs=self.removeHs,**kwargs)
        if self.preparse is not None: self.preparse(mol,self)
        ret=[]
        for at in mol.GetAtoms():
            ret.append(self.atomfn(at,mol,self))
        return np.array(ret)


# In[4]:


def countAtomTypes(filename,removeHs=False,**kwargs):
    myparse=PDBAtomParser(filename,removeHs=removeHs)
    atomseq=myparse.parse(**kwargs)
    return np.unique(atomseq,return_counts=True)


# In[61]:


print("PDB Tools Loaded")


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from rdkit import Chem


# In[2]:


class SequentialSMILESLoader:
    def __init__(self,smifilename,skip_failures=True,attach_names=False,count_max=True,default_batch=1,max_mols=-1):
        self.filename=smifilename
        if type(smifilename)==str:
            self.file=open(smifilename,"r")
        else:
            self.file=None
        self.counter=0
        self.autoskip=skip_failures
        self.name=attach_names
        self.ended=False
        if count_max:
            if self.file:
                self.linecount=len(self.file.readlines())
                self.file=open(smifilename,"r")
            else:
                self.linecount=len(smifilename) #Assuming input is a list of molecules
        else: self.linecount= None
        self.default_batch=default_batch
        self.mollim=max_mols
        self.skipcount=0
    
    def getFilename(self): return (self.filename if type(self.filename)==str else None)
    def restartSequence(self,keep_location=False):
        ret=SequentialSMILESLoader(self.filename,self.autoskip,self.name,self.linecount is None,self.default_batch)
        if keep_location: ret.skipNext(self.counter)
        return ret
    
    def skipNext(self,num=0):
        K=0
        if self.file is not None:
            for ln in self.file:
                self.counter+=1
                K+=1
                self.skipcount+=1
                if K>=num: break
            else: self.ended=True
        else:
            self.counter+=num
            self.skipcount+=num
            if self.counter>=len(self.filename): self.ended=True
            
    def getNext(self,num=0,as_smiles=False):
        if num==0: num=self.default_batch
        ret=[]
        if self.name:
            names=[]
        K=self.counter
        skip=0
        empty=0
        
        if self.ended: return None
        
        if self.file is None:
            retn=[]
            while len(retn)<num or num<0:
                self.counter+=1
                if self.counter>=len(self.filename) or (self.mollim>0 and self.counter>=self.mollim):
                    self.ended=True
                    break
                curmol=self.filename[self.counter]
                if type(curmol)==str and (not as_smiles):
                    try:
                        curmol=Chem.MolFromSmiles(curmol)
                    except:
                        continue
                retn.append(curmol)
            return retn
            
        for ln in self.file:
            ln=ln.strip().split()
            if not len(ln):
                self.counter+=1
                empty+=1
                continue
            
            mol=ln[0]
            if self.name: myname=ln[1].strip()
            else: myname = None
            if not as_smiles:
                mol=Chem.MolFromSmiles(mol)
                if self.autoskip and (mol is None):
                    self.counter+=1
                    skip+=1
                    continue
            ret.append(mol)
            if self.name: names.append(myname)
            
            self.counter+=1
            if (self.counter-K-skip-empty==num) or (self.mollim>0 and self.counter-self.skipcount>=self.mollim): break
        else: self.ended=True
        
        if self.name: return ret,names
        else: return ret
    
    def drain(self,as_smiles=False): return self.getNext(-1,as_smiles)
    def __del__(self):
        if self.file is not None:
            self.file.close()
    
    def __help__(self): print("This is help. The sequential smiles loader can (ideally) work with both SMILES files and lists of molecules/smiles strings")


# In[3]:


class RDKitMoleculeWrapper:
    def __init__(self,mol):
        self.mol = mol
        if type(self.mol)!=Chem.rdchem.Mol: self.mol=Chem.MolFromSmiles(self.mol)
    
    def __call__(self): return self.mol
    def get(self): return self.mol
    
    def getSize(self,includeHs=False):
        if includeHs: tempmol=Chem.AddHs(self.mol)
        else: tempmol=self.mol
        return (tempmol.GetNumHeavyAtoms() if (not includeHs) else len(tempmol.GetAtoms()))


# In[4]:


import copy
# Get a similarity score distribution
def getSimilarityDistribution(smilesloader,goal=max,post=None,default=0,fingerprint=Chem.RDKFingerprint,similarity=Chem.DataStructs.TanimotoSimilarity,precompute_fps=False,batch_mols=0):
    num_mol=smilesloader.linecount
    keep_count=False
    if num_mol is None:
        num_mol=0
        keep_count=True
    if type(goal)==str:
        if goal=="max":
            goal=max
            post=None #Post processing
        if goal=="mean":
            goal=lambda x,y:x+y
            post=lambda z: z/num_mol
    
    
    
    if precompute_fps:
        mols=smilesloader.drain(as_smiles=False)
        if type(mols)==tuple and type(mols[0])==list: mols=mols[0]
        mols=[fingerprint(mol) for mol in mols]
        print("Fingerprints computed for",len(mols),"molecules")
    
        ret=[default for _ in mols]
    
        for i in range(len(ret)):
            mol1=mols[i]
            if (i+1)%2500==0: print(i+1,"molecules completed of",len(ret))
            #if not precompute_fps: mol1=fingerprint(mol1)
            for j,mol2 in enumerate(mols):
                if j==i : continue
                #if not precompute_fps: mol2=fingerprint(mol2)
                simsc=similarity(mol1,mol2)
                ret[i]=goal(ret[i],simsc)
    else:
        ret=[]
        mainloader=smilesloader.restartSequence(keep_location=True)
        thismols=mainloader.getNext(batch_mols,as_smiles=False)
        while thismols is not None:
            if type(thismols)==tuple and type(thismols[0])==list: thismols=thismols[0]
            thisfps=[fingerprint(mol) for mol in thismols]
            #Do stuff here
            nxt=[default]*len(thismols)
            shf=len(ret)
            print(shf,"molecules processed")
            startloader=smilesloader.restartSequence(keep_location=True)
            allmols=startloader.getNext(batch_mols,as_smiles=False)
            blockind=0
            while allmols is not None:
                if type(allmols)==tuple and type(allmols[0])==list: allmols=allmols[0]
                allfps=[fingerprint(mol) for mol in allmols]
                for i,tmol in enumerate(thisfps):
                    for j,cfp in enumerate(allfps):
                        if j+blockind==i+shf: continue
                        nxt[i]=goal(nxt[i],similarity(tmol,cfp))
                blockind+=len(allmols)
                allmols=startloader.getNext(batch_mols,as_smiles=False)
            ret+=nxt
            thismols=mainloader.getNext(batch_mols,as_smiles=False)
    
    if post is not None: ret=[post(v) for v in ret]
    return np.array(ret)


# In[9]:


# Usage for getSimilarityDistribution()
'''
import time

starttime=time.time()
file="/home/venkata/CADD/denovo_benchmarking/analysis/test.smi"
sload=SequentialSMILESLoader(file,default_batch=256)
sload.skipNext(1024) #Skipping Part of the Iterator is Allowed
sdist=getSimilarityDistribution(sload,goal="max",precompute_fps=True) #You can use precompute_fps=False as well to save RAM
endtime=time.time()

import matplotlib.pyplot as plt
plt.hist(sdist)
print(endtime-starttime,len(sdist))
'''
pass


# In[7]:


print("Loaded Library 'General (RDKit)'")


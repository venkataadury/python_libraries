#!/usr/bin/env python
# coding: utf-8

# In[ ]:


GLOBAL_LOADED_ATOMRNN=True


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from CADD.PDBReader import *

import sys,os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence

DEVICE=torch.device("cuda")


# In[4]:


# Loading the PDB file as string
def readPDBAtomSeq(filename,skipHs=False):
    with open(filename, 'r') as f:
        lines = f.readlines()
    atypes=[ln[12:16].strip() for ln in lines if (ln[0:4]=="ATOM" or ln[0:6]=="HETATM") and ((skipHs and ln[12:16].strip()[0]!="H") or (not skipHs))]
    return atypes

def readPDBFiles(files,N,S,skipHs=False):
    rawdata=[]
    K=0
    L=0
    print("Target:",N)
    #Skip first 'S', and read 'N' files
    file_names = [fn for fn in files]
    sorted(file_names)
    for f in file_names:
        if K<S:
            K+=1
            continue
        if (L+1)%2500==0: print(L+1,f)
        rawdata.append([OPENFLAG]+readPDBAtomSeq(folder+"/"+f,skipHs=skipHs))
        L+=1
        if L>=N: break
    print(len(rawdata),"files read")
    return rawdata
def samplechar(poss):
  posses=np.sum(poss)
  if posses<1e-10: return np.random.choice(range(len(poss)))
  else:
    poss=np.nan_to_num(poss,0.)
    poss/=sum(poss)
  return np.random.choice(range(len(poss)),p=poss)
def buildVocabulary(asets):
    global MASK
    vocab=set()
    for aset in asets:
        vocab=set.union(vocab,set(aset))
    return vocab


# In[3]:


# Some default parameters
HIDE=""
MASK='\0'
SPLIT="~"
OPENFLAG="^"
MAXLEN=72
BUFFER_SIZE=1000
DTYPE_INT=torch.int64

datafunc=readPDBAtomSeq


# In[5]:


def encodeAll(seqlist,keys=dict(),intlevel=torch.int32,strict=False):
    maxint=max([keys[k] for k in keys.keys()],default=0)
    nextint=maxint+1
    ret=[]
    for seq in seqlist:
        intseq=[]
        for el in seq:
            if el in keys.keys(): intseq.append(keys[el])
            else:
                if strict: raise ValueError("Key not found: "+str(el))
                intseq.append(nextint)
                keys[el]=nextint
                nextint+=1
        ret.append(torch.tensor(intseq,dtype=intlevel).to(DEVICE))
    return ret,keys
def rawdataToDataset(rawdata,keys=dict(),intlevel=DTYPE_INT,shuffle=True):
    rawdata_encoded,encoded_keys=encodeAll(rawdata,keys,intlevel)
    rawdata_encoded=pad_sequence(rawdata_encoded,batch_first=True)
    if shuffle:
        randord=torch.randperm(len(rawdata_encoded))
        return rawdata_encoded[randord],encoded_keys
    else:
        return rawdata_encoded,encoded_keys
    
def forcePadding(padded_data,padding,crop=False):
    if padding<=padded_data.shape[-1]: return padded_data if not crop else padded_data[:,:padding]
    extzeros=torch.zeros((len(padded_data),padding-padded_data.shape[-1])).to(padded_data.device)
    return torch.cat((padded_data,extzeros),dim=1)
    

def loadPDBsForTraining(filenames,N,S,shuffle=True,skipHs=False,keys=dict()): #Quick shortcut
    rawdata=readPDBFiles(filenames,N,S,skipHs=skipHs)
    return rawdataToDataset(rawdata,shuffle=shuffle,keys=keys)


# In[ ]:


EMBED=64
RNN_UNITS=512
DENSE_UNITS=256
from torch import nn
import torch

class RecurrentTemplate(nn.Module):
    def __init__(self,vocab,seqlen):
        super(RecurrentTemplate,self).__init__()
        self.maxlen=seqlen
        self.vocabulary=vocab
        self.vocabulary[""]=0
        self.constructVocabularyInverse()
        self.vocab_size=len(self.vocabulary)
        self.embedlayer=torch.nn.Embedding(num_embeddings=len(self.vocabulary)+2,embedding_dim=EMBED)
        
    def constructVocabularyInverse(self):
        self.vocabulary_inverse=dict()
        for k in self.vocabulary.keys():
            self.vocabulary_inverse[self.vocabulary[k]]=k
    
    def encode(self,lst): #Batched (B,?)
        return encodeAll(lst,self.vocabulary)[0]
    def decode(self,tens): #Batched
        ret=[]
        for vec in tens:
            nv=[]
            for r in vec:
                nv.append(self.vocabulary_inverse[int(r)] if r!=0 else MASK)
            ret.append(nv)
        return ret
        

class RecurrentLearnerModel(RecurrentTemplate):
    def __init__(self,vocab,seqlen,grulayers=2):
        super(RecurrentLearnerModel,self).__init__(vocab,seqlen)
        self.gru=torch.nn.LSTM(EMBED,RNN_UNITS,num_layers=grulayers,batch_first=True)
        self.linear=torch.nn.Linear(RNN_UNITS,DENSE_UNITS)
        self.finallayer=torch.nn.Linear(DENSE_UNITS,self.vocab_size+1) #len(vocab) + MASK*1
        
    def forward(self,x,hidden=None):
        #x has shape (B,L)
        x=self.embedlayer(x) #Return shape (B,L,E) - where E is embedding dim
        outs,hidden=self.gru(x,hidden)
        outs=self.linear(outs)
        outs=self.finallayer(outs)
        return F.softmax(outs,dim=-1),outs,hidden #Outs needed for CrossEntropyLoss
    
    def predict(self,x,hidden=None,limit_k=-1):
        #x has shape (B,L)
        #Output has shape (B,1) - with one prediction at each place
        #limit_k limits outputs to top 'k' most probable results
        preds,_,hidden=self.forward(x,hidden)
        if limit_k<=0: limit_k = preds.shape[-1]
        preds,inds=preds.topk(limit_k)
        #print(preds.shape,inds.shape,torch.sum(preds,dim=-1)[:,:,np.newaxis].shape,preds[0].multinomial(1,replacement=False).shape)
        preds=preds/torch.sum(preds,dim=-1)[:,:,np.newaxis]
        predret=[]
        for i,pv in enumerate(preds):
            selind=torch.squeeze(pv.multinomial(1,replacement=False),dim=-1) # (seqlen) - list of seqlen indices chosen
            truinds=torch.stack([v[selind[j]] for j,v in enumerate(inds[i])])
            predret.append(truinds)
        predmul=torch.stack(predret)
        #predmul=torch.stack([torch.squeeze(inds[i,]) for i,pv in enumerate(preds)])
        return predmul,hidden
    
    def generate(self,num,maxlen,limit_k=-1,use_hidden=None):
        hidden=use_hidden
        base_start=[['^']]*num
        inseq=self.encode(base_start)
        inseq=torch.stack(inseq)
        curseq=inseq
        ended=torch.zeros(len(base_start),dtype=torch.bool).to(DEVICE)
        with torch.no_grad():
            while curseq.shape[1]<maxlen and not torch.all(ended):
                res,hidden=self.predict(inseq,hidden,limit_k)
                vals=torch.squeeze(res,dim=-1)==MASK
                ended=ended | vals
                curseq=torch.cat((curseq,res),dim=1)
        return self.decode(curseq)


# In[ ]:


BATCH_SIZE=32
import abc
class GenericModelTrainer(metaclass=abc.ABCMeta):
    def __init__(self,model,optimizer_type,learnrate,lossfn):
        self.model=model
        self.optimizer = optimizer_type(self.model.parameters(),lr=learnrate)
        self.lossfn=lossfn
    
    def useModel(self,inputs): return self.model(inputs)
    def evaluateModel(self,inputs,labels,batch_size=BATCH_SIZE,metrics=[],print_out=False):
        metric_profile=dict()
        for met in metrics: metric_profile[str(met)] = []
        with torch.no_grad():
            net_loss=[]
            for idx in range(len(inputs)//BATCH_SIZE):
                self.optimizer.zero_grad()
                inpdata=inputs[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                outdata=labels[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]

                sample = self.useModel(inpdata)
                loss=self.lossfn(sample,outdata)
                for met in metrics:
                    metx=met(sample,outdata)
                    metric_profile[str(met)].append(metx.item())
                net_loss.append(loss.item())
            if len(metrics):
                if print_out: print("Metrics:",end=" ")
                for met in metrics:
                    meanmet=np.mean(metric_profile[str(met)])
                    metric_profile[str(met)]=meanmet
                    if print_out: print(str(met),meanmet,end=", ")
                if print_out: print()
        if len(metrics): return np.mean(net_loss)
        else: return np.mean(net_loss),metric_profile
                
        
    
    def trainModel(self,inputs,labels,epochs,batch_size=BATCH_SIZE,metrics=[],save=False,save_every=5,save_path="model_training."):
        loss_profile=[]
        loss_metrics=dict()
        for met in metrics: loss_metrics[str(met)] = []
        EPOCHS=epochs
        SAVE=5
        SAVE_PATH = save_path # "saved_models/Seqlearner_Amac_PT_weightsonly"
        PROGRESS_STEP = (len(inputs)//BATCH_SIZE)//72
        for ep in range(EPOCHS):
            net_loss=[]
            timestart=time.time()
            for met in metrics: loss_metrics[str(met)].append([])
            print("Epoch",ep+1,"[",end="")
            for idx in range(len(inputs)//BATCH_SIZE):
                self.optimizer.zero_grad()
                inpdata=inputs[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                outdata=labels[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                
                sample = self.useModel(inpdata)
                loss=self.lossfn(sample,outdata)
                loss.backward()
                self.optimizer.step()
                
                for met in metrics:
                    metx=met(sample,outdata)
                    loss_metrics[str(met)][-1].append(metx.item())
                
                net_loss.append(loss.item())
                if idx%PROGRESS_STEP==0: print("=",end="",flush=True)
            timeend=time.time()
            print("]",flush=True)
            loss_profile.append(np.mean(net_loss))
            print("Epoch",ep+1,"with loss:",loss_profile[-1],"took time:",timeend-timestart)
            if len(metrics):
                print("Metrics:",end=" ")
                for met in metrics:
                    meanmet=np.mean(loss_metrics[str(met)][-1])
                    loss_metrics[str(met)][-1]=meanmet
                    print(str(met),meanmet,end=", ")
                print()
            if save and ep%SAVE==0: torch.save(mymodel,SAVE_PATH) #torch.save(mymodel.state_dict(),SAVE_PATH)
        if save: torch.save(mymodel,SAVE_PATH)
        print("Completed")
        if len(metrics): return loss_profile,loss_metrics
        else: return loss_profile


# In[ ]:


class RecurrentClassifier(RecurrentTemplate):
    def __init__(self,vocab,seqlen,classes,rnn_depth=1,deep_model=None):
        super(RecurrentClassifier,self).__init__(vocab,seqlen)
        self.num_classes=classes
        self.gru=torch.nn.LSTM(EMBED,RNN_UNITS,num_layers=rnn_depth,batch_first=True)
        if deep_model is None: deep_model = torch.nn.Linear(RNN_UNITS,2)
        self.linear = deep_model
        
    def forward(self,x,hidden=None):
        #x has shape (B,L)
        x=self.embedlayer(x) #Return shape (B,L,E) - where E is embedding dim
        outs,hidden=self.gru(x,hidden)
        outs=self.linear(outs)
        return F.softmax(outs,dim=-1),outs,hidden #Outs needed for CrossEntropyLoss 


# In[ ]:


def manual_metric_accuracy(preds,trues):
    preds=torch.max(preds,dim=-1).indices.detach()
    return torch.sum(preds==trues)/len(preds)
def constantScore(inp): return 1.


# In[ ]:


try:
    if GLOBAL_LOADED_REINFORCE: pass
except NameError:
    get_ipython().run_line_magic('run', '/home/venkata/python/python_libraries/python/Reinforcement_Module.ipynb')
'''
    Molecule Generation Game: The goal of this game is to generate molecules that satisfy certain criteria (such as net charge, size, etc.)
    The generator returns one of 'N' characters (decided apriori), and the generation ends when the model generates a '0'
'''
class RNNMolGenerationEnvironment(GenericDiscreteEnvironmentExtension,metaclass=abc.ABCMeta):
    def __init__(self,num_tokens,max_len,end_on=0,startWith=None,score_fn=constantScore,fail_fn=constantScore,score_kws=dict(),fail_kws=dict()):
        super(RNNMolGenerationEnvironment,self).__init__((1,),getBoundedObservables((1,),0,num_tokens+1),num_tokens+1,False)
        self.endtoken=end_on
        self.num_tokens = num_tokens
        self.maxlen = max_len
        self.scoring = score_fn
        self.failing = fail_fn
        self.fail_kws=fail_kws
        if startWith is not None: self.start=np.array(startWith,dtype=np.int64) #Need to implement
        else: self.start = np.zeros((1,),dtype=np.int64)
        
        self.state=self.start
        self.ended=False
        self.genseq=self.state
        
        #Correcting "Action Space" for batching
        #self.action_space = getBoundedObservables((1,),0,num_tokens+1)
        self.action_space = IntegerActions(0,num_tokens+1)
        
        self.score_kws=score_kws
    
    def isTerminalState(self): return self.genseq.shape[-1]>self.maxlen or self.ended
    def getObservation(self): return self.state[np.newaxis,:]
    
    def getReward(self):
        if self.isTerminalState(): return (self.scoring(self.genseq,**self.score_kws) if self.genseq.shape[-1]<=self.maxlen else torch.tensor(self.failing(self.genseq,**self.fail_kws)*(~self.ended),dtype=torch.float32))
        else: return 0.
    
    def reset(self):
        self.state=self.start
        self.genseq=self.state
        self.ended=False
        return self.getObservation()
    
    def resolveAction(self,act): #act is a set of batched actions (B,1) like (5,6,2,2,2,4,2,5,0 ... ) as a NUMPY array
        self.state=np.array([act])
        self.genseq=np.append(self.genseq,act)
        self.ended=(act==0)
        return self.isTerminalState()


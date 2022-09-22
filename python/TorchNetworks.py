#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from torch_geometric.nn.inits import glorot, zeros
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence


# In[ ]:


# A generic PyTorch module wrapper for easy use
class GenericModule(nn.Module):
    def __init__(self,inshape,outshape):
        super(GenericModule,self).__init__()
        if type(inshape) == int: self.inshape=(inshape,)
        else: self.inshape=inshape
        if type(outshape) == int: self.outshape=(outshape,)
        else: self.outshape=outshape


# In[ ]:


class GCNLayer(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix,eps=1e-4):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1,keepdims=True)
        node_feats = self.projection(node_feats.transpose(1,2)).transpose(1,2)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / (num_neighbours+eps)
        return node_feats
class MultiheadGCNLayer(nn.Module):
    def __init__(self, c_in, c_out,filters):
        super().__init__()
        self.filters = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((filters,c_out,c_in),device=DEVICE)))
        self.projection=nn.Linear(filters,1)

    def forward(self, node_feats, adj_matrix,eps=1e-4):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1,keepdims=True)
        node_feats_ext=torch.matmul(self.filters,node_feats.transpose(-1,-2)[:,np.newaxis,:,:]).transpose(-1,-2).transpose(1,-1)
        node_feats=self.projection(node_feats_ext).transpose(1,-1).squeeze()
        #node_feats = self.projection(node_feats.transpose(1,2)).transpose(1,2)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / (num_neighbours+eps)
        return node_feats
class DeepIterativeGCN(nn.Module):
    def __init__(self, c_in, c_outs,iters=2,acts=None): #c_outs is a list
        super().__init__()
        self.projections=[]
        self.parameterlist=[]
        if acts is None: acts=[None for _ in range(iters)]
        if len(acts)<iters: acts+=[None]*(iters-len(acts))
        self.acts=acts
        self.iters=iters
        for i in range(iters):
            if len(c_outs)<=i: outv=c_outs[-1]
            else: outv = c_outs[i]
            self.projections.append(GCNLayer(c_in,outv))
            for param in self.projections[-1].parameters(): self.parameterlist.append(param)
            c_in=outv
    
    def parameters(self): return nn.ParameterList(self.parameterlist)
    def to(self,dev):
        for proj in self.projections: proj=proj.to(dev)
        return self

    def forward(self, node_feats, adj_matrix,eps=1e-4):
        ret=node_feats
        for i in range(self.iters):
            ret=self.projections[i](ret,adj_matrix,eps)
            if self.acts[i] is not None: ret=self.acts[i](ret)
        return ret


# In[ ]:


#Aggregations for GCN
def meanAggregation(v):
    return torch.mean(v,axis=1) #Axis 0 is batch-size
def sigmoidMeanAggregation(v):
    return F.sigmoid(torch.mean(v,axis=1))
#A pure fully-connected layer wrapper.
class PureNetwork(GenericModule):
    # The pure decoder module that attempts to decode a molecule to a rep vector from the encoded vector
    def __init__(self,encsize,layers=[1024,1024],acts=[F.relu,F.sigmoid]):
        super(PureNetwork,self).__init__(encsize,layers[-1])
        self.layers=[]
        self.acts=[]
        self.parameterlist=[]
        cursize=int(np.prod(encsize))
        for i,lnc in enumerate(layers):
            self.layers.append(nn.Linear(cursize,lnc))
            self.acts.append(acts[i])
            for param in self.layers[-1].parameters(): self.parameterlist.append(param)
            cursize=lnc
    
    def parameters(self): return nn.ParameterList(self.parameterlist)
    def addLayer(self,lay,actF=None):
        layers.append(lay)
        acts.append(actf)
        self.parameterlist.append(lay.parameters())
    
    def to(self,dev):
        for lay in self.layers: lay=lay.to(DEVICE)
        return self
    
    
    def transform(self,x): return x; #Overridable
    
    def forward(self,x):
        x=self.transform(x)
        for i,lay in enumerate(self.layers):
            x=lay(x)
            if self.acts[i] is not None: x=self.acts[i](x)
        return x


# In[ ]:


print("PyTorch Commons Loaded")


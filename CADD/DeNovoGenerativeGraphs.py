#!/usr/bin/env python
# coding: utf-8

# In[1]:


GLOBAL_LOADED_DENOVOGRAPHS=True


# In[2]:


from CADD.DeNovoReader import *
from python.Reinforcement import *
from python.TorchNetworks import * #Handles all the torch imports (mostly)

DEVICE=torch.device("cuda")


# In[3]:


class GraphConvFeaturizer(nn.Module,GenericAtomFeaturizer):
    def __init__(self,feat_dim,max_nodes,ff,convmodel=None,embed=None,amplify=1):
        #embed is a model that produces a feat_dim vector using the Molecule's featurizer. It is expected to be an object of the GenericAtomFeaturizer parent class
        nn.Module.__init__(self)
        GenericAtomFeaturizer.__init__(self,feat_dim)
        #super(GraphConvFeaturizer,self).__init__()
        self.feat_type="mol" #Molecule Level Featurizer
        self.ff=ff
        self.maxsize=max_nodes
        if embed is None: embed=FFOneHotFeaturizer(self.ff) #By default just one-hot encode them
        self.hotenc=embed
        self.feat_dim=feat_dim
        if convmodel is None: convmodel = GCNLayer(self.feat_dim,int(amplify*self.feat_dim))
        self.conv=convmodel
    
    def getFeatures(self,mol):
        nodes=torch.tensor(mol.getNodeMatrix(self.hotenc),dtype=torch.float32,device=DEVICE)
        edges=torch.zeros((self.maxsize,self.maxsize),device=DEVICE,dtype=torch.float32)
        edges[:len(nodes),:len(nodes)]=torch.tensor(mol.getAdjacencyMatrix(),dtype=torch.float32,device=DEVICE)
        nodes=torch.cat([nodes,torch.zeros((self.maxsize-len(nodes),nodes.shape[-1]),device=DEVICE,dtype=torch.float32)])
        return nodes,edges
    def getFeaturesMultiple(self,mols):
        nodes=[]
        edges=[]
        for mol in mols:
            nd,ed=self.getFeatures(mol)
            nodes.append(nd)
            edges.append(ed)
        nodes=torch.stack(nodes)
        edges=torch.stack(edges)
        return nodes,edges
    def featurize(self,mols,neighs=None,*args):
        nodes,edges = self.getFeaturesMultiple(mols)
        return self.forward(nodes,edges)
    
    def forward(self,nodes,edges): return self.conv(nodes,edges),edges


# In[4]:


constantScore=lambda x: 1.
class RNNMolGenerationEnvironment(GenericDiscreteEnvironmentExtension,metaclass=abc.ABCMeta):
    def __init__(self,num_tokens,max_len,end_on=0,startWith=None,score_fn=constantScore,fail_fn=constantScore,score_kws=dict(),fail_kws=dict()):
        super(MolGeneration,self).__init__((1,),getBoundedObservables((1,),0,num_tokens+1),num_tokens+1,False)
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


# In[5]:


'''
Add nodes one by one to the molecule graph, while also predicting the atom-types that bind to a spot.
The game is played like this:
 - The molecular graph is made atom-by-atom.
 - At each step, from the given molecule graph, the environment will randomly pick one free atom. 
 - The aim of the model is to produce an atom type that will connect (validly) to that atom-type
 
The generation ends if:
1. No free valences
2. ???
'''
import torch
class GraphMolGenerationEnvironment(GenericDiscreteEnvironmentExtension,metaclass=abc.ABCMeta):
    def __init__(self,ff,max_nodes,err_penalty=0.25,succ_reward=0.25):
        super(GraphMolGenerationEnvironment,self).__init__((1,),getBoundedObservables((1,),0,num_tokens+1),num_tokens+1,False)
        self.atomtypes=tuple(ff.getAtomNames())
        self.indices=dict()
        for i,atype in enumerate(self.atomtypes): self.indices[atype]=i
        
        self.ff=ff
        self.sizelim=max_nodes
        self.err=-err_penalty
        self.succ=succ_reward
        self.embed_dim=len(self.atomtypes)+10
        self.embedlayer=torch.nn.Embedding(num_embeddings=len(self.atomtypes),embedding_dim=embed_dim)
        self.reset()
    def reset(self):
        self.nodes=torch.zeros((self.sizelim,self.embed_dim))
        self.edges=torch.zeros((self.sizelim,self.sizelim))
        self.valence=torch.zeros(self.sizelim)
        self.cursize=0
        return self.getObservation()
    def getObservation(self): return self.nodes,self.edges
    
    def startGeneration(self):
        addAtom=np.random.choice(self.ff.getSeeds())
        self.nodes[0,:] = self.embedlayer(torch.tensor([self.indices[addAtom]]))
    


# In[6]:


myff=DeNovoForceFieldLoader("/home/venkata/dnv/data/final_ff_parameters.ffin")
myff.loadBonds("/home/venkata/dnv/data/itps/bondtypes.itp")
myff.loadCategories("/home/venkata/dnv/data/categories.data")
myff.loadRules("/home/venkata/dnv/data/definitions.data")
myff.updateKeyList()


myfeat=GraphConvFeaturizer(72,72,myff).to(DEVICE)
sample_dec=nn.Linear(166*2,1).to(DEVICE)
hotenc=myfeat.hotenc

opts=torch.stack([torch.tensor(hotenc(myff.atom_list[atm],None),dtype=torch.float32,device=DEVICE) for atm in myff.atom_list])

mymol=GrowingDeNovoMolecule(72,hotenc,feat_dim=166,ff=myff) #,"/home/venkata/CADD/data_files/PDBs/result_prot_dnv_CheMBLdist_size11_10000.pdb",ff=myff,includeHs=True)
print(mymol.addAtom("CA",[]),mymol.addAtom("CA",1))
mymol.adj
#nodes,edges=myfeat.featurize([mymol],None)


# In[19]:


nex,eex = myfeat.getFeaturesMultiple([mymol])
conv_ts = torch.jit.trace_module(myfeat.cpu(),{"forward": [nex.cpu(),eex.cpu()]})


# In[20]:


conv_ts.save("/home/venkata/python/torchscript_saves/convfeat.ts")


# In[ ]:


'''
chatomkey=np.random.choice(tuple(mymol.atoms.keys()))
chatom=mymol.atoms[chatomkey]
reqidx=mymol.getNodeIndex(chatomkey)
print(chatomkey,"is chosen atom key")
nodes,edges=myfeat.featurize([mymol])
node_sel=nodes[0,reqidx]
opts_ext=torch.tensor([myff.indexOfAtom(atstr) for atstr in myff.getBondableAtoms(chatom)],dtype=torch.long,device=DEVICE)
for ind in opts_ext: print(myff.ind_list[ind.item()],end=" ")
print()
opts_mod=torch.cat((opts[opts_ext],node_sel.unsqueeze(0).repeat(opts_ext.shape[0],1)),dim=1)
#myff.ind_list[opts_ext[sample_dec(opts_mod).squeeze().argmax()].item()]
natname=myff.ind_list[opts_ext[F.softmax(sample_dec(opts_mod).squeeze(),dim=-1).multinomial(1)].item()]
print("Adding",natname)
mymol.addAtom(natname,chatomkey)

print(mymol.adj)
print(mymol.featmat.argmax(axis=1))
mymol.adj.shape,len(mymol.atoms),len(mymol.bonds),mymol.featmat.shape
'''
print("Graph Generation (for DeNovo) loaded")


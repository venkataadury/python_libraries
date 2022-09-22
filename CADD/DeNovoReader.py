#!/usr/bin/env python
# coding: utf-8

# In[ ]:


GLOBAL_LOADED_DENOVOREAD=True


# In[ ]:


import numpy as np
import sys
from rdkit import Chem
import abc


# In[ ]:


import copy
class DeNovoAtom:
    def __init__(self,name,valence,mass,charge,sigma,eps,vdw=None,hybrid=None):
        sigma = float(sigma)
        if hybrid is None: hybrid = valence -1
        if vdw is None: vdw = sigma/2.0
        valence = int(valence)
        self.atomtype=str(name)
        self.valence = valence
        self.max_valence = valence
        self.sigma=float(sigma)
        self.epsilon = float(eps)
        self.vdw = float(vdw)
        self.hybridization = int(hybrid)
        self.charge = float(charge)
        self.mass = float(mass)
    
    def __str__(self): return "Atom Type: "+self.atomtype
    def __repr__(self): return str(self)
    def clone(self): return DeNovoAtom(self.atomtype,self.valence,self.mass,self.charge,self.sigma,self.epsilon,self.vdw,self.hybridization)
    
    def reduceValency(self,amt=1): self.valence-=amt
            
class DeNovoRule:
    def __init__(self,targets,minc,maxc):
        self.targets=targets
        self.m=minc
        self.M=maxc
    
    def satisfied(self,neighs,central,ff=None,incomplete=True): #neighs is a string list of neighbour atom-types
        c=0
        for n in neighs:
            if type(n)!=str: nobj=n.atomtype
            else: nobj=n
            if nobj in self.targets: c+=1
        return c<=self.M and c>=self.m or (incomplete and ((ff is None) or ff.getMaxValency(central)-len(neighs)>=self.m-c))
    def __str__(self): return "Rule: "+str(self.targets)+": Min="+str(self.m)+(" Max="+str(self.M) if self.M<100 else "")
    def __repr__(self): return str(self)

def loadRule(rulestr,ff,central=None):
    keys=rulestr.split()
    keys[0]=keys[0].strip()
    if keys[0][0]=='@': return None
    
    targ=ff.expandTarget(keys[0],central)
    mv=int(keys[1].strip())
    if len(keys)>2: Mv=int(keys[2].strip())
    else: Mv=100
    return DeNovoRule(targ,mv,Mv)

class DeNovoForceFieldLoader:
    def __init__(self,ff_file):
        self.infile=ff_file
        self.atom_list=dict()
        self.bonds=dict()
        self.categories=dict()
        self.rules=dict()
        self.loadForceField()
        self.seeds=[]
        self.ind_list=dict()
        self.atom_inds=dict()
    
    def loadForceField(self):
        tempfile = open(self.infile,"r")
        for l in tempfile:
            l=l.strip()
            if not l or l[0]=="#": continue
            l = l.split()
            self.atom_list[l[0].strip()]=DeNovoAtom(l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7])
        tempfile.close()
    
    def updateKeyList(self):
        self.ind_list=dict()
        self.atom_inds=dict()
        for i,key in enumerate(self.atom_list):
            self.ind_list[i]=key
            self.atom_inds[key]=i
    def indexOfAtom(self,atstr): return self.atom_inds[atstr]
    def getAtomNameAt(self,ind): return self.ind_list[ind]
    def expandTarget(self,tgstr,central=None):
        els=tgstr.split("|")
        if len(els)>1: els=sum([self.expandTarget(sel.strip()) for sel in els],[])
        for i in range(len(els)):
            if els[i][0]=='[':
                catname=els[i][1:].strip()
                catname=catname[:-1].strip()
                els[i]=list(self.categories[catname])
            elif els[i][-1]=='*':
                prefix=els[i][:-1]
                lst=[]
                for atype in self.atom_list.keys():
                    if atype.startswith(prefix) and ((central is None) or self.canBond(atype,central)): lst.append(atype)
                els[i]=lst
            else: els[i]=[els[i]]
        els=sum(els,[])
        return els
        
    def loadBonds(self,bondfile):
        bf=open(bondfile,"r")
        for l in bf:
            l=l.split()
            l[0]=l[0].strip()
            l[1]=l[1].strip()
            if l[0] not in self.atom_list or l[1] not in self.atom_list: continue
            if l[0] not in self.bonds: self.bonds[l[0]]=[]
            if l[1] not in self.bonds: self.bonds[l[1]]=[]
            self.bonds[l[0]].append(l[1])
            if l[0]!=l[1]: self.bonds[l[1]].append(l[0])
        bf.close()
    
    def loadCategories(self,catfile):
        cf=open(catfile,"r")
        for l in cf:
            l=l.split()
            l=sum([lo.split("|") for lo in l],[])
            self.categories[l[0].strip()]=tuple(l[1:])
        cf.close()
    def loadRules(self,rulefile,catfile=None):
        if catfile is not None: self.loadCategories(catfile)
        rf = open(rulefile,"r")
        for l in rf:
            l=l.split()
            atype=l[0].strip()
            if atype not in self.rules: self.rules[atype]=[]
            rules=' '.join(l[1:])
            rules=rules.split(";")
            for rule in rules:
                robj=loadRule(rule,self,atype)
                if robj is not None: self.rules[atype].append(robj)
        rf.close()
    
    def canBond(self,a1,a2): return a1 in self.bonds and a2 in self.bonds[a1]
    def getBondableAtoms(self,at):
        if type(at)!=str: at=at.atomtype
        return self.bonds[at]
    def getMaxValency(self,a1): return self.atom_list[a1].max_valence
    def isSatisfied(self,cent,neighs,incomplete=True):
        if type(cent)!=str: cent=cent.atomtype
        if len(neighs)>self.atom_list[cent].max_valence: return False
        if cent not in self.rules: return True
        
        for rule in self.rules[cent]:
            if not rule.satisfied(neighs,cent,self,incomplete):
                return False
        return True
    
    def __len__(self): return len(self.atom_list)
    def __getitem__(self,idx): return self.atom_list[idx]
    def __contains__(self,atype):
        if type(atype)!=str: atype=atype.atomtype
        return atype in self.atom_list
    def getAtomNames(self): return self.atom_list.keys()
    def getNewAtom(self,name): return self.atom_list[name].clone()
    def assignSeeds(self,seedlist): self.seeds=tuple(seedlist)
    def getSeeds(self): return self.seeds
    

class DeNovoMolecule:
    def __init__(self,pdbfile=None,ff=None,bond_cutoff=0.27,includeHs=True,readConects=True,proximalBonding=True):
        self.molfile=pdbfile
        self.ff=ff
        self.atoms=dict()
        self.bonds=dict()
        self.adj=None
        self.featmat=None
        self.loadMolecule(cutoff=bond_cutoff,includeHs=includeHs,readConects=readConects,proximalBonding=proximalBonding)
    
    def loadMolecule(self,cutoff=0.27,includeHs=True,readConects=True,proximalBonding=True):
        if (not includeHs) and proximalBonding:
            print("Disabling proximal bonding without hydrogen atoms")
            proximalBonding=False
        self.atoms=dict()
        poses=[]
        self.bonds=dict()
        if self.molfile is None: return
        tempfile = open(self.molfile,"r")
        for ln in tempfile:
            ln=ln.strip()
            if not (ln[:6]=="ATOM  " or ln[:6]=="HETATM"):
                if (not readConects) or (ln[:6]!="CONECT"): continue
                ind1=int(ln[7:11])
                rest=ln[12:].split()
                rest=[int(ind) for ind in rest]
                self.atoms[ind1].reduceValency(len(rest))
                self.bonds[ind1]+=rest
                continue
            elem=ln[12:16].strip()
            idx=int(ln[6:11])
            xc=float(ln[32:39])
            yc=float(ln[39:46])
            zc=float(ln[47:54])
            if (not includeHs) and (elem[0]=="H"): continue
            self.atoms[idx]=self.ff.getNewAtom(elem)
            poses.append(np.array([xc,yc,zc]))
            self.bonds[idx]=[]
        tempfile.close()
        effcut2=(cutoff*10)**2 #Square of "effective" (in A instead of nm) cut-off distance squared
        if proximalBonding:
            keylist=list(self.atoms.keys())
            proxima=dict()
            for i,pos1 in enumerate(poses):
                proxima[keylist[i]]=[]
                for j,pos2 in enumerate(poses):
                    if i>=j: continue
                    distv=np.sum((pos2-pos1)**2)
                    if distv<effcut2:
                        proxima[keylist[i]].append((keylist[j],distv))
                        
                if len(proxima[keylist[i]]):
                    proxima[keylist[i]]=sorted(proxima[keylist[i]],key=lambda el: el[1])
                    proxima[keylist[i]]=proxima[keylist[i]][:self.atoms[keylist[i]].valence]
                    for atnh in proxima[keylist[i]]:
                        j=atnh[0]
                        self.bonds[keylist[i]].append(j)
                        self.bonds[j].append(keylist[i])
                        self.atoms[keylist[i]].reduceValency()
                        self.atoms[j].reduceValency()
                        
                        
    
    def getSize(self): return len(self.atoms)
    def getNeighboursOf(self,elid): return (self.atoms[idx] for idx in self.bonds[elid])
    
    def getAdjacencyMatrix(self):
        if self.adj is not None: return self.adj
        self.adj=np.zeros((len(self.atoms),len(self.atoms)),dtype=np.float32)
        atids=list(self.atoms.keys())
        for i,k in enumerate(atids):
            for nhid in self.bonds[k]: self.adj[i,atids.index(nhid)]=1.
        return self.adj
    
    def getNodeMatrix(self,featurizer,*featargs):
        if self.featmat is not None: return self.featmat
        if featurizer.feat_type=="atom":
            #featurizer is a function that takes Atom object and list of Atom objects (neighbours) with any other optional parameters as per design to produce an encoding/representation numpy array for the node
            feats=[featurizer(self.atoms[node],self.getNeighboursOf(node),*featargs) for node in self.atoms]
            self.featmat=np.stack(feats) if len(feats) else np.zeros((0,0),dtype=np.float32)
        else:
            #Featurizer  instead acts on the entire Molecule Object
            self.featmat=featurizer(self)
        return self.featmat
    
    def getGraphFeatures(self,featurizer): return self.getNodeMatrix(),self.getAdjacencyMatrix()
    def getNodeIndex(self,keyidx):
        #Convert key index to row number
        return list(self.atoms.keys()).index(keyidx)

class GrowingDeNovoMolecule(DeNovoMolecule):
    def __init__(self,maxsize,feat,feat_dim,pdbfile=None,ff=None,bond_cutoff=0.27,includeHs=True,readConects=True,proximalBonding=True):
        super(GrowingDeNovoMolecule,self).__init__(pdbfile,ff,bond_cutoff,includeHs,readConects,proximalBonding)
        self.maxsize=maxsize
        self.featurizer=feat
        assert self.featurizer.feat_type=="atom", "Featurizer must be atom-level for a growing molecule"
        self.getNodeMatrix(featurizer=self.featurizer)
        self.getAdjacencyMatrix()
        if self.featmat.shape[0]==0: self.featmat=np.zeros((0,feat_dim))
        self.bondables=[]
        for k in self.atoms:
            if self.atoms[k].valence: self.bondables.append(k)
    
    def addAtom(self,atomname,toatoms): #toatom is the index
        if type(toatoms)!=list: toatoms=[toatoms]
        nat=self.ff.getNewAtom(atomname)
        neighs=[]
        keylist=list(self.atoms.keys())
        mykey=keylist[-1]+1 if len(keylist) else 1
        self.atoms[mykey]=nat
        self.bonds[mykey]=[]
        keylist.append(mykey)
        self.adj=np.concatenate((self.adj,np.zeros((1,self.adj.shape[1]))))
        self.adj=np.concatenate((self.adj,np.zeros((self.adj.shape[0],1))),axis=1)
        for nid in toatoms:
            neighs.append(self.atoms[nid])
            if not self.addBond(mykey,nid,keylist):
                self.atoms.pop(mykey)
                self.bonds.pop(mykey)
                self.adj=self.adj[:len(self.adj)-1,:len(self.adj)-1]
                return False
        feat=self.featurizer(nat,neighs)
        if self.featmat is not None:
            self.featmat=np.concatenate((self.featmat,feat[np.newaxis,:]))
        if self.atoms[mykey].valence: self.bondables.append(mykey)
        return True
        
        
    def addBond(self,id1,id2,keylist=None):
        if self.atoms[id1].valence==0 or self.atoms[id2].valence==0: return False
        self.bonds[id1].append(id2)
        self.bonds[id2].append(id1)
        self.atoms[id1].reduceValency()
        self.atoms[id2].reduceValency()
        if self.adj is not None:
            if keylist is None: keylist=list(self.atoms.keys())
            i1,i2=keylist.index(id1),keylist.index(id2)
            self.adj[i1,i2]=1.
            self.adj[i2,i1]=1.
        if id1 in self.bondables and self.atoms[id1].valence==0: self.bondables.remove(id1)
        if id2 in self.bondables and self.atoms[id2].valence==0: self.bondables.remove(id2)
        return True
        
        
    
    
class GenericAtomFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.feat_type="atom"
    
    @abc.abstractmethod
    def featurize(self,node,neighs,*args): pass
    
    def __call__(self,node,neighs=None,*args): return self.featurize(node,neighs,*args)
    
class FFOneHotFeaturizer(GenericAtomFeaturizer):
    def __init__(self,ff):
        super(FFOneHotFeaturizer,self).__init__()
        self.ff=ff
        self.atomnames=tuple(self.ff.atom_list.keys())
        self.sz=len(self.atomnames)
        
    def featurize(self,node,neighs,dtype=np.float32):
        ret=np.zeros(self.sz,dtype=dtype)
        ret[self.atomnames.index(node.atomtype)]=1.
        return ret


# In[ ]:


'''
myff=DeNovoForceFieldLoader("/home/venkata/dnv/data/final_ff_parameters.ffin")
myff.loadBonds("/home/venkata/dnv/data/itps/bondtypes.itp")
myff.loadCategories("/home/venkata/dnv/data/categories.data")
myff.loadRules("/home/venkata/dnv/data/definitions.data")



myfeat=FFOneHotFeaturizer(myff)
mymol=DeNovoMolecule("/home/venkata/CADD/data_files/PDBs/result_prot_dnv_CheMBLdist_size11_10000.pdb",ff=myff,includeHs=True)
for i in mymol.atoms: print(i,mymol.atoms[i].atomtype,myff.isSatisfied(mymol.atoms[i],tuple(mymol.getNeighboursOf(i))))
'''
print("DeNovo Reader loaded")


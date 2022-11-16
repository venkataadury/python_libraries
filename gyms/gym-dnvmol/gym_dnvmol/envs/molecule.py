import gym
import itertools
import numpy as np

# import gym_molecule
import copy
import networkx as nx
from gyms.gym_dnvmol.envs.sascorer import calculateScore
from gyms.gym_dnvmol.dataset.dataset_utils import gdb_dataset,mol_to_nx,nx_to_mol
import random
import time
import matplotlib.pyplot as plt
import csv

from contextlib import contextmanager
import sys, os

# block std out
@contextmanager
def nostdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import torch
from CADD.DeNovoReader import *
from CADD.DeNovoGenerativeGraphs import * 

# Parameters
DEVICE = torch.device("cuda") #Change if needed
MAX_SIZE=72

def constantScore(mol,const=0.): return const

class DeNovoMolEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        pass

    def init(self,ff,node_feat,feat_dim=None,molsize=MAX_SIZE,minsize=10,seeds=None,step_rew=0.25,crash_rew=-0.25,term_rew=constantScore,term_prob=None):
        #Seed molecule is a GrowingDeNovoMolecule object. seeds are the list of allowed seed atom-type names (WITH repetitions ALLOWED)
        #term_rew is a function that scores the final DeNovoMolecule Object with a score
        if seeds is None: seeds=list(ff.atom_list.keys())
        if feat_dim is None: feat_dim = node_feat.feat_dim
        self.ff=ff
        self.maxsize=molsize
        self.sizes=np.array(list(range(minsize,self.maxsize+1)),dtype=np.int32)
        self.seedatoms=np.array(seeds,dtype=str)
        self.enc=node_feat
        self.feat_dim=feat_dim
        self.counter = 0
        self.focal=None
        self.mol=None
        self.force_term=term_prob
        if self.force_term is not None:
            self.force_term=np.array([self.force_term(s) for s in range(minsize,self.maxsize+1)])
            self.force_term/=np.sum(self.force_term)
        self.targsize=-1
        
        #Rewards
        self.step_rew=step_rew
        self.err_rew = crash_rew
        self.goal_rew = term_rew
        self.reset()
        #Will add based on what is needed
    
    def getSeedMolecule(self,seeds=None):
        if seeds is None: seeds=self.seedatoms
        seed=np.random.choice(seeds)
        mol=GrowingDeNovoMolecule(MAX_SIZE,self.enc,feat_dim=self.feat_dim,ff=self.ff)
        mol.addAtom(seed,[])
        return mol
    
    def reset(self,seedmol=None): 
        if seedmol is None: seedmol = self.getSeedMolecule()
        self.mol = seedmol
        self.focal = list(self.mol.atoms.keys())[-1]
        ob = self.get_observation()
        self.counter = 0
        if self.force_term is not None: self.targsize=np.random.choice(self.sizes,p=self.force_term)
        return ob
    
    def get_observation(self):
        return self.mol,self.focal
    
    def step(self, action): #Action is a string (atom-type name)
        """
        :param action:
        :return:
        """
        ### init
        info = {}  # info we care about
        total_nodes = len(self.mol.atoms)

        ### take action
        succ=self.mol.addAtom(action,self.focal)
        if succ == False: #Failed
            rew = self.err_rew
            done = False
            pass #TODO: Fix for returning CRASHREWARD
        else:
            rew = self.step_rew #Successful Step
            done = (total_nodes+1 >= self.targsize) or (succ is None)
            if not done: self.focal = succ
            else: rew+=self.goal_rew(self.mol) # Terminal Reward Computation

        # get observation
        ob = self.get_observation()

        self.counter += 1
        return ob, rew, done
        
    
    #Redundant Carry-overs
    def level_up(self): pass
    def render(self, mode='human', close=False): return
    


#Parts of this need to be used in the GrowingDeNovoMolecule Class (e.g. getSmiles at completion)
'''
class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def check_chemical_validity(self):
        """
        Checks the chemical validity of the mol object. Existing mol object is
        not modified. Radicals pass this test.
        :return: True if chemically valid, False otherwise
        """
        s = Chem.MolToSmiles(self.mol, isomericSmiles=True)
        m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
        if m:
            return True
        else:
            return False

    # TODO(Bowen): check if need to sanitize again
    def get_final_smiles(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)
'''
if __name__ == '__main__':
    env = gym.make('dnvmol-v0') # in gym format
    
    #FF Setup
    folder = "/home/venkata/dnv/data"
    myff=DeNovoForceFieldLoader(folder+"/final_ff_parameters.ffin")
    myff.loadBonds(folder+"/itps/bondtypes.itp")
    myff.loadCategories(folder+"/categories.data")
    myff.loadRules(folder+"/definitions.data")
    myff.updateKeyList()
    
    #Featurizer Setup
    myfeat=GraphConvFeaturizer(72,72,myff).to(DEVICE)
    sample_dec=nn.Linear(166*2,1).to(DEVICE)
    hotenc=myfeat.hotenc

    opts=torch.stack([torch.tensor(hotenc(myff.atom_list[atm],None),dtype=torch.float32,device=DEVICE) for atm in myff.atom_list])

    ## debug
    m_env = DeNovoMolEnv()
    seeds = ("CT1","CT2","CT3","CA","NX","CAS","NX","CPT","NY","CY","CT2x","CT1x","CP2","CE1A","CE2A","PL","SL","C3","OH1","OC")
    
    m_env.init(myff,hotenc,molsize=MAX_SIZE,seeds=seeds)
    print("Test Load Complete - Mol Generation Environment")
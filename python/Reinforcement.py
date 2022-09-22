#!/usr/bin/env python
# coding: utf-8

# In[7]:


import random,time
import numpy as np
import gym
from gym import Env,spaces
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.animation as animation
from PIL import Image

import torch

from collections import namedtuple, deque
from itertools import count

import abc
from collections.abc import Iterable
import copy

IS_IPYTHON = 'inline' in matplotlib.get_backend()
if IS_IPYTHON:
    from IPython import display
    #from google.colab.patches import cv2_imshow

DEVICE=torch.device("cuda")
BATCH_SIZE=32
STEP_LIM=100


# In[8]:


def getBoundedObservables(shp,lowv,highv,set_dtype=np.float32):
    return spaces.Box(low = np.ones(shp)*lowv, high = np.ones(shp)*highv, dtype = set_dtype)
class GenericEnvironmentExtension(Env,metaclass=abc.ABCMeta):
    def __init__(self,observe_shape,observe_space,action_space,renderable=False):
        super(GenericEnvironmentExtension, self).__init__()
        
        # Define a 2-D observation space
        self.observation_shape = observe_shape
        self.observation_space = observe_space
        self.ep_return=0
    
        
        # Define an action space ranging from 0 to 4
        self.action_space = action_space #spaces.Discrete(6,)
        
        self.renderable=renderable
                        
        # Create a canvas to render the environment images upon 
        if renderable: self.canvas = np.ones(self.observation_shape) * 1
        
    
    def resetEpisodicReturn(self): self.ep_return=0
    def getEpisodicReturn(self): return self.ep_return
    def addEpisodicReward(self,rew): self.ep_return+=rew
        
    
    def draw_elements_on_canvas(self,elements):
        if not self.renderable: return False
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the heliopter on canvas
        for elem in elements:
            if elem.icon is None: print("None icon:",elem.name)
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        return True
    
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            print("Human Mode is not supported!")
            #cv2.imshow("Window",self.canvas)
            #cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas
    def close(self): pass #cv2.destroyAllWindows() #Overridable if needed
    
    def checkIfActionAllowed(self,action): return self.action_space.contains(action)
    
    @abc.abstractmethod
    def reset(self): pass
    
    @abc.abstractmethod
    def getObservation(self): pass
    
    def step(self,action): #Returns observation (after step), reward (after step), done (Y/N)?, optional extras (in case the function is overridden)
        done = False
        # Assert that it is a valid action
        assert self.checkIfActionAllowed(action), "Invalid Action"
        
        done = done or self.beforeAction(action)
        done = done or self.resolveAction(action)
        done = done or self.afterAction(action)
        
        reward=self.getReward()
        self.addEpisodicReward(reward)
        
        return self.getObservation(), reward, done, []
    
    @abc.abstractmethod
    def resolveAction(self,action): pass
    
    @abc.abstractmethod
    def getReward(self): pass
    
    def beforeAction(self,action): return False
    def afterAction(self,action): return False
        

class GenericDiscreteEnvironmentExtension(GenericEnvironmentExtension,metaclass=abc.ABCMeta):
    def __init__(self,observe_shape,observe_space,nact,renderable=False):
        super(GenericDiscreteEnvironmentExtension,self).__init__(observe_shape,observe_space,spaces.Discrete(nact,),renderable=renderable)
        self.action_meanings=dict()
    
    def getActionMeanings(self): return self.action_meanings
    def addActionMeanings(self,acts,means): self.addActionMeaning(acts,means)
    def addActionMeaning(self,acts,means):
        if not isinstance(acts,Iterable):
            acts=[acts]
            means=[means]
        for i,av in enumerate(acts):
            self.action_meanings[av]=means[i]
    
    def checkIfActionAllowed(self,action): return self.action_space.contains(action)
    def getActionCount(self): return self.action_space.n
    def getObservationShape(self): return self.observation_shape
class IntegerActions:
    def __init__(self,minv,maxv):
        self.minv=minv
        self.maxv=maxv
    
    def contains(self,val): return self.minv <= val and self.maxv>=val

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical


class GenericTorchReinforcementModel(nn.Module):
    def __init__(self,inshape,outshape):
        super(GenericTorchModel,self).__init__()
        self.inshape=inshape
        self.outshape=outshape
    
    def transform(self,x): return x #Overridable
    def resetModel(self): pass #Overridable

class SimpleModel(GenericTorchReinforcementModel): #(D)QN (Q-value predictor) for this problem
    def __init__(self,inshape,outshape,actcount,activation=F.elu):
        super(SimpleModel,self).__init__(inshape,outshape)
        self.acts=actcount
        self.activation=activation
        
        #Layers
        self.layer=nn.Linear(np.prod(inshape),np.prod(outshape))
        
    def forward(self,inp):
        if type(inp)!=torch.Tensor: inp=torch.tensor(inp,dtype=torch.float32).to(DEVICE)
        else: inp=inp.to(DEVICE)
        return self.activation(self.layer(inp))


# In[9]:


class ActorCriticAgent:
    def __init__(self,model,nacts):
        self.num_actions=nacts
        self.basemodel=model
        self.action_weights=nn.Linear(np.prod(model.outshape),nacts) # Probability over action space
        self.action_values=nn.Linear(np.prod(model.outshape),1) # One value prediction for the state
        self.refreshParameterList()
    
    def refreshParameterList(self):
        self.parameterlist=[]
        for param in self.basemodel.parameters(): self.parameterlist.append(param)
        for param in self.action_values.parameters(): self.parameterlist.append(param)
        for param in self.action_weights.parameters(): self.parameterlist.append(param)
    
    def getActionAndValue(self,obs):
        x=self.basemodel(obs)
        actions = F.softmax(self.action_weights(x),dim=-1)
        value = self.action_values(x)
        return actions,value
    
    def parameters(self): return nn.ParameterList(self.parameterlist)
    def to(self,dev):
        self.action_values = self.action_values.to(dev)
        self.action_weights = self.action_weights.to(dev)
        self.basemodel = self.basemodel.to(dev)
        return self
    
    def selectAction(self,obs,eps=0.,training=False,return_value=False,return_distribution=False,return_logits=False):
        if return_logits: return_distribution=True
        with torch.no_grad() if not training else torch.enable_grad():
            acts,val=self.getActionAndValue(obs)
            # create a categorical distribution over the list of probabilities of actions
            m = Categorical(acts)
            # and sample an action using the distribution
            action = m.sample()
        if return_value:
            if return_distribution: return action,val,(acts if return_logits else m)
            else: return action,val
        else: return ((action,(acts if return_logits else m)) if return_distribution else action)

class ActorCriticTrainer: #Episode-wise trainer
    def __init__(self,env,agent,optimizer,learnrate=1e-3):
        self.env=env
        self.agent=agent
        self.optimizer=optimizer(self.agent.parameters(),lr=learnrate) #Reconstruct trainer object to reset optimizer
        
        self.resetMemory()
    
    def resetMemory(self):
        self.saved_actions=[]
        self.rewards=[]
        
    def supervisedTraining(inputs,labels,lossfn,batch_size=BATCH_SIZE,epochs=1): #Both are torch tensors
        for ep in range(epochs):
            losses=[]
            for i in range(inputs.shape[0]//BATCH_SIZE):
                self.optimizer.zero_grad()
                curins=inputs[i*BATCH_SIZE,(i+1)*BATCH_SIZE]
                curouts=labels[i*BATCH_SIZE,(i+1)*BATCH_SIZE]
                modouts=self.agent.action_weights(self.agent.basemodel()) # Note: Softmax NOT APPLIED on purpose
                loss=lossfn(modouts,curouts)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print("Epoch",ep,"with loss",np.mean(losses))
        print("Completed Training")
        
    
    def playEpisode(self,steplim=STEP_LIM,printevery=250,remember=True,batched_action=False):
        obs=self.env.reset()
        total_reward=0
        done = False
        for t in count(1):
            # "training=True" enables gradients (torch autograd)
            act,estval,m=self.agent.selectAction(obs,return_value=True,return_distribution=True,training=True)
            if not batched_action: obs, reward, done, _ = env.step(act.item())
            else: obs, reward, done, _ = env.step(act.detach().cpu().view(-1,1))
            total_reward+=reward
            
            if remember:
                self.saved_actions.append((m.log_prob(act),estval))
                self.rewards.append(reward)
            
            if done or t>steplim: break
            if printevery>0 and (t+1)%printevery==0: print(t,end=" ",flush=True)
        if batched_action:
            total_reward=torch.mean(total_reward).item()
            try: pass
            except:
                print("WARN: Total reward is not a tensor! This might mean that the environment is badly configured to return numeric/non-torch tensors as rewards!\nTrying to directly interpret as a number")
                print("Object is:",total_reward)
                try: total_reward=float(total_reward)
                except:
                    print("Total reward cannot be converted to float.")
                    print("Trying to use numpy instead!")
                    total_reward=float(np.mean(total_reward)) 
                    
        return total_reward,copy.deepcopy(env)
    
    def learnFromMemory(self,clear=True,value_loss=F.smooth_l1_loss,compute_only_loss=False,batched_action=False,batch_aggregation=torch.mean,agg_kws=dict()):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + GAMMA * R
            returns.insert(0, R)
        if not batched_action: print("Found R=",R)

        try: #Assume no batching (`returns` is a list of numbers )
            returns = torch.tensor(returns,dtype=torch.float32,device=DEVICE)
            returns = (returns - returns.mean()) / (returns.std() + EPS_ZERO)
        except: # With batching (`returns` is a list of tensors)?
            returns=torch.stack(returns).to(DEVICE)
            returns = (returns-returns.mean(dim=1).view(-1,1))/(returns.std(dim=1).view(-1,1)+EPS_ZERO)

        for (log_prob, value), R in zip(saved_actions, returns):
            if batched_action and batch_aggregation is not None:
                advantage = batch_aggregation(R - value.detach(),**agg_kws).item()
            else:
                advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            if batched_action: value_losses.append(value_loss(value, R))
            else: value_losses.append(value_loss(value, torch.tensor([R])))

        # reset gradients
        if not compute_only_loss: self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        if clear:
            self.resetMemory()
            self.agent.basemodel.resetModel()
            
        if compute_only_loss: return loss
        
        # perform backprop
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.agent.parameters(), CLIP_NORMS)
        self.optimizer.step() #TODO Enable this to train

        # reset rewards and action buffer
        
        return loss
            
    
    def trainEpisodewise(self,running=20,steplim=STEP_LIM,episode_lim=-1,reward_high=None,batched_action=False,batch_aggregation=None,agg_kws=dict(),save_every=None,save_name="agent.pt",**kwargs):
        if save_every is None: save_every=running
        running_reward=0
        ratio=1/running
        sat=False
        for epno in count(1):
            print("Episode",epno)
            self.resetMemory()
            total_rew,lastenv=self.playEpisode(steplim=steplim,remember=True,batched_action=batched_action,**kwargs)
            print("Played")
            self.learnFromMemory(clear=True,batched_action=batched_action,batch_aggregation=batch_aggregation,agg_kws=agg_kws)
            
            if episode_lim>0 and epno>=episode_lim: break
            running_reward=(1-ratio)*running_reward+total_rew
            print("Running Reward:",running_reward)
            
            if reward_high is not None:
                #if reward_high<running_reward and epno>running:
                if reward_high<running_reward and epno>running: # Treating rewards as loss?
                    print("Target reward reached!")
                    sat=True
                    break
            if save_every>0 and epno%save_every==0:
                torch.save(self.agent,save_name)
                print("Saved model at '"+str(save_name)+"'")
        torch.save(self.agent,save_name)
        return sat
    
    def trainBatchwise(self,batch_size,running=20,steplim=STEP_LIM,batch_lim=-1,reward_high=None,**kwargs):
        self.lossmem=[]
        running_reward=0
        ratio=1/running
        sat=False
        batchK=0
        for epno in count(1):
            print("Episode",epno)
            self.resetMemory()
            total_rew,lastenv=self.playEpisode(steplim=steplim,remember=True,**kwargs)
            print("Played")
            myloss=self.learnFromMemory(clear=True,compute_only_loss=True)
            self.lossmem.append(myloss)
            
            if len(self.lossmem) >= batch_size:
                self.optimizer.zero_grad()
                final_loss=torch.sum(torch.stack(self.lossmem))
                final_loss.backward()
                self.optimizer.step()
                self.lossmem=[]
                batchK+=1
            
            if batch_lim>0 and batchK>=batch_lim: break
            running_reward=(1-ratio)*running_reward+total_rew
            print("Running Reward:",running_reward)
            
            if reward_high is not None:
                if reward_high<running_reward:
                    print("Target reward reached!")
                    sat=True
                    break
        return sat


# In[ ]:


def combineParameters(parsets):
    ret=[]
    for paramlist in parsets:
        for pobj in paramlist: ret.append(pobj)
    return torch.nn.ParameterList(ret)


# In[ ]:


print("Reinforcement Module Loaded")


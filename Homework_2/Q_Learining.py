import numpy as np
import pandas as pd
import gymnasium as gym
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

'''
Il Replay buffer serve nel Q-Learning per memorizzare le esperienze passate e campionandole per stabilizzare l'addestramento.
Infatti nel REINFORCE classico le esperienze passate sono correlate tra loro e questo è uno dei punti che rende instabiole l'addestramento.
Il compito del REplay buffer e quello di memorizzare le esperienze passate e campionarle per minibatch in maniera che siano casuali.
Un altra importante caratteristica è che esso è circolare, ovvero ha una capacità limitata quindi quando raggiunge questo limite allora inizia
a sovrascrivere le esperienze nuve su quelle passate 
'''
class ReplayBuffer:
    def __init__(self, capacity, obs_shape,device):
        self.capacity= int(capacity)
        self.device= device
        self.pos= 0
        self.size=0

        obs_dim= np.prod(obs_shape)
        #allocazione memoria del buffer
        #allocazione dello stato attuale 
        self.observation_buffer = torch.zeros((self.capacity, obs_dim))
        #allocazione dello stato successivo
        self.next_observation_buffer= torch.zeros((self.capacity, obs_dim))
        #allocazione per le azioni compiute
        self.acttion_buffer= torch.zeros((self.capacity,))
        #allocazione dei reward ottenuti
        self.reward_buffer= torch.zeros((self.capacity,))
        #allocazione se episodio finito
        self.done_buffer= torch.zeros((self.capacity,))
    '''
    In questo metodo aggiungiamo la singola transazione compiuta, in questo caso viene data la circolarità del buffer tramite il calcolo della posizione
    nel buffer in modo circolare 
    '''
    def add(self, observation, action, reward, next_observation,done):
        #Trasformo in array perchè solo al momento del sampling serve che siano tensori se no troppa allocazione
        #di memoria
        self.observation_buffer[self.pos]= np.array(observation).reshape(-1)
        self.next_observation_buffer[self.pos]= np.array(next_observation).reshape(-1)
        self.acttion_buffer[self.pos]= action
        self.reward_buffer[self.pos]= reward
        self.done_buffer[self.pos]= float(done)

        #Quando pos arriva alla fine riparte con il primo indice nel tensore
        self.pos= (self.pos + 1) % self.capacity
        #ovviamente si carica in memoria dei tensori che hanno già capacità massima quindi devo aggiornare anche la size
        #in modo tale che quando campiono non vado a prendere posizioni vuote all' interno dei tensori
        self.size= min(self.size+1, self.capacity)
        #in questo modo scelgo il minimo e quindi sono sempre dentro la grandezza effettiva del tensore diverso da 0

    def sample(self, batch_size):
        
        #in questo modo creo un array di indici random per la grandezza del batch
        idxs= np.random.randint(0, self.size, size=batch_size)

        obs= torch.tensor(self.observation_buffer[idxs], device=self.device)
        actions= torch.tensor(self.acttion_buffer[idxs], device=self.device)
        rewards= torch.tensor(self.reward_buffer[idxs],device = self.device)
        next_obs= torch.tensor(self.next_observation_buffer[idxs], device=self.device)
        done = torch.tensor(self.done_buffer[idxs], device = self.device)

        return obs, actions, rewards, next_obs, done
    
    def __len__(self):
        return self.size


'''Andiamo adesso a definire la rete QNetwork, ovvero colei che approssima la funzione Q 
Per cartpole o lunar si può definire delle reti semplici composi da un MLP
'''

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_action, hidden_size= 128 ):
        super(QNetwork,self).__init__()
        self.net= nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_action),
        )
    def forward(self,x):
        return self.net(x)
    




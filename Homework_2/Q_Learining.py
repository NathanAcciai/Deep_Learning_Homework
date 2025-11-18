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


'''Definiamo adesso l'agente che sfrutta il Q-Learning
Definisco anche una epsilon per l'esplorazione in modo tale da avere un comportamento più esplorativo all' inziio
mentre andando avanti riduco portando la rete a un comportameneto più deterministico e quindi selezioni di azioni migliori.
Si parte con un epsilon alto quindi azione casuale e piano piano va a descrescere prendendo azioni sempre migliori.
'''
'''
Piccolo sunto della rete target per ricordare il motivo del perchè risulta essere una copia della rete online.
La rete target serve a dare un obbiettivo stabile alla rete target anche se questo è abbastanza distante dalla realtà perchè 
il problema del RL era che ogni volta la rete cambiava i propri target dunque era come avvicinarsi a un target che successivamnete si sarebbe mosso
questo ovviamente portava molta instabilità, ma con l'aggiunta della rete target e con un lr sufficientemente basso allora nache se essa diverge dalla realtà
quindi diverge dalla stima vera del target comunque offre alla rete online un ancora sotto cui poter convergere e quindi con l'aggiornamento della rete
target anche l'errore rispetto alla realtà va a diminuire.
Questo anche perchè nel buffer si ha una storia delle esperinze e quindi andando comunque a riempirlo con nuove esperienze dove anche la scelta delle azioni diventa sempre
più deterministica.
Infatti inizialmente si ha un alta esplorazione per cercare le azioni migliori fino a migliorare sulla base delle esperienze avute
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.
'''

class DQNAgent:
    def __init__(self,
                 obs_dim,
                 n_action,
                 replay_buffer: ReplayBuffer,
                 lr=0.001,
                 gamma=0.99,
                 batch_size=16,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay= 10000,
                 device="cpu",
                 target_update_freq= 1000):
        
        self.device= device
        self.replay_buffer= replay_buffer
        self.n_action= n_action
        self.lr= lr
        self.gamma= gamma
        self.batch_size= batch_size
        self.espilon_start= epsilon_start
        self.epislon_end=epsilon_end
        self.epsilon_decay= epsilon_decay
        self.target_update_freq= target_update_freq
        self.total_step=0

        #Rete online si aggiorna passo dopo passo e vi calcolo la LOSS
        self.q_network= QNetwork(obs_dim=obs_dim, n_action=n_action).to(device)
        #rete che viene aggiornata periodicamente e serve per stabilizzare l'addestramento 
        #poichè mi da il massimo ritorno atteso nello stato successivo
        self.q_target= QNetwork(obs_dim=obs_dim, n_action=n_action).to(device)
        #partono con gli stessi pesi 
        self.q_target.load_state_dict(self.q_network)

        self.optimizer= optim.AdamW(params=self.q_network.parameters(), lr=lr)
        self.epsilon= epsilon_start

    #aggiorno la epsilon in modo da avere un carattere più stocastico all' inizio quindi di scelta casuale e maggiroe esplorazione
    #Piano piano la epsilon decresce del fattore diventando al minimo 0.05 e quindi porta a un comportamento deterministico
    def update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.total_steps / self.epsilon_decay)
        )

    def take_action(self,state):
        self.total_step += 1
        self.update_epsilon()

        #numero casuale tra 0 e 1 , fa si che quando epsilon scende allora avrò sempre meno esplorazioni.
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_action)
        else:
            state= torch.tensor(state, device=self.device).unsqueeze(0)
            q_values= self.q_network(state)
            return int(torch.argmax(q_values, dim=1).item())
    '''da ricordare non si mette model.train o eval perchè non ho dropout ne batch norm essendo che la rete è un semplice MLP
    ''' 
    def training_step(self):
        #prima di tutto devo controllare se il replay buffer ha una dimensione pari a un batch 
        if len(self.replay_buffer)< self.batch_size:
            return
        
        obs,action, reward,next_obs,done =self.replay_buffer.sample(self.batch_size)
        #ritorno atteso della rete principale
        q_values= self.q_network(obs)
        #essendo che ho un tensore della forma [batch_size, num_action], devo prendere le azioni che sono state intraprese con [batch,1]
        #Prendo le azioni del sample perchè ho la descrizione dell' esperienza avuta
        q_values= q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            #calcolo le ricompense target
            next_q= self.q_target(next_obs)
            #prendo la ricompensa massima
            next_q_max= next_q.max(dim=1)[0]
            #ora calolco la ricompensa target y
            # r_t + gamma (1-d_t) * max Q(s_t+1,a')
            target= reward + self.gamma* (1-done)*next_q_max
        #ora che ho la ricompensa target posso calcolare MSE per la loss rispetto a quella della rete
        loss= nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #aggiorno la rete target in modo lento in maniera che abbia stabilità durante il training
        
        if self.total_step % self.target_update_freq==0:
            self.q_target.load_state_dict(self.q_network.state_dict())

def train(env, agent: DQNAgent, num_episode= 500):
    #collezziono i ritorni per ogni episodio
    episode_rewards=[]
    for episode in range(num_episode):
        obs= env.reset()[0]
        episode_reward=0
        done=False

        while not done:
            #l'agente prende un azione con epsilon greedy inizialmente
            action= agent.take_action(obs)
            #vado dunque a eseguire l'azione nell' ambiente 
            next_obs, reward,terminated, truncated,_= env.step(action)
            #verifico se l'episodio è finito perchè fallito oppure terminato
            done = terminated or truncated
            #aggiungo esperienza al replay buffer
            agent.replay_buffer.add(
                obs,action, reward, next_obs, done
            )
            #step di training dell'agente sulla base di quello che è successo 
            agent.training_step()
            obs= next_obs
            #sommo le reward che ho ottenuto
            episode_reward+=reward
        
        episode_rewards.append(episode_reward)
        print(f'Episode {episode}, reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}')
    
    return episode_rewards

def evaluation(env, agent, num_episode):
    
    episode_rewards=[]
    for episode in range(num_episode):
        


    




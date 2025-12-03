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
from tqdm import tqdm
import copy


import random
import numpy as np
import torch


"""Settare TUTTI i seed per riproducibilità completa"""
seed=(111)
# 1. Python random
random.seed(seed)

# 2. NumPy (per sampling nel replay buffer, epsilon-greedy)
np.random.seed(seed)

# 3. PyTorch (pesi della rete, dropout se presente)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # se multi-GPU

# 4. PyTorch CUDA backend (per performance riproducibili)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 5. Environment (se supporta il seed)
# Nota: alcuni ambienti non sono completamente deterministici

    
'''
Il Replay buffer serve nel Q-Learning per memorizzare le esperienze passate e campionandole per stabilizzare l'addestramento.
Infatti nel REINFORCE classico le esperienze passate sono correlate tra loro e questo è uno dei punti che rende instabiole l'addestramento.
Il compito del REplay buffer e quello di memorizzare le esperienze passate e campionarle per minibatch in maniera che siano casuali.
Un altra importante caratteristica è che esso è circolare, ovvero ha una capacità limitata quindi quando raggiunge questo limite allora inizia
a sovrascrivere le esperienze nuve su quelle passate 
'''
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device="cpu"):
        self.capacity = int(capacity)
        self.device = device
        self.pos = 0
        self.size = 0

        obs_dim = np.prod(obs_shape)
        # allocazione memoria del buffer come tensori PyTorch sul device
        self.observation_buffer = torch.zeros((self.capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.next_observation_buffer = torch.zeros((self.capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.action_buffer = torch.zeros((self.capacity,), dtype=torch.long, device=self.device)
        self.reward_buffer = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.done_buffer = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)

    def add(self, observation, action, reward, next_observation, done):
        # converto direttamente in tensori sul device
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(-1)
        next_obs_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device).reshape(-1)
        action_tensor = torch.as_tensor(action, dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # salvo nel buffer
        self.observation_buffer[self.pos] = obs_tensor
        self.next_observation_buffer[self.pos] = next_obs_tensor
        self.action_buffer[self.pos] = action_tensor
        self.reward_buffer[self.pos] = reward_tensor
        self.done_buffer[self.pos] = done_tensor

        # gestione circolare del buffer
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = self.observation_buffer[idxs]
        actions = self.action_buffer[idxs]
        rewards = self.reward_buffer[idxs]
        next_obs = self.next_observation_buffer[idxs]
        done = self.done_buffer[idxs]

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
                 writer:SummaryWriter,
                 device="cpu",
                 path_experiment=None,
                 hidden_size= 128,
                 lr=0.0001,
                 gamma=0.99,
                 batch_size=16,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay= 20000
                 ):
        
        self.device= device
        self.replay_buffer= replay_buffer
        self.n_action= n_action
        self.writer= writer
        self.lr= lr
        self.gamma= gamma
        self.batch_size= batch_size
        self.epsilon_start= epsilon_start
        self.epsilon_end=epsilon_end
        self.epsilon_decay= epsilon_decay
        self.total_step=0
        self.start_episode=0
        self.path_experiment= path_experiment
        self.best_value_valid= -float("inf")
        self.hidden_size= hidden_size
        self.tau= 0.005

        #Rete online si aggiorna passo dopo passo e vi calcolo la LOSS
        self.q_network= QNetwork(obs_dim=obs_dim, n_action=n_action, hidden_size=hidden_size).to(device)
        #rete che viene aggiornata periodicamente e serve per stabilizzare l'addestramento 
        #poichè mi da il massimo ritorno atteso nello stato successivo
        self.q_target= QNetwork(obs_dim=obs_dim, n_action=n_action,hidden_size=hidden_size).to(device)
        #partono con gli stessi pesi 
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.optimizer= optim.AdamW(params=self.q_network.parameters(), lr=lr)
        self.epsilon= epsilon_start
        if os.path.exists(self.path_experiment):
            self._load_checkpoint()
        else:
            os.makedirs(self.path_experiment, exist_ok=True)
            self._save_hyperparamtres()


    def _save_hyperparamtres(self):
        dict_hp={
            'learning rate': self.lr,
            'batch size': self.batch_size,
            'epsilon start': self.epsilon_start,
            'epsilon decay': self.epsilon_decay,
            'gamma':self.gamma,
            "hidden_size": self.hidden_size
        }

        df= pd.DataFrame(dict_hp, index=[0])
        df.to_csv(os.path.join(self.path_experiment,"hyperparametres.csv"), index=False)

    def _save_checkpoint(self,episode,checkpoint=False,data=None):
        if data is None:
            data={
                'episode': episode,
                'epsilon': self.epsilon,
                'QNetwork': self.q_network.state_dict(),
                'QTarget': self.q_target.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_value_validation': self.best_value_valid
            }
        if checkpoint:
            path_save = os.path.join(self.path_experiment,"checkpoint.pth")
        else:
            path_save= os.path.join(self.path_experiment,"best_model.pth")
        torch.save(data, path_save)
        return data


    def _load_checkpoint(self,path):
        data= torch.load(path, map_location="cpu")
        self.q_network.load_state_dict(data["QNetwork"])
        self.q_target.load_state_dict(data["QTarget"])
        self.epsilon= data["epsilon"]
        self.start_episode= data["episode"]+1
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.best_value_valid= data["best_value_validation"]

        
    #aggiorno la epsilon in modo da avere un carattere più stocastico all' inizio quindi di scelta casuale e maggiroe esplorazione
    #Piano piano la epsilon decresce del fattore diventando al minimo 0.05 e quindi porta a un comportamento deterministico
    def _update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.total_step / self.epsilon_decay)
        )
        


    def _update_target_network(self):
    # Soft update: θ_target = τ*θ_online + (1-τ)*θ_target
        for target_param, online_param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )

    def _take_action(self,state):
        self.total_step += 1
        self._update_epsilon()

        #numero casuale tra 0 e 1 , fa si che quando epsilon scende allora avrò sempre meno esplorazioni.
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_action)
        else:
            state= torch.tensor(state, device=self.device).unsqueeze(0)
            q_values= self.q_network(state)
            return int(torch.argmax(q_values, dim=1).item())
    '''da ricordare non si mette model.train o eval perchè non ho dropout ne batch norm essendo che la rete è un semplice MLP
    ''' 
    def _training_step(self, episode):
        #prima di tutto devo controllare se il replay buffer ha una dimensione pari a un batch 
        self.q_network.train() 
        if len(self.replay_buffer)< self.batch_size:
            return
        criterion = nn.SmoothL1Loss(reduction='mean')  
        obs,action, reward,next_obs,done =self.replay_buffer.sample(self.batch_size)

        
        #ritorno atteso della rete principale
        q_values= self.q_network(obs)
        #essendo che ho un tensore della forma [batch_size, num_action], devo prendere le azioni che sono state intraprese con [batch,1]
        #Prendo le azioni del sample perchè ho la descrizione dell' esperienza avuta
        q_values= q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            #selezione azione migliore della rete online
            next_q = self.q_target(next_obs)
            next_q_max = next_q.max(dim=1)[0]
            target = reward + self.gamma * (1 - done) * next_q_max
        #ora che ho la ricompensa target posso calcolare MSE per la loss rispetto a quella della rete
        
        loss= criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        #aggiorno la rete target in modo lento in maniera che abbia stabilità durante il training
        
        self._update_target_network()
        self.writer.add_scalar("Training/Loss (MSE)", loss.item(), episode)

    def train(self,env, env_val,num_episode= 500,num_episode_val= 10):
        #collezziono i ritorni per ogni episodio
        avg_reward_validation=-float("inf")
        episode_rewards=[]
        for episode in tqdm(range(self.start_episode, num_episode), desc= "Train Epsiode"):
            obs= env.reset()[0]
            episode_reward=0
            done=False
            self.q_network.train()

            while not done:
                #l'agente prende un azione con epsilon greedy inizialmente
                action= self._take_action(obs)
                #vado dunque a eseguire l'azione nell' ambiente 
                next_obs, reward,terminated, truncated,_= env.step(action)
                #verifico se l'episodio è finito perchè fallito oppure terminato
                done = terminated or truncated
                #aggiungo esperienza al replay buffer
                self.replay_buffer.add(
                    obs,action, reward, next_obs, done
                )
                #step di training dell'agente sulla base di quello che è successo 
                self._training_step(episode)
                obs= next_obs
                #sommo le reward che ho ottenuto
                episode_reward+=reward
            
            episode_rewards.append(episode_reward)
            tqdm.write(f'Episode {episode}, reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}')
            if episode %10==0:
                avg_reward_validation=self.evaluation(env_val , num_episode_val)
                self.writer.add_scalar("Evaluetion/Average Reward",avg_reward_validation,episode)

            if avg_reward_validation>= self.best_value_valid:
                self.best_value_valid= avg_reward_validation
                best_model= self._save_checkpoint(episode, True)
                
            self.writer.add_scalar("Training/Reward", episode_reward,episode)
        self._save_checkpoint(episode,False,best_model)
        return episode_rewards

    def evaluation(self,env, num_episode_val):
        #serve per ripristinare la epsilon che ho in train 
        old_epsilon=copy.deepcopy(self.epsilon)
        #imposto quindi la scelta deterministica dell' azione (greedy)
        self.epsilon=0
        self.q_network.eval()
        
        episode_rewards=[]
        with torch.no_grad():
            for episode in tqdm(range(num_episode_val),desc="Evaluetion Epsiode",leave=False):
                obs= env.reset()[0]
                done =False
                episode_reward=0

                while not done:
                    obs_tensor= torch.tensor(obs, device= self.device).unsqueeze(0)
                    q_values= self.q_network(obs_tensor)
                    action= int(torch.argmax(q_values, dim=1).item())

                    next_obs,reward,terminate,truncated, _= env.step(action)
                    done = terminate or truncated

                    obs = next_obs
                    episode_reward+=reward
                episode_rewards.append(episode_reward)
        #ripristino la epsilon greedy 
        self.epsilon= old_epsilon
        avg_reward= sum(episode_rewards)/ num_episode_val
        tqdm.write(f'Avergae reward of evaluation of agent network {avg_reward}')
        return avg_reward





        




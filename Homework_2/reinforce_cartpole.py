import numpy as np
import gymnasium as gym
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os


"""
Prima di tutto si costruisce la classe della policy da apprendere

"""

class PolicyNetwork(nn.Module):
    def __init__(self,obs_dim, action_dim, hidden_size=128):
        super(PolicyNetwork,self).__init__()
        self.fc1= nn.Linear(obs_dim, hidden_size)
        self.fc2= nn.Linear(hidden_size, action_dim)
    
    def forward(self, x,tempreature=1.0):
        x= F.relu(self.fc1(x))
        logits= self.fc2(x)/ tempreature
        return F.softmax(logits, dim=-1)
    
"""
Definiamo adesso invece la scelta dell'azione tramite una distribuzione categorica essendo ovviamente che si parla
di uno spazio deterministico quindi in out dovro avere solo una scelta.
il fattore di temperatura agisce come normalizzatore :
se i fattore è vicino allo 0 ho un comportamento deterministico essendo che non fa esplodere i valori più alti nel softmax
se il fattore è 1 ovviamnete si ha un comportamento stocastico normale della distribuzione in uscita
se il valore è maggiore allora si porta i valori più vicini tra loro quindi la rete cerca di esplorare maggiormentne l'azioen da prendere
Nel caso infatti T tendesse a infinito avremmo uno scenario teorico dove tutte le azioni hanno la stessa probabilità

"""

class ReinforceAgent(nn.Module):
    def __init__(self,
                enviroment,
                logdir,
                policy,
                gamma=0.99,
                max_lenght=500
                  
                ):
        super(ReinforceAgent,self).__init__()
        self.env= enviroment
        self.gamma= gamma
        self.file_writer= SummaryWriter(logdir)
        self.policy= policy
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_lenght= max_lenght
    """
    Decido come prendere l'azione, quindi la rete ritorna dei logits puri poi si va ad applicare la temperatiura
    in modo tale da decidere che atteggiamento deve avere la policy
    Bisogna fare una distinzione anche tra train e validation, perchè in validation vogliamo un comportamento del tutto deterministico impostando la temperatura a 1 e applciando
    un semplice softmax per l'azione più probabile
    """
    def select_action(self, obs,temperature=1.0,train=True):
        obs= torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.policy(obs)
        logits_norm= logits/ temperature

        if train:
            dist= Categorical(logits=logits_norm)
            action = dist.sample()
            log_prob= dist.log_prob(action)
            action= action.item()
        else:
            action = torch.argmax(logits_norm).item()
            log_prob = None

        return action, log_prob
    """
    In questo caso si va a calcolare i ritorni scontati di tutti i passi
    per la normalizzazione si usa una normalizzazione standard, che rispetto a una
    normalizzazione tra minmax che è sensibile a outliers e perchè è garantisce stabilità nei gradienti
    Quindi per normalizzare i ritorni implemnetimo una versione con una baseline(che sarà la media dei ritorni)
    e una versione invece con una normalizzazione normale 
    """
    def compute_discount_returns(self, rewards,normalize=False, baseline=False):
        returns=[]
        G=0
        for r in reversed(rewards):
            G= r + self.gamma *G
            returns.insert(0,G)
        returns = torch.tensor(returns, device = self.device)
        if baseline:
            returns= returns- returns.mean()
        elif normalize:
            mean = returns.mean()
            std = returns.std()
            returns = (returns - mean) / (std + 1e-8)
        
        return returns
    """
    questo codice run il singolo episodio dove si calcola l'azione della policy e ovviamente si applica allo step dell'enviroment
    dato questo si deve capire anche se l'enviroment ha riportato un fallimento oppure se ha completato il suo obbiettivo
    """
    def run_episode(self, temperature,train=True):
        observations = []
        actions = []
        log_probs = []
        rewards = []

        (obs, _)= self.env.reset()
        for _ in range(self.max_lenght):
            obs= torch.tensor(obs, device=self.device)
            (action, log_prob)= self.select_action(obs,temperature, train)
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)

            (obs, reward, term, trunc, _ )= self.env.step(action)
            rewards.append(reward)
            if term or trunc:
                break
        return torch.cat(log_probs), rewards
    
"""
La classe sotto è una generalizazione di un trainer per qualsiasi agente con una qualsiasi policy, quindi non importa che policy o agente è stato implemnetato
"""

class TrainAgentRenforce(nn.Module):
    def __init__(self,
                 reinforcagent : ReinforceAgent,
                 lr= 1e-2,
                 num_episode= 10,
                 num_episode_validation=10,
                 check_val=10,
                 checkpoint_path="checkpoint.pt",
                 best_model_path="best_model.pt"
                ):
        super(TrainAgentRenforce,self).__init__()
        self.reinforceagent= reinforcagent
        self.num_episode= num_episode
        self.num_episode_validation= num_episode_validation
        self.check_val= check_val
        self.lr= lr
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path= checkpoint_path
        self.best_model_path= best_model_path
        self.opt= optim.Adam(self.reinforceagent.policy.parametres(), lr=self.lr)
        self.best_eval_reward = -float("inf")
        self.start_episode = 1

        os.makedirs(self.best_model_path, exist_ok=True)

        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)
            print(f"✅ Checkpoint founded {self.start_episode}")
        else:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            print("🚀 New Training")

    """
    In questo si salva il checkpoint per poter ripartire nel caso di crash, oppure per salvare il miglior modello trovato secondo una specifica metrica.
    Bisogna però fare una distinzione tra quale dei due si sta sviluppando.
    """
    def save_checkpoint(self, episode,best_eval, checkpoint=True):
        data={
            'episode': episode,
            'policy_state_dict': self.reinforceagent.policy.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'best_eval_reward': best_eval,
        }
        path_save= self.checkpoint_path if checkpoint else self.best_model_path
        torch.save(data, path_save)


    """
    Si verifica che se esiste un checkpoint allora va ricaricato e si riaprte ad allenare da questo
    Esso ovviamente gestisce qualsiasi situazione deve infatti ripristinare se c'è stato un crash di qualsiasi tipo ricaricando lo stato salvato
    """
    def load_checkpoint(self, path):
        data = torch.load(path, map_location="cpu")
        self.reinforceagent.policy.load_state_dict(data["policy_state_dict"])
        self.opt.load_state_dict(data["optimizer_state_dict"])
        self.best_eval_reward = data["best_eval_reward"]
        self.start_episode = data["episode"] + 1


    def evaluate(self):
        self.reinforceagent.policy.eval()
        total_reward=0
        with torch.no_grad():
            for _ in range(self.num_episode_validation):
                log_probs, rewards = self.reinforceagent.run_episode(1,False)
                total_reward += rewards

        avg_reward = total_reward / self.num_episode_validation 
        return avg_reward
    #da rivedere questa cosa, le metriche in genberale

                
    def train_agent(self, temperature_train, temoerature_validation):

        running_rewards= []

        self.reinforceagent.policy.train()
    
        for episode in range(self.num_episode):

            log_probs, rewards= self.reinforceagent.run_episode(temperature_train)

            returns= torch.tensor(self.reinforceagent.compute_discount_returns(rewards,True), device= self.device)
        
            self.opt.zero_grad()
            loss = (-log_probs * returns).mean()
            loss.backward()
            self.opt.step()
            
            running_rewards.append(sum(rewards))

            if episode % self.check_val==0:
                pass #da aggiungere la validazione


            
        
        return running_rewards



        


        

        

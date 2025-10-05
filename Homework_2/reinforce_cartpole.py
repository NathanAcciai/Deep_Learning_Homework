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
                name_agent,
                gamma=0.95,
                max_lenght=500
                  
                ):
        super(ReinforceAgent,self).__init__()
        self.env= enviroment
        self.gamma= gamma
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)
        self.file_writer= SummaryWriter(logdir)

        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy= policy.to(self.device)
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
            log_prob= log_prob.reshape(1)
            action= action.item()
        else:
            action = torch.argmax(logits_norm).item()
            log_prob = torch.tensor([], device=self.device)

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
    def run_episode(self, temperature,termination_value,train=True):
        observations = []
        actions = []
        log_probs = []
        rewards = []
        
        
        (obs, _)= self.env.reset()
        for _ in range(self.max_lenght):
            obs= torch.tensor(obs, dtype=torch.float32, device=self.device)
            (action, log_prob)= self.select_action(obs,temperature, train)
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)

            (obs, reward, term, trunc, _ )= self.env.step(action)
            rewards.append(reward)
            #Solo per cartpolevale la seguente termination values
            if trunc:
                termination_value["success"]+=1
                break
            elif term:
                termination_value["failure"]+=1
                break
        return torch.cat(log_probs), rewards, termination_value
    
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
        self.file_witer= self.reinforceagent.file_writer
        self.num_episode= num_episode
        self.num_episode_validation= num_episode_validation
        self.check_val= check_val
        self.lr= lr
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path= checkpoint_path
        self.best_model_path= best_model_path
        self.opt= optim.Adam(self.reinforceagent.policy.parameters(), lr=self.lr)
        self.best_eval_reward = -float("inf")
        self.start_episode = 1
        self.termination_value_train={"failure": 0, "success":0}
        self.termination_value_test={"failure": 0, "success":0}
        os.makedirs(self.best_model_path, exist_ok=True)

        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)
            print(f"Checkpoint founded {self.start_episode}")
        else:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            print("Checkpoint not founded, start a new Training")

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
        
        if checkpoint:
            path_save=os.path.join(self.checkpoint_path,'checkpoint.pt')
        else:
            path_save=os.path.join(self.best_model_path,'best_model.pt')
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

    """
    Per la validazione si useranno più metriche per il modello così da avere più informazioni su come esso si comporta e non solo i reward applicati
    media: ovviamente la prendiamo per capire i reward medi per ogni singolo episodio
    deviazione standard: misura la stabiliutà dell'agente se è bassa ovviamente l'agente performa stabilmente ed è consistente
    massimo-minimo reward: Indica la misura massima che l'agente ha ricevuto come ricomopensa per episodio 
    percentile: il valore sotto il quale si trova il 10&% delle ricompense peggiori, utile per capire le performance dell'agente nei casi peggiori
    """

    def evaluate(self,episode):
        self.reinforceagent.policy.eval()
        total_reward=[]
        episode_lenght=[]
        with torch.no_grad():
            for _ in range(self.num_episode_validation):
                log_probs, rewards,self.termination_value_test = self.reinforceagent.run_episode(1,self.termination_value_test,False)
                total_reward.append(sum(rewards))
                episode_lenght.append(len(rewards))

            all_reward= np.array(total_reward)
            mean_reward= all_reward.mean()
            avg_lenght_episode= np.mean(episode_lenght)
            std= all_reward.std()
            min_reward= all_reward.min()
            max_reward= all_reward.max()
            percentile_reward= np.percentile(all_reward, 10)

            self.file_witer.add_scalar("Evaluation/Mean", mean_reward, episode)
            self.file_witer.add_scalar("EValuation/Average Lenght Episode", avg_lenght_episode, episode)
            self.file_witer.add_scalar("Evaluation/Standard Deviation", std, episode)
            self.file_witer.add_scalar("Evaluation/Min Reward", min_reward, episode)
            self.file_witer.add_scalar("Evaluation/Max Reward", max_reward,episode)
            self.file_witer.add_scalar("Evaluation/Percentile 10%",percentile_reward, episode)
            for type_term, count in self.termination_value_test.items():
                self.file_witer.add_scalar(f'Evaluation/Termination {type_term}', count, episode)
        return mean_reward
    

                
    def train_agent(self, temperature_train, normalizzation_discount, baseline_discount):

        running_rewards= []

        for episode in range(self.num_episode):

            self.reinforceagent.policy.train()

            log_probs, rewards, self.termination_value_train= self.reinforceagent.run_episode(temperature_train,self.termination_value_train)

            returns= torch.tensor(self.reinforceagent.compute_discount_returns(rewards,normalizzation_discount,baseline_discount), device= self.device)
        
            self.opt.zero_grad()
            loss = (-log_probs * returns).mean()
            loss.backward()
            self.opt.step()
            
            running_rewards.append(sum(rewards))
            self.file_witer.add_scalar("Training/Loss", loss.item(), episode)
            self.file_witer.add_scalar("Training/Reward", sum(rewards), episode)
            for term_type, count in self.termination_value_train.items():
                self.file_witer.add_scalar(f"Evaluation/Termination_{term_type}", count, episode)

            if episode % self.check_val==0:
                avg_reward_val= self.evaluate(episode)
                
                if avg_reward_val>= self.best_eval_reward:
                    self.best_eval_reward= avg_reward_val
                    self.save_checkpoint(episode,avg_reward_val,False)
                else:
                    self.save_checkpoint(episode,self.best_eval_reward,True)

                print(f'Running reward train from episode {episode}: {running_rewards[-1]}')
                print(f'Average reward test from episode {episode}: {avg_reward_val}\n')

        return running_rewards



        


        

        

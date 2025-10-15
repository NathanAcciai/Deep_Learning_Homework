# %% [markdown]
# # Deep Reinforcement Learning Laboratory
# 
# In this laboratory session we will work on getting more advanced versions of Deep Reinforcement Learning algorithms up and running. Deep Reinforcement Learning is **hard**, and getting agents to stably train can be frustrating and requires quite a bit of subtlety in analysis of intermediate results. We will start by refactoring (a bit) my implementation of `REINFORCE` on the [Cartpole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/). 

# %% [markdown]
# ## Exercise 1: Improving my `REINFORCE` Implementation (warm up)
# 
# In this exercise we will refactor a bit and improve some aspects of my `REINFORCE` implementation. 
# 
# **First Things First**: Spend some time playing with the environment to make sure you understand how it works.

# %%
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Instantiate a rendering and a non-rendering environment.
env_render = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')

# %% [markdown]
# CartPole è un carrello che può muoversi a sinistra o a destra su un binario, sopra il carello è montato un pali collegato con una cerniera alla base, l'agente deve bilanciare il palo muovendo il carello 
# Stati (osservazioni):
# 1) posizione del carello
# 2) velocità del carello 
# 3) angolo del palo
# 4) velocità angolare del palo 
# 
# Azioni: 
# 0) spinge il carello a sinistra
# 1) spinge il carello a destra
# 
# Ricompensa:
# Se il palo rimane in piedi allora avro una reward +1
# 
# Condizioni di fine episodio
# 1) angolo troppo grande: Il palo cade
# 2) carello esce dai limiti di pista
# 3) si supera 500 passi
# 
# Obbiettivo: imaprare una politica che muova il carello in modo da non far cadere il palo il più a lungo possibile

# %%
#obs, info= env.reset()
#
#print(f'Observation after the reset: {obs}')
#print(f'information of action {info}')
#
#print("Observation space:", env.observation_space)
#print("Action space:", env.action_space)


# %% [markdown]
# observation space: [posizione carello, velocità carello, angolo del palo, velocità angolare del palo]
# action space: (discreto) [0 sposta a sinistra, 1 sposta a destra]

# %% [markdown]
# **Next Things Next**: Now get your `REINFORCE` implementation working on the environment. You can import my (probably buggy and definitely inefficient) implementation here. Or even better, refactor an implementation into a separate package from which you can `import` the stuff you need here. 

# %% [markdown]
# **Last Things Last**: My implementation does a **super crappy** job of evaluating the agent performance during training. The running average is not a very good metric. Modify my implementation so that every $N$ iterations (make $N$ an argument to the training function) the agent is run for $M$ episodes in the environment. Collect and return: (1) The average **total** reward received over the $M$ iterations; and (2) the average episode length. Analyze the performance of your agents with these new metrics.

# %%
#import numpy as np
#import gymnasium as gym
#import torch 
#import torch.nn as nn
#import torch.optim as optim
#from torch.distributions import Categorical
#import torch.nn.functional as F
#import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
#import os
#import datetime
#from reinforce_cartpole import PolicyNetwork, ReinforceAgent, TrainAgentRenforce
##import pygame
##_ = pygame.init()
#
#now= datetime.datetime.now()
#data_ora_formattata = now.strftime("%d_%m_%yT%H_%M")
#name= f'run_{data_ora_formattata}'
#
##env = gym.make("CartPole-v1", render_mode="human")
#env = gym.make("CartPole-v1")
##pygame.display.init() 
#name_agent="CartPole_REINFORCE"
#temperature_train=0.7
#general_path= f'Reinforcment_Learning/{name_agent}_{data_ora_formattata}_temp_{temperature_train}'
#
#checkpoint_path=general_path+"/checkpoint"
#bestmodel_path= general_path+"/best_model"
#hyperparamtres_path= general_path+"/hyperparametres"
#
#obs_dim = env.observation_space.shape[0]
#action_dim = env.action_space.n
#
#policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
#
#logdir= f'tensorboard/Reinforcment_Learning/{name_agent}/{name}_temp_{temperature_train}'
#
#agent = ReinforceAgent(
#    enviroment=env,
#    logdir=logdir, #da modificare
#    policy=policy,
#    name_agent=name_agent,
#    gamma=0.99,
#    max_lenght=500
#)
#
#trainer = TrainAgentRenforce(
#    reinforcagent=agent,
#    lr=1e-2,
#    num_episode=500,
#    num_episode_validation=10,
#    check_val=10,
#    checkpoint_path=checkpoint_path,
#    best_model_path=bestmodel_path,
#    hyperparams_path=hyperparamtres_path,
#    temperature_train=temperature_train
#)
#
##try:
#running_rewards = trainer.train_agent()
##finally:
##    env.close()
##    pygame.display.quit()
##    pygame.quit()
#
#
##pygame.display.quit()
#
##da fare quella ccosa della finestra pygame e modificarla

# %% [markdown]
# -----
# ## Exercise 2: `REINFORCE` with a Value Baseline (warm up)
# 
# In this exercise we will augment my implementation (or your own) of `REINFORCE` to subtract a baseline from the target in the update equation in order to stabilize (and hopefully speed-up) convergence. For now we will stick to the Cartpole environment.
# 
# 

# %% [markdown]
# **First Things First**: Recall from the slides on Deep Reinforcement Learning that we can **subtract** any function that doesn't depend on the current action from the q-value without changing the (maximum of our) objecttive function $J$:  
# 
# $$ \nabla J(\boldsymbol{\theta}) \propto \sum_{s} \mu(s) \sum_a \left( q_{\pi}(s, a) - b(s) \right) \nabla \pi(a \mid s, \boldsymbol{\theta}) $$
# 
# In `REINFORCE` this means we can subtract from our target $G_t$:
# 
# $$ \boldsymbol{\theta}_{t+1} \triangleq \boldsymbol{\theta}_t + \alpha (G_t - b(S_t)) \frac{\nabla \pi(A_t \mid s, \boldsymbol{\theta})}{\pi(A_t \mid s, \boldsymbol{\theta})} $$
# 
# Since we are only interested in the **maximum** of our objective, we can also **rescale** our target by any function that also doesn't depend on the action. A **simple baseline** which is even independent of the state -- that is, it is **constant** for each episode -- is to just **standardize rewards within the episode**. So, we **subtract** the average return and **divide** by the variance of returns:
# 
# $$ \boldsymbol{\theta}_{t+1} \triangleq \boldsymbol{\theta}_t + \alpha \left(\frac{G_t - \bar{G}}{\sigma_G}\right) \nabla  \pi(A_t \mid s, \boldsymbol{\theta}) $$
# 
# This baseline is **already** implemented in my implementation of `REINFORCE`. Experiment with and without this standardization baseline and compare the performance. We are going to do something more interesting.

# %%
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
import datetime
from reinforce_cartpole import PolicyNetwork, ReinforceAgent, TrainAgentRenforce
#import pygame
#_ = pygame.init()

now= datetime.datetime.now()
data_ora_formattata = now.strftime("%d_%m_%yT%H_%M")
name= f'run_{data_ora_formattata}'

#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
#pygame.display.init() 
name_agent="CartPole_REINFORCE"
normalizzation_discount=False
baseline_discount=False
temperature_train=[0.3, 0.5, 0.7, 1 , 1.5, 2]
for temp in temperature_train:
    for subtract in [True,False]:
        baseline= "simple_subtract" if subtract else "normalization"


        general_path= f'Reinforcment_Learning_Classic_Baseline_sub_std/{name_agent}_{data_ora_formattata}_temp_{temp}_Baseline_{baseline}'

        checkpoint_path=general_path+"/checkpoint"
        bestmodel_path= general_path+"/best_model"
        hyperparamtres_path= general_path+"/hyperparametres"

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)

        logdir= f'tensorboard/Reinforcment_Learning_Baseline_sub_std/{name_agent}/{name}_temp_{temp}_BaseLine_{baseline}'

        agent = ReinforceAgent(
            enviroment=env,
            logdir=logdir, #da modificare
            policy=policy,
            name_agent=name_agent,
            gamma=0.99,
            max_lenght=500
        )

        trainer = TrainAgentRenforce(
            reinforcagent=agent,
            lr=1e-2,
            num_episode=500,
            num_episode_validation=10,
            check_val=10,
            checkpoint_path=checkpoint_path,
            best_model_path=bestmodel_path,
            hyperparams_path=hyperparamtres_path,
            temperature_train=temp
        )
        if subtract:
            running_rewards = trainer.train_agent(normalizzation_discount=False,baseline_discount_sub=True)
        else:
            running_rewards=trainer.train_agent(normalizzation_discount=True,baseline_discount_sub=False)

# %% [markdown]
# **The Real Exercise**: Standard practice is to use the state-value function $v(s)$ as a baseline. This is intuitively appealing -- we are more interested in updating out policy for returns that estimate the current **value** worse. Our new update becomes:
# 
# $$ \boldsymbol{\theta}_{t+1} \triangleq \boldsymbol{\theta}_t + \alpha (G_t - \tilde{v}(S_t \mid \mathbf{w})) \frac{\nabla \pi(A_t \mid s, \boldsymbol{\theta})}{\pi(A_t \mid s, \boldsymbol{\theta})} $$
# 
# where $\tilde{v}(s \mid \mathbf{w})$ is a **deep neural network** with parameters $w$ that estimates $v_\pi(s)$. What neural network? Typically, we use the **same** network architecture as that of the Policy.
# 
# **Your Task**: Modify your implementation to fit a second, baseline network to estimate the value function and use it as **baseline**. 

# %%
# Your code here.

# %% [markdown]
# -----
# ## Exercise 3: Going Deeperq
# 
# As usual, pick **AT LEAST ONE** of the following exercises to complete.
# 
# ### Exercise 3.1: Solving Lunar Lander with `REINFORCE` (easy)
# 
# Use my (or even better, improve on my) implementation of `REINFORCE` to solve the [Lunar Lander Environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/). This environment is a little bit harder than Cartpole, but not much. Make sure you perform the same types of analyses we did during the lab session to quantify and qualify the performance of your agents.
# 
# ### Exercise 3.2: Solving Cartpole and Lunar Lander with `Deep Q-Learning` (harder)
# 
# On policy Deep Reinforcement Learning tends to be **very unstable**. Write an implementation (or adapt an existing one) of `Deep Q-Learning` to solve our two environments (Cartpole and Lunar Lander). To do this you will need to implement a **Replay Buffer** and use a second, slow-moving **target Q-Network** to stabilize learning.
# 
# ### Exercise 3.3: Solving the OpenAI CarRacing environment (hardest) 
# 
# Use `Deep Q-Learning` -- or even better, an off-the-shelf implementation of **Proximal Policy Optimization (PPO)** -- to train an agent to solve the [OpenAI CarRacing](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN) environment. This will be the most *fun*, but also the most *difficult*. Some tips:
# 
# 1. Make sure you use the `continuous=False` argument to the environment constructor. This ensures that the action space is **discrete** (we haven't seen how to work with continuous action spaces).
# 2. Your Q-Network will need to be a CNN. A simple one should do, with two convolutional + maxpool layers, folowed by a two dense layers. You will **definitely** want to use a GPU to train your agents.
# 3. The observation space of the environment is a single **color image** (a single frame of the game). Most implementations stack multiple frames (e.g. 3) after converting them to grayscale images as an observation.
# 
#  



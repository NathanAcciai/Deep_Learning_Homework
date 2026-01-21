import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import config
import os
import pygame
from multiprocessing import Process, Queue

from stats_test_windows import StatsTestWindow



class QNetwork(torch.nn.Module):
    def __init__(self, obs_dim, n_action, hidden_size=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_action),
        )
    
    def forward(self, x):
        return self.net(x)

def load_model(path_model,obs_dim, n_action,name_key):
    checkpoint = torch.load(path_model, map_location="cpu", weights_only=False)
    Config= config.Config
    q_net= QNetwork(obs_dim, n_action, Config[name_key]["hidden_size"])
    
    q_net.load_state_dict(checkpoint["QNetwork"])
    return q_net

from torch.utils.tensorboard import SummaryWriter

def test_model(
    path_model,
    env_test,
    num_episode,
    logdir,
    render=False,
    name_key="cartpole"
):
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    if render:
        _ = pygame.init()
        q = Queue()
        window = StatsTestWindow(q, env_name=env_test)
        p = Process(target=window.run)
        p.start()

    print(f'\nCaricamento del modello per test su {env_test}')
    if render:
        env = gym.make(env_test, render_mode="human")
    else:
        env = gym.make(env_test)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = load_model(path_model, obs_dim, n_actions, name_key)
    q_net.eval()

    print("\nModello caricato correttamente\n")
    print("Avvio test")

    rewards = []
    steps_list = []

    for ep in tqdm(range(num_episode), desc="Testing"):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            obs_tensor = torch.tensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_net(obs_tensor)
                action = torch.argmax(q_vals, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            q.put({
                "steps": steps,
            })


        rewards.append(total_reward)
        steps_list.append(steps)

        # ---- TensorBoard logging per episodio ----
        writer.add_scalar("test/episode_reward", total_reward, ep)
        writer.add_scalar("test/episode_steps", steps, ep)

        writer.add_scalar("test/mean_reward", np.mean(rewards), ep)
        writer.add_scalar("test/std_reward", np.std(rewards), ep)
        writer.add_scalar("test/best_reward", np.max(rewards), ep)
        

        if render:
            q.put({
                "episode": ep + 1,
                "reward": total_reward,
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "best_reward": np.max(rewards),
                "completed": len(rewards)
            })

        tqdm.write(f"Episodio {ep+1:02d} → Reward {total_reward}")

    env.close()
    writer.close()

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)

    print("\nRISULTATI FINALI")
    print(f"   Reward medio: {mean_r:.2f} ± {std_r:.2f}")
    print(f"   Reward max : {np.max(rewards)}")
    print(f"   Reward min : {np.min(rewards)}")
    print(f"   Steps medi : {np.mean(steps_list):.1f}")

    

    if render:
        p.terminate()
        p.join()

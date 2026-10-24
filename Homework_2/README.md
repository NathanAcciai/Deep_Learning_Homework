# Homework 2 — Deep Reinforcement Learning

Policy-gradient and value-based methods on Gymnasium environments: **REINFORCE** with different variance-reduction baselines on CartPole, and **Deep Q-Learning** on CartPole and LunarLander.

**Report:** [Weights & Biases](https://wandb.ai/nathan-acciai-universit-di-firenze/Homework_2/reports/Homework-2-Report--VmlldzoxNTY4OTY2OA?accessToken=d1wtwji86uvduybwxa0vn4ylwq9nblligngtymhg3g8tqjxep2jauvqahah10k3s)

## Contents

```
Homework_2/
├── DLA-Lab2-DRL.ipynb        # main notebook
├── reinforce_cartpole.py     # PolicyNetwork, ValueNetwork, ReinforceAgent, TrainAgentRenforce
├── Q_Learining.py            # ReplayBuffer, QNetwork, DQNAgent
├── config.py                 # DQN hyperparameters (CartPole / LunarLander) and render flag
├── Test_model.py             # evaluates a saved DQN model, optional rendering
├── stats_windows.py          # live pygame statistics window during training
├── stats_test_windows.py     # live pygame statistics window during testing
├── Script_H2.py              # script export of the notebook
├── Reinforcment_Learning/                          # REINFORCE, temperature sweep (no baseline)
├── Reinforcment_Learning_Classic_Baseline_sub_std/ # return standardization vs mean subtraction
├── Reinforcment_Learning_Value_Net/                # learned value-function baseline (18 runs)
├── Reinforcement_Learning_CartLunar/               # DQN runs (cartpole/, lunars/)
└── tensorboard/                                    # curves of every run
```

Each REINFORCE run folder contains `best_model/`, `checkpoint/` and `hyperparametres/hyperparametres.csv`; each DQN run contains `best_model.pth`, `checkpoint.pth` and `hyperparametres.csv`. All weights are versioned, so trained agents can be replayed without retraining.

## What was implemented

**REINFORCE on CartPole-v1** (`reinforce_cartpole.py`)
- Policy network with a softmax **temperature** on the logits (training and evaluation).
- Periodic evaluation: every 10 episodes the agent plays 10 evaluation episodes; logged metrics are mean reward, episode length, standard deviation, min/max, 10th percentile and success/failure terminations. A pygame window plots them live.
- Return handling selectable at training time: none (plain REINFORCE), **standardization** of the returns `(G − mean)/std`, **mean subtraction** `G − mean`, or a **learned value network** `v(s)` used as baseline (trained with its own optimizer, learning rate `lr_value_net`).
- Checkpoint and best-model saving, resumable training.

**Deep Q-Learning** (`Q_Learining.py`)
- GPU-resident replay buffer, MLP Q-network with two hidden layers, ε-greedy exploration with linear decay (1.0 → 0.05), target network copied periodically (hard update), Huber (SmoothL1) loss with gradient clipping, γ = 0.99, lr 1e-4.
- Evaluation episodes during training; best model kept on the validation reward.

| | CartPole-v1 | LunarLander-v3 |
|---|---|---|
| Hidden size / batch size | 128 / 64 | 512 / 128 |
| Replay buffer | 100k | 1M |
| ε decay steps / target update every | 10k / 700 | 50k / 1700 |
| Training episodes | 500 | 1000 |

## Results

### REINFORCE — final evaluation reward (max = 500)

Mean reward of the last evaluation (10 episodes) of each 500-episode run, for each softmax temperature:

| Temperature | No baseline | Standardized returns | Mean subtraction |
|---|---|---|---|
| 0.3 | 9.8 | **500.0** | 460.7 |
| 0.5 | 86.8 | **500.0** | **500.0** |
| 0.7 | 123.1 | **500.0** | **500.0** |
| 1 | 9.4 | 173.2 | **500.0** |
| 1.5 | — | 491.3 | 417.5 |
| 2 | 15.1 | 162.0 | 115.0 |

Learned **value-network baseline**, varying the value-net learning rate:

| Temperature | lr_v = 1e-2 | lr_v = 1e-3 | lr_v = 1e-4 |
|---|---|---|---|
| 0.3 | **500.0** | 8.8 | 9.5 |
| 0.5 | **500.0** | **500.0** | **500.0** |
| 0.7 | 464.2 | 499.3 | 163.6 |
| 1 | 496.9 | 384.6 | 125.6 |
| 1.5 | 253.2 | 192.0 | 357.1 |
| 2 | **500.0** | 419.9 | 359.8 |

Without a baseline REINFORCE is very unstable; any baseline helps, with temperatures around 0.5–0.7 the most robust. With the value network, lr 1e-2 is the most reliable (4/6 runs end at ≥ 495, against 2/6 for 1e-3 and 1/6 for 1e-4).

### DQN — test of the best checkpoint

| Environment | Episodes | Result |
|---|---|---|
| CartPole-v1 | 50 | reward **500** in every episode (std 0) |
| LunarLander-v3 | 100 | mean reward **≈ 260** (best episode ≈ 306, solved threshold 200) |

Models used for testing: `Reinforcement_Learning_CartLunar/cartpole/run_04_12_25T15_48` and `lunars/run_06_12_25T16_11`. `lunars/run_05_12_25T16_18` is an earlier run of the same setup (final validation 230); `lunars/run_22_01_26T12_11` was interrupted after a few episodes.

## How to run

```bash
pip install torch gymnasium "gymnasium[box2d]" pygame tensorboard pandas numpy matplotlib tqdm jupyter
jupyter notebook DLA-Lab2-DRL.ipynb
tensorboard --logdir tensorboard
```

Replay a trained DQN agent (set `"human": True` in `config.py` or `render=True` to watch it):

```python
import Test_model
Test_model.test_model(
    path_model="Reinforcement_Learning_CartLunar/lunars/run_06_12_25T16_11/best_model.pth",
    env_test="LunarLander-v3",
    num_episode=50,
    logdir="tensorboard/Reinforcement_Learning_CartLunar/lunars/test",
    render=True,
)
```

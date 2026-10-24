# Deep Learning Applications — Homework & Paper Presentation

Coursework repository for the **Deep Learning Applications** course (University of Florence, Artificial Intelligence track).
It contains the four lab assignments of the course (CNNs, Deep Reinforcement Learning, Transformers, OOD detection & adversarial learning) and the presentation of a conference paper in Reinforcement Learning.

**Author:** Nathan Acciai

---

## At a glance

| # | Topic | Main experiments | Headline result | Report |
|---|---|---|---|---|
| [1](./Homework_1) | MLP & CNN, residual connections, transfer learning | Plain vs residual MLP (MNIST) and CNN (CIFAR-10) at depth 2/6/10; CIFAR-10 → CIFAR-100 feature extraction and fine-tuning | A 10-layer plain MLP does not train (11.4%), the residual one reaches 96.9% | [W&B](https://api.wandb.ai/links/nathan-acciai-universit-di-firenze/a3xnw969) |
| [2](./Homework_2) | Deep Reinforcement Learning | REINFORCE with baselines on CartPole; DQN on CartPole and LunarLander | DQN reaches the maximum score (500) on CartPole and ≈260 reward on LunarLander (100 test episodes) | [W&B](https://wandb.ai/nathan-acciai-universit-di-firenze/Homework_2/reports/Homework-2-Report--VmlldzoxNTY4OTY2OA?accessToken=d1wtwji86uvduybwxa0vn4ylwq9nblligngtymhg3g8tqjxep2jauvqahah10k3s) |
| [3](./Homework_3) | Transformers & Hugging Face | DistilBERT sentiment analysis (frozen features vs full fine-tuning); CLIP + LoRA on ImageNette, Tiny ImageNet and ImageNet-Sketch | LoRA lifts CLIP on ImageNet-Sketch from 43.8% to 73.6% | [W&B](https://wandb.ai/nathan-acciai-universit-di-firenze/Homework_3/reports/Homework-3-Report--VmlldzoxNTcxMjExMQ?accessToken=zeu8m2dyd65fu6oyuuvl8kabn15ej3hfrlxa2usaqfovqbl4z76vpoa4xaau949c) |
| [4](./Homework_4) | OOD detection & adversarial attacks | OOD scores (max-logit, max-softmax, autoencoder error) with CIFAR-10 as in-distribution; targeted FGSM | — | notebook only |
| — | Paper presentation | *1000 Layer Networks for Self-Supervised RL* | — | [slides](./Presentazione_Acciai_Nathan.pdf) · [video](./Presentazione_Acciai_Nathan.mp4.zip) |

## Repository structure

```
Deep_Learning_Homework/
├── Homework_1/                  # CNNs, residual connections, fine-tuning
│   ├── Lab1-CNNs.ipynb          #   main notebook
│   ├── Simple_MLP/              #   baseline MLP runs (hyperparameters)
│   ├── Residual_vs_Simple_MLP/  #   plain vs residual MLP + gradient analysis conclusions
│   ├── CNN_Residual_vs_Base/    #   plain vs residual CNN, depth 2/6/10
│   ├── Reusing_Model/           #   CIFAR-10 → CIFAR-100: feature extraction + fine-tuning grid
│   └── tensorboard/             #   training curves of every run above
├── Homework_2/                  # Deep RL
│   ├── DLA-Lab2-DRL.ipynb       #   main notebook
│   ├── reinforce_cartpole.py    #   REINFORCE agent (policy net, value-net baseline, trainer)
│   ├── Q_Learining.py           #   DQN agent (replay buffer, target network)
│   ├── config.py                #   DQN hyperparameters for CartPole / LunarLander
│   ├── Test_model.py            #   evaluates a saved DQN model (optional rendering)
│   ├── stats_windows.py, stats_test_windows.py   # live pygame statistics windows
│   ├── Reinforcment_Learning/                    # REINFORCE, temperature sweep
│   ├── Reinforcment_Learning_Classic_Baseline_sub_std/  # return normalization vs mean subtraction
│   ├── Reinforcment_Learning_Value_Net/          # learned value-function baseline
│   ├── Reinforcement_Learning_CartLunar/         # DQN runs (CartPole, LunarLander)
│   └── tensorboard/
├── Homework_3/                  # Transformers
│   ├── Lab3-Transformers.ipynb  #   main notebook (Script_H3.py is the script export)
│   ├── Exercise_1/              #   DistilBERT features + SVM reports
│   ├── Exercise_2/              #   DistilBERT fine-tuning reports
│   ├── Exercise_3/              #   CLIP zero-shot + LoRA reports (ImageNette, TinyImagenet, SketchImageNet)
│   └── classnames.txt           #   the 1000 ImageNet class names used to build CLIP prompts
├── Homework_4/
│   └── Lab4_OOD_2025.ipynb      # OOD detection and FGSM
├── Presentazione_Acciai_Nathan.pdf       # slides of the paper presentation
├── Presentazione_Acciai_Nathan.mp4.zip   # recording of the presentation (unzip first)
└── README.md
```

Conventions: every run lives in a timestamped folder (`Run_DD_MM_YYTHH_MM`, `run_YYYY-MM-DD_HH-MM-SS`) with a `hyperparametres.csv` (or JSON reports) and, when available, the TensorBoard logs.

---

## Homework 1 — MLPs, CNNs and residual connections

Small-scale replication of the ResNet paper (He et al., CVPR 2016), followed by transfer learning. Training pipeline: custom `Trainer` with TensorBoard logging, 100 epochs, batch size 128, lr 1e-3, weight decay 1e-3, seed 123.

**1.1 Baseline MLP (MNIST).** Two layers of width 16: **95.9%** test accuracy.

**1.2 Residual MLP vs plain MLP (MNIST, width 16).** Residual blocks (two linear layers + skip connection, projection when dimensions differ). Test accuracy (%):

| Depth | Plain MLP | Residual MLP |
|---|---|---|
| 2 | 95.89 | 96.63 |
| 6 | 95.32 | 96.70 |
| 10 | 11.35 | **96.86** |

Gradient-norm analysis on a training batch ([conclusions](./Homework_1/Residual_vs_Simple_MLP/Conclusion_of_gradient_anlysis)): in the plain MLP the gradient shrinks rapidly towards the input layers (vanishing gradient), so the deep network cannot train; with skip connections the gradient norms stay balanced across layers.

**1.3 Residual CNN vs plain CNN (CIFAR-10).** 64-channel 3×3 convolution blocks with batch norm; residual variant built on torchvision's `BasicBlock`. Test accuracy (%):

| Depth | Plain CNN | Residual CNN |
|---|---|---|
| 2 | 65.89 | **77.67** |
| 6 | 74.92 | 76.65 |
| 10 | **75.81** | 75.65 |

The residual advantage is large for the shallowest network; at depth 10 the two variants are on par.

**2.1 Reusing the model: CIFAR-10 → CIFAR-100.** The residual CNN of depth 6 trained on CIFAR-10 is reused in two ways.

*Feature extractor + classical classifier* (global-average-pool features, test accuracy):

| Classifier | Accuracy (%) |
|---|---|
| Linear SVM | **26.73** |
| Gaussian classifier | 14.46 |
| k-NN | 11.23 |

*Fine-tuning* (30 epochs, batch size 256, new 100-class head). The head and the first *k* residual blocks are unfrozen (k = 1, 2, 3) and three optimizers are compared with a "high" and a "low" learning-rate setting. Test accuracy (%):

| Optimizer (lr) | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| RMSprop (1e-4) | 30.12 | 33.43 | **40.31** |
| AdamW (1e-3) | 29.99 | 36.63 | 39.52 |
| AdamW (1e-4) | 22.42 | 27.54 | 30.60 |
| RMSprop (1e-5) | 22.23 | 26.65 | 30.21 |
| SGD + momentum (1e-3) | 17.98 | 21.66 | 22.89 |
| SGD + momentum (1e-4) | 4.96 | 5.62 | 5.72 |

Unfreezing more blocks and using higher learning rates helps, but accuracy stays at 30–40%. [Conclusion](./Homework_1/Reusing_Model/Conlusion_of_Fine_Tuning_and_FE): the network is small (64 channels everywhere) and the features learned on 10 CIFAR-10 classes are not general enough for 100 fine-grained classes; small gradients in the early layers further limit adaptation.

> Model weights are not versioned (`*.pt` is git-ignored): to run the transfer-learning cells, first re-train the CNNs of Exercise 1.3.

---

## Homework 2 — Deep Reinforcement Learning

Environments from Gymnasium. Code is organised in modules (`reinforce_cartpole.py`, `Q_Learining.py`, `Test_model.py`) driven from the notebook.

**REINFORCE on CartPole-v1.** Policy network with a softmax temperature; 500 training episodes, γ = 0.99, lr 1e-2. Every 10 episodes the agent is evaluated on 10 episodes, logging mean reward, episode length, std, min/max, 10th percentile and success/failure terminations. A pygame window shows live statistics.

**Baselines and temperature study.** The same agent is compared across softmax temperatures {0.3, 0.5, 0.7, 1, 1.5, 2} with different variance-reduction strategies. A run counts as converged if its final evaluation reward is ≥ 495 (maximum is 500):

| Variant | Runs | Converged |
|---|---|---|
| Plain REINFORCE (temperatures 0.3 – 2, no baseline) | 5 | 0 |
| Returns standardized (mean/std) | 6 | 3 |
| Returns minus mean | 6 | 3 |
| Learned value network as baseline (lr of value net ∈ {1e-2, 1e-3, 1e-4}) | 18 | 7 |

Findings: temperatures around 0.5–0.7 are the most robust and high temperatures (≥ 1.5) are less reliable; with the value-network baseline a value-net learning rate of 1e-2 is the most reliable (4/6 runs converge, against 2/6 for 1e-3 and 1/6 for 1e-4).

**Deep Q-Learning (CartPole-v1 and LunarLander-v3).** Replay buffer, ε-greedy exploration with linear decay (1.0 → 0.05), periodic hard update of a target network, Huber (SmoothL1) loss with gradient clipping, γ = 0.99, lr 1e-4 (settings in [`config.py`](./Homework_2/config.py)):

| | CartPole-v1 | LunarLander-v3 |
|---|---|---|
| Hidden size / batch size | 128 / 64 | 512 / 128 |
| Replay buffer | 100k | 1M |
| ε decay steps / target update | 10k / 700 | 50k / 1700 |
| Training episodes | 500 | 1000 |
| **Test reward** (best checkpoint) | **500** on all 50 episodes | **≈260** mean over 100 episodes (best episode 306) |

Saved models and the best checkpoints of every run are included, so `Test_model.py` can replay the trained agents without retraining.

---

## Homework 3 — Transformers and the Hugging Face ecosystem

**Sentiment analysis on Rotten Tomatoes (DistilBERT).**

| Approach | Validation acc. (%) | Test acc. (%) |
|---|---|---|
| Frozen DistilBERT (`[CLS]` of the last layer) + Linear SVM — *stable baseline* | 82.18 | 79.83 |
| Full fine-tuning with the Hugging Face `Trainer` (3 epochs) | 84.24 | **84.43** |

**CLIP (`openai/clip-vit-base-patch16`) with LoRA.** Zero-shot classification with the prompt "This is a photo of {class}", then parameter-efficient fine-tuning with LoRA (rank 4, α = 14, dropout 0.1, on `q_proj`/`v_proj`; AdamW, lr 1e-5, batch size 4). Three adaptation settings are compared: text encoder only, vision encoder only, or both. Validation accuracy (%) after the last epoch:

| Dataset | Zero-shot | LoRA text enc. | LoRA vision enc. | LoRA both |
|---|---|---|---|---|
| ImageNette | 97.74 | 97.74 | **99.62** | **99.62** |
| Tiny ImageNet | 61.45 | 80.17 | **81.23** | 81.11 |
| ImageNet-Sketch | 43.75 | 67.24 | **73.57** | 70.16 |

CLIP is already near-saturated on ImageNette, whereas the domain shift of sketches leaves a large margin: adapting the vision encoder gives the biggest gain (+29.8 points). On ImageNette, text-only LoRA peaks at 98.87% after the second epoch and then drifts back to the zero-shot level.

---

## Homework 4 — OOD detection and adversarial learning

`Lab4_OOD_2025.ipynb` (seed 42, deterministic PyTorch):

- **OOD detection pipeline.** CIFAR-10 is the in-distribution set (a 5-convolution CNN classifier) and `FakeData` is the out-of-distribution set. Three OOD scores are compared with histograms: **max-logit**, **max-softmax** (with temperature) and the **reconstruction error of a convolutional autoencoder** trained on CIFAR-10.
- **Performance metrics.** ROC curves (`sklearn.metrics.RocCurveDisplay`) for the autoencoder score and for the logit/softmax scores.
- **FGSM adversarial examples.** Fast Gradient Sign Method, including the *targeted* variant (e.g. steering a sample towards the class "deer" with ε = 1/255), with the perturbation visualised after undoing the input normalisation.

---

## Paper presentation

> **1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities**
> Wang, K., Javali, I., Bortkiewicz, M., & Eysenbach, B. (2025). *arXiv preprint arXiv:2503.14858*.

The presentation ([slides](./Presentazione_Acciai_Nathan.pdf), [video](./Presentazione_Acciai_Nathan.mp4.zip)) covers:

1. **Motivation** — classic RL suffers from sparse rewards, heavy dependence on exploration and noisy value estimates, which makes training unstable as model capacity grows.
2. **Goal-conditioned RL** — the problem is reformulated as goal reaching; the reward measures the probability of reaching the goal.
3. **Contrastive RL** — an actor–critic method where the critic measures the distance between state-action and goal embeddings and is trained with the InfoNCE loss.
4. **Residual connections** — improve gradient propagation and preserve information across layers, making networks of hundreds to ~1000 layers trainable.
5. **Results and limitations** — large gains in performance and stability, generalisation to distant unseen goals and emergent behaviours; limitations are the need for large data and batches, a learned distance that is only a quasi-metric, and the computational cost of very deep networks.

---

## Running the code

Python 3 with a CUDA GPU recommended. Libraries used:

| Homework | Libraries |
|---|---|
| 1 | `torch`, `torchvision`, `scikit-learn`, `tensorboard`, `pandas`, `numpy`, `matplotlib`, `tqdm` |
| 2 | `torch`, `gymnasium` (+ `box2d` for LunarLander), `pygame`, `tensorboard`, `pandas`, `numpy`, `matplotlib`, `tqdm` |
| 3 | `torch`, `torchvision`, `transformers`, `datasets`, `peft`, `accelerate`, `scikit-learn`, `tensorboard` |
| 4 | `torch`, `torchvision`, `scikit-learn`, `numpy`, `matplotlib` |

```bash
git clone https://github.com/NathanAcciai/Deep_Learning_Homework.git
cd Deep_Learning_Homework
jupyter notebook Homework_1/Lab1-CNNs.ipynb   # or the notebook of the homework you want
```

Datasets (MNIST, CIFAR-10/100, Rotten Tomatoes, ImageNette, Tiny ImageNet, ImageNet-Sketch) are downloaded automatically by `torchvision` / Hugging Face `datasets`. To inspect the training curves:

```bash
tensorboard --logdir Homework_1/tensorboard   # likewise Homework_2/tensorboard
```

---

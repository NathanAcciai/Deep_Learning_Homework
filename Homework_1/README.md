# Homework 1 — MLPs, CNNs and Residual Connections

Small-scale replication of the main message of the ResNet paper (He et al., CVPR 2016): *deeper networks do not guarantee better training, residual connections fix that*. The second part reuses a residual CNN trained on CIFAR-10 for CIFAR-100 (feature extraction and fine-tuning).

**Report:** [Weights & Biases](https://api.wandb.ai/links/nathan-acciai-universit-di-firenze/a3xnw969)

## Contents

```
Homework_1/
├── Lab1-CNNs.ipynb              # main notebook (all exercises)
├── Simple_MLP/                  # Ex. 1.1 — baseline MLP: hyperparameters of each attempt
├── Residual_vs_Simple_MLP/      # Ex. 1.2 — plain vs residual MLP (depth 2/6/10)
│   └── Conclusion_of_gradient_anlysis   # written analysis of gradient norms
├── CNN_Residual_vs_Base/        # Ex. 1.3 — plain vs residual CNN (depth 2/6/10)
├── Reusing_Model/               # Ex. 2.1 — CIFAR-10 → CIFAR-100
│   ├── Fine_Tuning_run_20_01_26T20_00/   # fine-tuning grid (block × optimizer × lr)
│   └── Conlusion_of_Fine_Tuning_and_FE   # written conclusions
└── tensorboard/                 # loss/accuracy curves of every run above
```

Each run folder is named `Run_DD_MM_YYTHH_MM` and holds a `hyperparametres.csv` with epochs, batch size, learning rate, weight decay, depth and (for fine-tuning) optimizer and unfrozen blocks.

## What was implemented

| Component | Description |
|---|---|
| `My_MLP`, `SkipBlock` | Parameterised MLP from a list of layer widths; with `use_skip=True` each layer becomes a residual block (two linear layers + skip, linear projection when the dimensions differ). |
| `CNN_Customize` | CNN with a convolutional head, `depth` blocks of 64 channels and global average pooling. Blocks are either plain (conv-BN-ReLU) or residual (`BasicBlock` from torchvision). |
| `Trainer` | Training/validation pipeline shared by all experiments: TensorBoard logging of losses and accuracy, gradient-norm monitoring (`study_grad`), best-model saving, hyperparameter CSV. |
| `Feature_Extractor`, `custom_classifier` | Removes the last layer of the CNN and fits a scikit-learn classifier (SVM, k-NN, Gaussian) on the pooled features. |
| `fine_tuning` | Replaces the classifier with a 100-class head and unfreezes a chosen list of blocks. |

Setup: seed 123, 100 epochs, batch size 128, lr 1e-3, weight decay 1e-3 for Ex. 1.1–1.3; 30 epochs and batch size 256 for the fine-tuning grid.

## Results

### Ex. 1.1–1.2 — MLP on MNIST (width 16)

A two-layer baseline MLP reaches **95.92%** test accuracy. Plain vs residual MLP (test accuracy, %):

| Depth | Plain MLP | Residual MLP |
|---|---|---|
| 2 | 95.89 | 96.63 |
| 6 | 95.32 | 96.70 |
| 10 | 11.35 | **96.86** |

The gradient-norm analysis explains the gap: in the plain network the gradients vanish towards the input layers and the deep model cannot train; with skip connections the norms stay balanced across all layers.

### Ex. 1.3 — CNN on CIFAR-10 (64 channels)

| Depth | Plain CNN | Residual CNN |
|---|---|---|
| 2 | 65.89 | **77.67** |
| 6 | 74.92 | 76.65 |
| 10 | **75.81** | 75.65 |

The residual variant is clearly better at small depth; at depth 10 the two are on par.

### Ex. 2.1 — CIFAR-10 → CIFAR-100 with the residual CNN of depth 6

*Frozen features + classical classifier (test accuracy):* Linear SVM **26.73%**, Gaussian classifier 14.46%, k-NN 11.23%.

*Fine-tuning* — new 100-class head plus the first *k* residual blocks unfrozen (test accuracy, %):

| Optimizer (lr) | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| RMSprop (1e-4) | 30.12 | 33.43 | **40.31** |
| AdamW (1e-3) | 29.99 | 36.63 | 39.52 |
| AdamW (1e-4) | 22.42 | 27.54 | 30.60 |
| RMSprop (1e-5) | 22.23 | 26.65 | 30.21 |
| SGD + momentum (1e-3) | 17.98 | 21.66 | 22.89 |
| SGD + momentum (1e-4) | 4.96 | 5.62 | 5.72 |

Accuracy stays at 30–40%: the network is small (64 channels everywhere) and the features learned on 10 classes do not transfer well to 100 fine-grained classes; moreover the gradients reaching the early layers are tiny, so the base features barely change.

## How to run

```bash
pip install torch torchvision scikit-learn tensorboard pandas numpy matplotlib tqdm jupyter
jupyter notebook Lab1-CNNs.ipynb
tensorboard --logdir tensorboard      # inspect all the runs
```

MNIST, CIFAR-10 and CIFAR-100 are downloaded automatically into `./data`. Model weights (`*.pt`) are **not** versioned (see `.gitignore`): to run the transfer-learning cells, first re-train the CNNs of Ex. 1.3 so that `best_model.pt` exists.

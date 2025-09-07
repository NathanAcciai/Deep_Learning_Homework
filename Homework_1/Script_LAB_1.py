# %% [markdown]
# # Deep Learning Applications: Laboratory #1
# 
# In this first laboratory we will work relatively simple architectures to get a feel for working with Deep Models. This notebook is designed to work with PyTorch, but as I said in the introductory lecture: please feel free to use and experiment with whatever tools you like.
# 
# **Important Notes**:
# 1. Be sure to **document** all of your decisions, as well as your intermediate and final results. Make sure your conclusions and analyses are clearly presented. Don't make us dig into your code or walls of printed results to try to draw conclusions from your code.
# 2. If you use code from someone else (e.g. Github, Stack Overflow, ChatGPT, etc) you **must be transparent about it**. Document your sources and explain how you adapted any partial solutions to creat **your** solution.
# 
# 

# %% [markdown]
# ## Exercise 1: Warming Up
# In this series of exercises I want you to try to duplicate (on a small scale) the results of the ResNet paper:
# 
# > [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016.
# 
# We will do this in steps using a Multilayer Perceptron on MNIST.
# 
# Recall that the main message of the ResNet paper is that **deeper** networks do not **guarantee** more reduction in training loss (or in validation accuracy). Below you will incrementally build a sequence of experiments to verify this for an MLP. A few guidelines:
# 
# + I have provided some **starter** code at the beginning. **NONE** of this code should survive in your solutions. Not only is it **very** badly written, it is also written in my functional style that also obfuscates what it's doing (in part to **discourage** your reuse!). It's just to get you *started*.
# + These exercises ask you to compare **multiple** training runs, so it is **really** important that you factor this into your **pipeline**. Using [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) is a **very** good idea -- or, even better [Weights and Biases](https://wandb.ai/site).
# + You may work and submit your solutions in **groups of at most two**. Share your ideas with everyone, but the solutions you submit *must be your own*.
# 
# First some boilerplate to get you started, then on to the actual exercises!

# %% [markdown]
# ### Preface: Some code to get you started
# 
# What follows is some **very simple** code for training an MLP on MNIST. The point of this code is to get you up and running (and to verify that your Python environment has all needed dependencies).
# 
# **Note**: As you read through my code and execute it, this would be a good time to think about *abstracting* **your** model definition, and training and evaluation pipelines in order to make it easier to compare performance of different models.

# %%
# Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# %% [markdown]
# #### Data preparation
# 
# Here is some basic dataset loading, validation splitting code to get you started working with MNIST.

# %%
# Standard MNIST transform.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST train and test.
ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
ds_test = MNIST(root='./data', train=False, download=True, transform=transform)

# Split train into train and validation.
val_size = 5000
I = np.random.permutation(len(ds_train))
ds_val = Subset(ds_train, I[:val_size])
ds_train = Subset(ds_train, I[val_size:])

# %% [markdown]
# #### Boilerplate training and evaluation code
# 
# This is some **very** rough training, evaluation, and plotting code. Again, just to get you started. I will be *very* disappointed if any of this code makes it into your final submission.

# %%
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Function to train a model for a single epoch over the data loader.
def train_epoch(model, dl, opt, epoch='Unknown', device='cpu'):
    model.train()
    losses = []
    for (xs, ys) in tqdm(dl, desc=f'Training epoch {epoch}', leave=True):
        xs = xs.to(device)
        ys = ys.to(device)
        opt.zero_grad()
        logits = model(xs)
        loss = F.cross_entropy(logits, ys)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)

# Function to evaluate model over all samples in the data loader.
def evaluate_model(model, dl, device='cpu'):
    model.eval()
    predictions = []
    gts = []
    for (xs, ys) in tqdm(dl, desc='Evaluating', leave=False):
        xs = xs.to(device)
        preds = torch.argmax(model(xs), dim=1)
        gts.append(ys)
        predictions.append(preds.detach().cpu().numpy())
        
    # Return accuracy score and classification report.
    return (accuracy_score(np.hstack(gts), np.hstack(predictions)),
            classification_report(np.hstack(gts), np.hstack(predictions), zero_division=0, digits=3))

# Simple function to plot the loss curve and validation accuracy.
def plot_validation_curves(losses_and_accs):
    losses = [x for (x, _) in losses_and_accs]
    accs = [x for (_, x) in losses_and_accs]
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss per Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(accs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Best Accuracy = {np.max(accs)} @ epoch {np.argmax(accs)}')

# %% [markdown]
# #### A basic, parameterized MLP
# 
# This is a very basic implementation of a Multilayer Perceptron. Don't waste too much time trying to figure out how it works -- the important detail is that it allows you to pass in a list of input, hidden layer, and output *widths*. **Your** implementation should also support this for the exercises to come.

# %%
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(nin, nout) for (nin, nout) in zip(layer_sizes[:-1], layer_sizes[1:])])
    
    def forward(self, x):
        return reduce(lambda f, g: lambda x: g(F.relu(f(x))), self.layers, lambda x: x.flatten(1))(x)

# %% [markdown]
# #### A *very* minimal training pipeline.
# 
# Here is some basic training and evaluation code to get you started.
# 
# **Important**: I cannot stress enough that this is a **terrible** example of how to implement a training pipeline. You can do better!

# %%
## Training hyperparameters.
#device = 'cuda' if torch.cuda.is_available else 'cpu'
#epochs = 100
#lr = 0.0001
#batch_size = 128
#
## Architecture hyperparameters.
#input_size = 28*28
#width = 16
#depth = 2
#
## Dataloaders.
#dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
#dl_val   = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4)
#dl_test  = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True, num_workers=4)
#
## Instantiate model and optimizer.
#model_mlp = MLP([input_size] + [width]*depth + [10]).to(device)
#opt = torch.optim.Adam(params=model_mlp.parameters(), lr=lr)
#
## Training loop.
#losses_and_accs = []
#for epoch in range(epochs):
#    loss = train_epoch(model_mlp, dl_train, opt, epoch, device=device)
#    (val_acc, _) = evaluate_model(model_mlp, dl_val, device=device)
#    losses_and_accs.append((loss, val_acc))
#
## And finally plot the curves.
#plot_validation_curves(losses_and_accs)
#print(f'Accuracy report on TEST:\n {evaluate_model(model_mlp, dl_test, device=device)[1]}')

# %% [markdown]
# ### Exercise 1.1: A baseline MLP
# 
# Implement a *simple* Multilayer Perceptron to classify the 10 digits of MNIST (e.g. two *narrow* layers). Use my code above as inspiration, but implement your own training pipeline -- you will need it later. Train this model to convergence, monitoring (at least) the loss and accuracy on the training and validation sets for every epoch. Below I include a basic implementation to get you started -- remember that you should write your *own* pipeline!
# 
# **Note**: This would be a good time to think about *abstracting* your model definition, and training and evaluation pipelines in order to make it easier to compare performance of different models.
# 
# **Important**: Given the *many* runs you will need to do, and the need to *compare* performance between them, this would **also** be a great point to study how **Tensorboard** or **Weights and Biases** can be used for performance monitoring.

# %%
# Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import random
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tensorboard
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import json

# %%
seed= 123
random.seed(seed)             
np.random.seed(seed)          
torch.manual_seed(seed)       
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)

# %%
class SkipBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SkipBlock,self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        identity = self.projection(x)
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + identity)


# %%
class My_MLP(nn.Module):
    def __init__(self,layer_sizes,use_skip=False):
        super(My_MLP, self).__init__()
        layers=[]
        layers.append(nn.Flatten()) 
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if use_skip==True:
                layers.append(SkipBlock(in_dim, out_dim))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
                if out_dim != layer_sizes[-1]:  
                    layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x=self.model(x)
        return x

# %%
def Load_Data():
    transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds_train= MNIST(root='./data', train= True, download=True, transform=transform)
    ds_test=MNIST(root='./data', train= False, download=True, transform=transform)

    
    return ds_train, ds_test

# %%
def Validation_Model(model, dl_val,device, batch_size):
    model.eval()
    predictions=[]
    ground_truth=[]
    criterion= torch.nn.CrossEntropyLoss()
    losses=[]
    dl_validation= DataLoader(dl_val, batch_size=batch_size, shuffle=False)
    
    for (data, labels) in tqdm(dl_validation, desc="Evaluating", leave=False):
        data= data.to(device)
        labels= labels.to(device)
        logits= model(data)
        loss= criterion(logits, labels)
        prediction= torch.argmax(logits, dim=1)
        losses.append(loss.item())
        ground_truth.append(labels.detach().cpu().numpy())
        predictions.append(prediction.detach().cpu().numpy())
    return (accuracy_score(np.hstack(ground_truth), np.hstack(predictions)),
            classification_report(np.hstack(ground_truth), np.hstack(predictions), zero_division=0, digits=3),
            np.mean(losses))


# %%
def get_grad_norms(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms


# %%
def Training_Model(model,X, file_writer,device,epochs=50,batch_size=8, learning_rate=0.001, weight_decay=0.001,study_grad=False):
    total_size = len(X)
    val_size = int(0.2 * total_size)  
    train_size = total_size - val_size

    indices = np.random.permutation(total_size)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    ds_train = Subset(X, train_indices)
    ds_val = Subset(X, val_indices)
    dl_train=DataLoader(ds_train, batch_size=batch_size,shuffle=True )
    

    optimizer= torch.optim.Adam(params=model.parameters(), lr= learning_rate, weight_decay=weight_decay)
    criterion= torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Model Training"):
        model.train()
        losses=[]
        count=0
        for (data,labels) in tqdm(dl_train, desc=f'Training epoch {epoch}', leave=True):
            data= data.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            output= model(data)
            loss= criterion(output, labels)
            loss.backward()
            if count == 0 and study_grad==True:
                grad_norms = get_grad_norms(model)

                # Esempio: loggare su TensorBoard
                for name, norm in grad_norms.items():
                    file_writer.add_scalar(f"GradNorms/{name}", norm, epoch)
            count=1
            optimizer.step()
            losses.append(loss.item())
        
        loss_average= np.mean(losses)
        print(loss_average)
        accurancy, report_dict, losses_val= Validation_Model(model, ds_val, device, batch_size)
        file_writer.add_scalars(
                "Loss",
                {
                    "Train": loss_average,
                    "Validation": losses_val
                },
                epoch
            )
        file_writer.add_scalar("Train/Accurancy", accurancy, epoch)
        report_str = json.dumps(report_dict, indent=4)
        file_writer.add_text("Train/Classification Report", f"<pre>{report_str}</pre>", epoch)

    return model
    

# %%
class Trainer(nn.Module):
    def __init__(self,model,logdir,date, num_classes,in_channels,depth=None,epochs=50, batch_size=8, learning_rate=0.001, weight_decay=0.001,path_exp= "Simple_MLP",study_grad=False):
        super(Trainer,self).__init__()
        self.model=model
        self.study_grad=study_grad
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate= learning_rate
        self.weight_decay= weight_decay
        self.beast_model= None
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path_experiments=f'{path_exp}/Run_{date}'
        self.num_classes= num_classes
        self.file_writer= SummaryWriter(logdir)
        self.in_channels= in_channels
        self.depth= depth
        


    def get_hyperparamtres_dict(self):
        result={
            'Epochs': self.epochs,
            'Batch size':self.batch_size,
            'Learning Rate': self.learning_rate,
            'Weight Decay':self.weight_decay,
            'Num Classes': self.num_classes
        }
        if self.depth is not None:
            result["depth"]= self.depth
        return result

    def save_hyperparametres(self):
        hyperparametres_dict= self.get_hyperparamtres_dict()
        path= os.path.join(self.path_experiments, 'hyperparametres.csv')
        file_exists = os.path.isfile(path)
        is_empty = not file_exists or os.stat(path).st_size == 0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=hyperparametres_dict.keys())
            if is_empty:
                writer.writeheader()
            writer.writerow(hyperparametres_dict)

    def Train(self,X):
        if not os.path.exists(self.path_experiments):
            os.makedirs(self.path_experiments)
        self.save_hyperparametres()
        self.model.to(self.device)
        self.beast_model= Training_Model(self.model,X, self.file_writer,self.device, self.epochs,self.batch_size, self.learning_rate, self.weight_decay)
        torch.save(self.beast_model.state_dict(), os.path.join(self.path_experiments,'beast_model.pt'))

    def Test(self,X, model=None):
        if model is None:
            acc, report_dict, loss= Validation_Model(self.beast_model, X, self.device,self.batch_size)
            self.file_writer.add_scalar("Test/Accurancy", acc, 0)
            self.file_writer.add_text("Test/Classification Report", f"<pre>{report_dict}</pre>", 1)
            self.file_writer.close()

        


# %%
#now= datetime.datetime.now()
#data_ora_formattata = now.strftime("%d_%m_%yT%H_%M")
#name= f'run_{data_ora_formattata}'
#logdir= f'tensorboard/Sample_MLP/{name}'
#input_size = 28*28
#width = 16
#depth = 2
#channels= [input_size] + [width]*depth + [10]
#
#minist_train, minist_test= Load_Data()
#model= My_MLP(channels)
#trainer= Trainer(model, logdir,data_ora_formattata,minist_train.classes,channels,100,128,0.001) 
#
#trainer.Train(minist_train)
#trainer.Test(minist_test)
#


# %% [markdown]
# ### Exercise 1.2: Adding Residual Connections
# 
# Implement a variant of your parameterized MLP network to support **residual** connections. Your network should be defined as a composition of **residual MLP** blocks that have one or more linear layers and add a skip connection from the block input to the output of the final linear layer.
# 
# **Compare** the performance (in training/validation loss and test accuracy) of your MLP and ResidualMLP for a range of depths. Verify that deeper networks **with** residual connections are easier to train than a network of the same depth **without** residual connections.
# 
# **For extra style points**: See if you can explain by analyzing the gradient magnitudes on a single training batch *why* this is the case. 

# %%
#now= datetime.datetime.now()
#data_ora_formattata = now.strftime("%d_%m_%yT%H_%M")
#name= f'run_{data_ora_formattata}'
#
#input_size = 28*28
#width = 16
#depths = [2,6,10]
#minist_train, minist_test= Load_Data()
#
#for depth in depths:
#    for use_skip in [True,False]:
#        model= My_MLP(channels,use_skip=use_skip )
#        channels= [input_size] + [width]*depth + [10]
#        if use_skip:
#            print(f'Run Training of Residual_depth{depth} ')
#            logdir= f'tensorboard/Residual_vs_Simple_MLP/{name}/Residual_depth{depth}'
#            path=f"Residual_vs_Simple_MLP/Residual_depth{depth}"
#            
#        else:
#            print(f'Run Training of Simple_depth{depth} ')
#            logdir= f'tensorboard/Residual_vs_Simple_MLP/{name}/Simple_depth{depth}'
#            path=f"Residual_vs_Simple_MLP/Simple_depth{depth}"
#            
#        trainer= Trainer(model,logdir,data_ora_formattata,minist_train.classes,0,100,128,0.001,0.001,path,True)
#
#        trainer.Train(minist_train)
#        trainer.Test(minist_test)


# %% [markdown]
# ### Exercise 1.3: Rinse and Repeat (but with a CNN)
# 
# Repeat the verification you did above, but with **Convolutional** Neural Networks. If you were careful about abstracting your model and training code, this should be a simple exercise. Show that **deeper** CNNs *without* residual connections do not always work better and **even deeper** ones *with* residual connections.
# 
# **Hint**: You probably should do this exercise using CIFAR-10, since MNIST is *very* easy (at least up to about 99% accuracy).
# 
# **Tip**: Feel free to reuse the ResNet building blocks defined in `torchvision.models.resnet` (e.g. [BasicBlock](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L59) which handles the cascade of 3x3 convolutions, skip connections, and optional downsampling). This is an excellent exercise in code diving. 
# 
# **Spoiler**: Depending on the optional exercises you plan to do below, you should think *very* carefully about the architectures of your CNNs here (so you can reuse them!).

# %%
import torchvision
from torchvision.models.resnet import BasicBlock

# %%
def Load_data_Cifar10():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    return train_set, test_set

# %%
class Residual_Block_CNN(nn.Module):
    def __init__(self, in_channels, use_resnet):
        super(Residual_Block_CNN,self).__init__()
        self.use_resnet= use_resnet

        if use_resnet:
            self.block_res= BasicBlock(in_channels, in_channels, stride=1, downsample=None)
        else:
            self.first_layer= nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels)
            )
            self.second_layer=nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3),
                nn.BatchNorm2d(in_channels)
            )
            self.relu= nn.ReLU()

    def forward(self,x):
        if self.use_resnet:
            return self.block_res(x)
        else:
            identity= x
            out= self.first_layer(x)
            out= self.relu(x)
            out= self.second_layer(x)
            out= out  + identity
            return self.relu(out)


# %%
class CNN_Block(nn.Module):
    def __init__(self, in_channels):
        super(CNN_Block, self).__init__()
        self.block= nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
    
    def forward(self,x):
        return self.block(x)
    

# %%
class CNN_Customize(nn.Module):
    def __init__(self,depth, in_channels,out_channels, num_classes, use_skip, use_resnet ):
        super(CNN_Customize, self).__init__()
        
        self.head= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        blocks=[]

        for i in range(depth):
            if use_skip:
                blocks.append(Residual_Block_CNN(out_channels, use_resnet))
            else:
                blocks.append(CNN_Block(out_channels))

        self.blocks= nn.Sequential(*blocks)
        self.pooling= nn.AdaptiveAvgPool2d((1,1))
        self.fully_connected= nn.Linear(out_channels, num_classes)

    def forward(self, x):
        out= self.head(x)
        out= self.blocks(out)
        out= self.pooling(out)
        out= torch.flatten(out, 1)
        return self.fully_connected(out)

# %%
now= datetime.datetime.now()
data_ora_formattata = now.strftime("%d_%m_%yT%H_%M")
name= f'run_{data_ora_formattata}'

in_channels = 3
out_channels= 64
depths = [2 ,6 ,10]
num_classes=10
cifar_train, cifartest= Load_data_Cifar10()

for depth in depths:
    for use_skip in [True,False]:

        if use_skip:
            print(f'Run Training of Residual_CNN{depth} ')
            logdir= f'tensorboard/CNN_Residual_vs_Base/{name}/Residual_depth{depth}'
            path=f"CNN_Residual_vs_Base/Residual_depth{depth}"
            
            use_res=True
        else:
            print(f'Run Training of Simple_depth{depth} ')
            logdir= f'tensorboard/CNN_Residual_vs_Base/{name}/Simple_depth{depth}'
            path=f"CNN_Residual_vs_Base/Simple_depth{depth}"
            
            use_res=False
        model= CNN_Customize(depth,in_channels,out_channels,num_classes,use_skip,use_res)
        trainer= Trainer(model,logdir,data_ora_formattata,num_classes,0,depth,100,128,0.001,0.001,path)

        trainer.Train(cifar_train)
        trainer.Test(cifartest)



# %% [markdown]
# -----
# ## Exercise 2: Choose at Least One
# 
# Below are **three** exercises that ask you to deepen your understanding of Deep Networks for visual recognition. You must choose **at least one** of the below for your final submission -- feel free to do **more**, but at least **ONE** you must submit. Each exercise is designed to require you to dig your hands **deep** into the guts of your models in order to do new and interesting things.
# 
# **Note**: These exercises are designed to use your small, custom CNNs and small datasets. This is to keep training times reasonable. If you have a decent GPU, feel free to use pretrained ResNets and larger datasets (e.g. the [Imagenette](https://pytorch.org/vision/0.20/generated/torchvision.datasets.Imagenette.html#torchvision.datasets.Imagenette) dataset at 160px).

# %% [markdown]
# ### Exercise 2.1: *Fine-tune* a pre-trained model
# Train one of your residual CNN models from Exercise 1.3 on CIFAR-10. Then:
# 1. Use the pre-trained model as a **feature extractor** (i.e. to extract the feature activations of the layer input into the classifier) on CIFAR-100. Use a **classical** approach (e.g. Linear SVM, K-Nearest Neighbor, or Bayesian Generative Classifier) from scikit-learn to establish a **stable baseline** performance on CIFAR-100 using the features extracted using your CNN.
# 2. Fine-tune your CNN on the CIFAR-100 training set and compare with your stable baseline. Experiment with different strategies:
#     - Unfreeze some of the earlier layers for fine-tuning.
#     - Test different optimizers (Adam, SGD, etc.).
# 
# Each of these steps will require you to modify your model definition in some way. For 1, you will need to return the activations of the last fully-connected layer (or the global average pooling layer). For 2, you will need to replace the original, 10-class classifier with a new, randomly-initialized 100-class classifier.

# %%


# %% [markdown]
# ### Exercise 2.2: *Distill* the knowledge from a large model into a smaller one
# In this exercise you will see if you can derive a *small* model that performs comparably to a larger one on CIFAR-10. To do this, you will use [Knowledge Distillation](https://arxiv.org/abs/1503.02531):
# 
# > Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network, NeurIPS 2015.
# 
# To do this:
# 1. Train one of your best-performing CNNs on CIFAR-10 from Exercise 1.3 above. This will be your **teacher** model.
# 2. Define a *smaller* variant with about half the number of parameters (change the width and/or depth of the network). Train it on CIFAR-10 and verify that it performs *worse* than your **teacher**. This small network will be your **student** model.
# 3. Train the **student** using a combination of **hard labels** from the CIFAR-10 training set (cross entropy loss) and **soft labels** from predictions of the **teacher** (Kulback-Leibler loss between teacher and student).
# 
# Try to optimize training parameters in order to maximize the performance of the student. It should at least outperform the student trained only on hard labels in Setp 2.
# 
# **Tip**: You can save the predictions of the trained teacher network on the training set and adapt your dataloader to provide them together with hard labels. This will **greatly** speed up training compared to performing a forward pass through the teacher for each batch of training.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 2.3: *Explain* the predictions of a CNN
# 
# Use the CNN model you trained in Exercise 1.3 and implement [*Class Activation Maps*](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.):
# 
# > B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. Learning Deep Features for Discriminative Localization. CVPR'16 (arXiv:1512.04150, 2015).
# 
# Use your CNN implementation to demonstrate how your trained CNN *attends* to specific image features to recognize *specific* classes. Try your implementation out using a pre-trained ResNet-18 model and some images from the [Imagenette](https://pytorch.org/vision/0.20/generated/torchvision.datasets.Imagenette.html#torchvision.datasets.Imagenette) dataset -- I suggest you start with the low resolution version of images at 160px.

# %%
# Your code here.



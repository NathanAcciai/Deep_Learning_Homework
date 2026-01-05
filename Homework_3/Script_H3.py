# %%
import datasets
from datasets import load_dataset, get_dataset_split_names
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
import matplotlib as plt
import torchvision
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import accelerate
import sentencepiece
import ipywidgets
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm
import json 
from datetime import datetime
import os



# %% [markdown]
# ## Working with Transformers in the HuggingFace Ecosystem
# 
# In this laboratory exercise we will learn how to work with the HuggingFace ecosystem to adapt models to new tasks. As you will see, much of what is required is *investigation* into the inner-workings of the HuggingFace abstractions. With a little work, a little trial-and-error, it is fairly easy to get a working adaptation pipeline up and running.

# %% [markdown]
# ### Exercise 1: Sentiment Analysis (warm up)
# 
# In this first exercise we will start from a pre-trained BERT transformer and build up a model able to perform text sentiment analysis. Transformers are complex beasts, so we will build up our pipeline in several explorative and incremental steps.
# 
# #### Exercise 1.1: Dataset Splits and Pre-trained model
# There are a many sentiment analysis datasets, but we will use one of the smallest ones available: the [Cornell Rotten Tomatoes movie review dataset](cornell-movie-review-data/rotten_tomatoes), which consists of 5,331 positive and 5,331 negative processed sentences from the Rotten Tomatoes movie reviews.
# 
# **Your first task**: Load the dataset and figure out what splits are available and how to get them. Spend some time exploring the dataset to see how it is organized. Note that we will be using the [HuggingFace Datasets](https://huggingface.co/docs/datasets/en/index) library for downloading, accessing, splitting, and batching data for training and evaluation.

# %%
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train_dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
valid_dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
test_dataset  = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")

# %% [markdown]
# #### Exercise 1.2: A Pre-trained BERT and Tokenizer
# 
# The model we will use is a *very* small BERT transformer called [Distilbert](https://huggingface.co/distilbert/distilbert-base-uncased) this model was trained (using self-supervised learning) on the same corpus as BERT but using the full BERT base model as a *teacher*.
# 
# **Your next task**: Load the Distilbert model and corresponding tokenizer. Use the tokenizer on a few samples from the dataset and pass the tokens through the model to see what outputs are provided. I suggest you use the [`AutoModel`](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html) class (and the `from_pretrained()` method) to load the model and `AutoTokenizer` to load the tokenizer).

# %%
#model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
#
## %%
##estraggo del testo che mi fornisce il dataset di train 
#texts= train_dataset["text"][:2]
##passo il testo delle due frasi selezionate al tokenizzatore di testo che restituisce inputs_ids e attention mask
#encoding= tokenizer(
#    texts,
#    padding= True,
#    truncation= True,
#    return_tensors="pt" 
#)
#print(encoding)
'''
    inputs_ids rappresenta l'assegnazione numerica data dopo la tokenizzazione, infatti spezza la frase in token e restituisce 
    per ogni token gli id interi ripresi dal vocabolario del modello
    mentre attention_mask indica quali token il modello deve cosniderare, così da ignorare il padding e evitare che il modello 
    presti attenzione a token finiti. in questo modo si ha più pulizia e minor rumore quando si calcola l 'attenzione all' interno del 
    mdoello.
    '''
    #In. questo caso si va a passare sia gli id che la maschera di attenzione al modello in modo elegante che estrae le feature
#with torch.no_grad():
#    outputs= model(**encoding)
##print(outputs)
#    #l'ultimo layer mi restituisce la rappresentazione contestualizzata della frase ed è prioprio questo che viene usato come rappresentazione
#    #per classificare i serntimenti.
#outputs.last_hidden_state.shape

# %%
#train_dataset["text"][:2][1]
#
## %%
#tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][1])
#print(tokens)
#len(tokens)

# %% [markdown]
# #### Exercise 1.3: A Stable Baseline
# 
# In this exercise I want you to:
# 1. Use Distilbert as a *feature extractor* to extract representations of the text strings from the dataset splits;
# 2. Train a classifier (your choice, by an SVM from Scikit-learn is an easy choice).
# 3. Evaluate performance on the validation and test splits.
# 
# These results are our *stable baseline* -- the **starting** point on which we will (hopefully) improve in the next exercise.
# 
# **Hint**: There are a number of ways to implement the feature extractor, but probably the best is to use a [feature extraction `pipeline`](https://huggingface.co/tasks/feature-extraction). You will need to interpret the output of the pipeline and extract only the `[CLS]` token from the *last* transformer layer. *How can you figure out which output that is?*

# %%
'''
L'estrattore delle features costruito cosi restituisce una lista lunga quanto i campioni 
Le quali contengono al suo interno una lista composta da le liste di token della frase
Ora per come viene usato il tokenizzatore in questione il primo vettore risulta essere quello di CLS mentre i successivi sono quelli ovviameente della parola della frase
infine come ultimi può inserire dei token di padding oppure di separazione.
'''
def extract_features_with_pipeline(model, tokenizer, texts, batch_size=32):
    #costruisco extractor
    extractor = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer,
        framework="pt"
    )

    all_features = []

    for i in tqdm(range(0, len(texts), batch_size), desc= "Features Extraction"):
        batch_texts = texts[i:i + batch_size]
        #passo il batch di testi 
        outputs = extractor(batch_texts)

        # estrai CLS per ogni frase che è nella posizione [0][0]
        cls_batch = [sentence[0][0] for sentence in outputs]
        all_features.extend(cls_batch)

    return np.array(all_features)


# %%
def FitSVM(model, clf, tokenizer, dataset):
    features= extract_features_with_pipeline(model, tokenizer,dataset["text"])
    labels= np.array(dataset["label"])
    clf.fit(features, labels)
    return clf

# %%
def save_report(dir,acc, cm, report,validation=True):
    labels = ["negative", "positive"]
    report_dict = {
        "accuracy": acc,
        "classification_report": report,
         "confusion_matrix": {labels[i]: {labels[j]: int(cm[i,j]) for j in range(len(labels))} for i in range(len(labels))}     
    }
    if validation:
        path= f"{dir}/svm_report_validation.json"
    else:
        path=f"{dir}/svm_report_test.json"
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=4)


# %%
def EvaluateSVM(model, clf, tokenizer, datasets,dir,validation=True):
    features_validation= extract_features_with_pipeline(model, tokenizer, datasets["text"])
    labels_validation= np.array(valid_dataset["label"])
    y_pred = clf.predict(features_validation)
    acc = accuracy_score(labels_validation, y_pred)
    report = classification_report(labels_validation, y_pred, target_names=["negative", "positive"], output_dict=True)
    acc = accuracy_score(labels_validation, y_pred)
    cm = confusion_matrix(labels_validation, y_pred)
    save_report(dir, acc, cm ,report, validation)
    return report, acc, cm

# %%
''' 
Per come viene processato il token all' interno di bert si ha che le frasi vengono rappresentati con il token speciale
[CLS] come primo token che da una rappresentazione globale della frase, di norma usato proprio per la classificazione
Infatti questo tipo di token è ottimizzato per riassumere il significato intero dell' intera frase.
Infatti dall' esempio prima si vede come la riconversione in token diano la parola e come primo token sempre il CLS.
'''
#seed= 42
#max_iter_svm =500000
#model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
#clf = LinearSVC(max_iter=max_iter_svm)
#
#now = datetime.now()
#formatted_data = now.strftime("%Y-%m-%d_%H-%M-%S")
#dir= f'run_{formatted_data}_{max_iter_svm}'
#os.makedirs(dir, exist_ok=True)
#clf= FitSVM(model, clf, tokenizer, train_dataset.shuffle(seed=seed))
#_ = EvaluateSVM(model, clf, tokenizer, valid_dataset, dir)
#_ = EvaluateSVM(model, clf, tokenizer, test_dataset, dir,False)
#



# %% [markdown]
# -----
# ### Exercise 2: Fine-tuning Distilbert

# %% [markdown]
# In this exercise we will fine-tune the Distilbert model to (hopefully) improve sentiment analysis performance.

# %% [markdown]
# #### Exercise 2.1: Token Preprocessing
# 
# The first thing we need to do is *tokenize* our dataset splits. Our current datasets return a dictionary with *strings*, but we want *input token ids* (i.e. the output of the tokenizer). This is easy enough to do my hand, but the HugginFace `Dataset` class provides convenient, efficient, and *lazy* methods. See the documentation for [`Dataset.map`](https://huggingface.co/docs/datasets/v3.5.0/en/package_reference/main_classes#datasets.Dataset.map).
# 
# **Tip**: Verify that your new datasets are returning for every element: `text`, `label`, `intput_ids`, and `attention_mask`.

# %%
train_dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
valid_dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
test_dataset  = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")

# %%
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# %%
def tokenizer_fn(batch):
    return tokenizer(
        batch["text"],
        padding= True,
        truncation= True,
        return_tensors="pt" 
    )

# %%
tokenized_train= train_dataset.map(tokenizer_fn, batched=True)
tokenized_validation= valid_dataset.map(tokenizer_fn, batched=True)
tokenized_test= test_dataset.map(tokenizer_fn, batched=True )

# %%
#print(tokenized_train)
#print(tokenized_validation)
#print(tokenized_test)

# %% [markdown]
# #### Exercise 2.2: Setting up the Model to be Fine-tuned
# 
# In this exercise we need to prepare the base Distilbert model for fine-tuning for a *sequence classification task*. This means, at the very least, appending a new, randomly-initialized classification head connected to the `[CLS]` token of the last transformer layer. Luckily, HuggingFace already provides an `AutoModel` for just this type of instantiation: [`AutoModelForSequenceClassification`](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification). You will want you instantiate one of these for fine-tuning.

# %%
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer= DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# %% [markdown]
# #### Exercise 2.3: Fine-tuning Distilbert
# 
# Finally. In this exercise you should use a HuggingFace [`Trainer`](https://huggingface.co/docs/transformers/main/en/trainer) to fine-tune your model on the Rotten Tomatoes training split. Setting up the trainer will involve (at least):
# 
# 
# 1. Instantiating a [`DataCollatorWithPadding`](https://huggingface.co/docs/transformers/en/main_classes/data_collator) object which is what *actually* does your batch construction (by padding all sequences to the same length).
# 2. Writing an *evaluation function* that will measure the classification accuracy. This function takes a single argument which is a tuple containing `(logits, labels)` which you should use to compute classification accuracy (and maybe other metrics like F1 score, precision, recall) and return a `dict` with these metrics.  
# 3. Instantiating a [`TrainingArguments`](https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/trainer#transformers.TrainingArguments) object using some reasonable defaults.
# 4. Instantiating a `Trainer` object using your train and validation splits, you data collator, and function to compute performance metrics.
# 5. Calling `trainer.train()`, waiting, waiting some more, and then calling `trainer.evaluate()` to see how it did.
# 
# **Tip**: When prototyping this laboratory I discovered the HuggingFace [Evaluate library](https://huggingface.co/docs/evaluate/en/index) which provides evaluation metrics. However I found it to have insufferable layers of abstraction and getting actual metrics computed. I suggest just using the Scikit-learn metrics...

# %%
def compute_metrics(eval_pred):
    logits, labels= eval_pred
    #prendo la classe con probabilità maggiore
    preds= np.argmax(logits, axis=-1)
    
    report = classification_report(labels, preds, output_dict=True)
    metrics={
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1-score": report["macro avg"]["f1-score"]
    }
    return metrics


# %%
def build_trainer(model, tokenizer, output_dir, train_dataset, eval_dataset, num_epochs=3,train_batch_size=8,eval_batch_size=8,lr=5e-5):
    data_collator= transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    trainer_args= transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size= eval_batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        eval_strategy="epoch"
    )

    trainer = transformers.Trainer(
        model= model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer


# %%
tokenizer= DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')



now = datetime.now()
formatted_data = now.strftime("%Y-%m-%d_%H-%M-%S")
general_dir= "Trainer_Distilbert"
dir= f'{general_dir}/run_{formatted_data}'
os.makedirs(dir, exist_ok=True)

trainer= build_trainer(model, tokenizer, dir, tokenized_train, tokenized_validation,10,8,8)

trainer.train()


# %% [markdown]
# -----
# ### Exercise 3: Choose at Least One
# 

# %% [markdown]
# #### Exercise 3.1: Efficient Fine-tuning for Sentiment Analysis (easy)

# %% [markdown]
# In Exercise 2 we fine-tuned the *entire* Distilbert model on Rotten Tomatoes. This is expensive, even for a small model. Find an *efficient* way to fine-tune Distilbert on the Rotten Tomatoes dataset (or some other dataset).
# 
# **Hint**: You could check out the [HuggingFace PEFT library](https://huggingface.co/docs/peft/en/index) for some state-of-the-art approaches that should "just work". How else might you go about making fine-tuning more efficient without having to change your training pipeline from above?

# %%
# Your code here.

# %% [markdown]
# #### Exercise 3.2: Fine-tuning a CLIP Model (harder)
# 
# Use a (small) CLIP model like [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) and evaluate its zero-shot performance on a small image classification dataset like ImageNette or TinyImageNet. Fine-tune (using a parameter-efficient method!) the CLIP model to see how much improvement you can squeeze out of it.
# 
# **Note**: There are several ways to adapt the CLIP model; you could fine-tune the image encoder, the text encoder, or both. Or, you could experiment with prompt learning.
# 
# **Tip**: CLIP probably already works very well on ImageNet and ImageNet-like images. For extra fun, look for an image classification dataset with different image types (e.g. *sketches*).

# %%
# Your code here.

# %% [markdown]
# #### Exercise 3.3: Choose your Own Adventure

# %% [markdown]
# There are a *ton* of interesting and fun models on the HuggingFace hub. Pick one that does something interesting and adapt it in some way to a new task. Or, combine two or more models into something more interesting or fun. The sky's the limit.
# 
# **Note**: Reach out to me by email or on the Discord if you are unsure about anything.

# %%
# Your code here.



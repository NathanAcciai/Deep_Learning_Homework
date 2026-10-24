# Homework 3 — Transformers and the Hugging Face Ecosystem

Adapting pre-trained Transformers to new tasks with Hugging Face: **DistilBERT** for sentiment analysis (frozen features vs full fine-tuning) and **CLIP** with **LoRA** (parameter-efficient fine-tuning) on three image-classification datasets, including a hard domain shift (sketches).

**Report:** [Weights & Biases](https://wandb.ai/nathan-acciai-universit-di-firenze/Homework_3/reports/Homework-3-Report--VmlldzoxNTcxMjExMQ?accessToken=zeu8m2dyd65fu6oyuuvl8kabn15ej3hfrlxa2usaqfovqbl4z76vpoa4xaau949c)

## Contents

```
Homework_3/
├── Lab3-Transformers.ipynb     # main notebook
├── Script_H3.py                # script export of the notebook (functions and runs)
├── classnames.txt              # the 1000 ImageNet class names used to build CLIP prompts
├── Exercise_1/                 # DistilBERT features + LinearSVC: validation/test reports (JSON)
├── Exercise_2/Trainer_Distilbert/   # full fine-tuning with the HF Trainer: reports + TensorBoard
└── Exercise_3/                 # CLIP experiments
    ├── ImageNette/
    ├── TinyImagenet/
    └── SketchImageNet/
        ├── Validation_Zero_Shot/            # zero-shot report
        └── Lora_Fine-Tuning/
            ├── Text_Encoder/                # LoRA on the text encoder
            ├── Vision_Encoder/              # LoRA on the vision encoder
            └── Text_and_Vision_Encoder/     # LoRA on both
```

Each report JSON contains accuracy, per-class precision/recall/F1 and the confusion matrix. Checkpoints are not versioned (`.gitignore`).

## Exercise 1–2 — Sentiment analysis on Rotten Tomatoes (DistilBERT)

Dataset: `cornell-movie-review-data/rotten_tomatoes` (binary sentiment, train/validation/test splits).

1. **Stable baseline.** `distilbert-base-uncased` as frozen feature extractor: the `[CLS]` vector of the last layer is taken from the Hugging Face feature-extraction pipeline and fed to a `LinearSVC`.
2. **Full fine-tuning.** `DistilBertForSequenceClassification` with the HF `Trainer`, `DataCollatorWithPadding` and custom metrics (accuracy, precision, recall, F1).

| Approach | Validation acc. (%) | Test acc. (%) |
|---|---|---|
| Frozen features + Linear SVM | 82.18 | 79.83 |
| Fine-tuning, 3 epochs | 84.24 | **84.43** |
| Fine-tuning, 5 epochs | 84.05 | — |

The two SVM runs in `Exercise_1` (`max_iter` 50 000 and 500 000) give identical results, i.e. the solver had already converged.

## Exercise 3 — CLIP with LoRA

Model: `openai/clip-vit-base-patch16`. Zero-shot classification compares each image with the prompts *"This is a photo of {class}."* for all the classes. Fine-tuning uses **LoRA** (PEFT): rank 4, α = 14, dropout 0.1, applied to `q_proj` and `v_proj`; AdamW, lr 1e-5, batch size 4. LoRA is applied to the text encoder, to the vision encoder, or to both. ImageNet-Sketch comes from `clip-benchmark/wds_imagenet_sketch` (1000 classes).

Validation accuracy (%) after each epoch:

**ImageNette** — zero-shot **97.74**

| LoRA on | ep. 0 | ep. 1 | ep. 2 | ep. 3 | ep. 4 |
|---|---|---|---|---|---|
| Text encoder | 98.42 | 98.87 | 98.78 | 98.85 | 97.74 |
| Vision encoder | 99.55 | 99.48 | 99.64 | 99.64 | 99.62 |
| Text + vision | 99.57 | 99.59 | 99.62 | 99.59 | 99.62 |

**Tiny ImageNet** — zero-shot **61.45**

| LoRA on | ep. 0 | ep. 1 | ep. 2 |
|---|---|---|---|
| Text encoder | 79.77 | 81.10 | 80.17 |
| Vision encoder | 79.97 | 80.47 | 81.23 |
| Text + vision | 79.04 | 80.60 | 81.11 |

**ImageNet-Sketch** — zero-shot **43.75**

| LoRA on | ep. 0 | ep. 1 | ep. 2 |
|---|---|---|---|
| Text encoder | 60.75 | 67.60 | 67.24 |
| Vision encoder | 65.90 | 71.19 | **73.57** |
| Text + vision | 63.22 | 67.07 | 70.16 |

Take-aways:
- On ImageNette, CLIP is already close to saturation (97.7% zero-shot); LoRA adds about two points, mostly through the vision encoder. Text-only LoRA peaks at epoch 1 and then falls back to the zero-shot level.
- On Tiny ImageNet, LoRA gains about 20 points whichever encoder is adapted.
- On sketches, where the domain shift is largest, adapting the **vision encoder** gives the biggest gain (+29.8 points over zero-shot), more than adapting the text side or both.

## How to run

```bash
pip install torch torchvision transformers datasets peft accelerate scikit-learn tensorboard tqdm jupyter
jupyter notebook Lab3-Transformers.ipynb
tensorboard --logdir Exercise_2       # or any Exercise_3 run folder
```

Datasets and models are downloaded automatically from the Hugging Face Hub. A GPU is recommended for Exercise 3 (the ImageNet-Sketch runs use the full 1000-class validation set).

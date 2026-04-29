# Cross-Resolution Consistency Learning for Robust Image Classification

> Training deep learning models that maintain consistent predictions across multiple image resolutions, demonstrated on the Bean Leaf Disease classification task.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Method](#method)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Architectures](#architectures)
- [Loss Function](#loss-function)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup & Usage](#setup--usage)
- [Configuration](#configuration)
- [Results](#results)
- [Requirements](#requirements)

---

## Overview

Deep learning classifiers are often brittle to changes in image resolution — a model trained on high-resolution images may fail when the same image is captured at a lower resolution. This project investigates **Cross-Resolution Consistency Learning**: a training strategy that forces a model to produce the same class prediction regardless of whether the input image is at full resolution or a degraded version of it.

The approach is applied to **Bean Leaf Disease classification** (3 classes: Angular Leaf Spot, Bean Rust, Healthy) and evaluated across three fundamentally different neural architectures.

---

## Problem Statement

Standard classification training only supervises the model on a single resolution. This project addresses two questions:

1. Can we train a single model to be robust across a range of input resolutions by adding a consistency constraint to the loss?
2. Do different architectures (CNN, ResNet, ViT) differ in their natural tendency towards scale-invariance?

---

## Method

### Multi-Scale Input Generation

For every training image, four resolution versions are generated and fed through the model in the same forward pass:

| Scale | Effective Resolution | Process |
|-------|---------------------|---------|
| 1.00  | 224 × 224           | Original (no degradation) |
| 0.75  | 168 → upscaled to 224 | Downscale then upscale |
| 0.50  | 112 → upscaled to 224 | Downscale then upscale |
| 0.25  |  56 → upscaled to 224 | Downscale then upscale |

The downscale-then-upscale operation simulates what happens when an image is captured or transmitted at lower quality. All versions are padded back to 224×224 so the same model weights handle all resolutions.

### Consistency Enforcement

Beyond standard cross-entropy on the full-resolution prediction, an additional **consistency loss** penalises the model whenever its softmax distributions differ between scales. This pushes the model to develop features that are inherently resolution-invariant.

---

## Project Structure

```
CrossResolution_BeanDisease.ipynb   ← Main Colab notebook (all-in-one)
│
├── config.py          Written to disk by the notebook
├── dataset.py         ─ multi-scale dataset & DataLoader
├── models.py          ─ VGGNet, ResNetModel, ViTModel
├── losses.py          ─ ConsistencyLoss + CrossResolutionLoss
├── trainer.py         ─ training loop with gradient clipping
├── evaluator.py       ─ per-scale metrics, PSI, CSA, ECE
│
├── data/
│   └── bean_disease/
│       ├── train/     angular_leaf_spot / bean_rust / healthy
│       └── val/
│
├── checkpoints/       best_{arch}.pth  saved per architecture
├── logs/              history_{arch}.json  training curves
└── results/           eval_{arch}_test.json  + curves_{arch}.png
```

---

## Dataset

**Bean Leaf Lesions Classification** ([Kaggle — marquis03](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification))

| Split | Angular Leaf Spot | Bean Rust | Healthy | Total |
|-------|:-----------------:|:---------:|:-------:|------:|
| Train | 345               | 348       | 342     | 1,035 |
| Val   | 44                | 45        | 44      |   133 |

Classes are well-balanced (~33% each), so no class weighting is required.

**Training augmentations:** random crop, horizontal/vertical flip, colour jitter (brightness, contrast, saturation, hue), random rotation ±15°, ImageNet normalisation.

**Val/test transforms:** resize to 224×224 + ImageNet normalisation only.

---

## Architectures

### 1. VGGNet (from scratch)
A lightweight VGG-style CNN with 5 pooling blocks and a 3-layer classifier head. No pretrained weights — trained entirely on the bean dataset.

```
Input (3, 224, 224)
→ VGGBlock(3→64)   → 112×112
→ VGGBlock(64→128) →  56×56
→ VGGBlock(128→256)→  28×28
→ VGGBlock(256→512)→  14×14
→ VGGBlock(512→512)→   7×7
→ AdaptiveAvgPool(4×4)
→ FC(8192→1024) → FC(1024→512) → FC(512→3)
```

### 2. ResNet-18 (pretrained)
Standard ResNet-18 backbone with ImageNet pretrained weights. The final fully-connected layer is replaced with a `Dropout(0.3) → Linear(512→3)` head and fine-tuned end-to-end.

### 3. Vision Transformer — ViT (from scratch)
A lightweight custom ViT with ~14M parameters, designed to fit within Colab T4 VRAM.

| Hyperparameter | Value |
|----------------|-------|
| Patch size     | 16×16 |
| Embedding dim  | 384   |
| Transformer depth | 6  |
| Attention heads | 6    |
| MLP dim        | 1024  |
| Dropout        | 0.1   |

Images are split into 14×14 = 196 patches. A learnable `[CLS]` token is prepended, and its final hidden state is used for classification.

---

## Loss Function

```
Total Loss = CE(logits_original, labels)
           + λ × ConsistencyLoss(logits at all 4 scales)
```

### Cross-Entropy (CE)
Standard label-smoothed cross-entropy (`smoothing=0.1`) applied only to the full-resolution (scale=1.0) prediction.

### Consistency Loss
Mean pairwise KL divergence between temperature-softened predictions across all scale pairs:

```
L_consist = (1/|P|) × Σ_{(i,j)∈P} KL( softmax(z_i/T) || softmax(z_j/T) )
```

Where `P` is the set of all ordered pairs of scales, and `T=2.0` is a temperature that softens the distributions for a smoother gradient signal.

**Key hyperparameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `λ` (lambda_consist) | 0.5 | Weight of consistency vs classification |
| `T` (temperature) | 2.0 | Softens distributions; higher = smoother KL |
| label_smoothing | 0.1 | Prevents overconfidence |

---

## Evaluation Metrics

The evaluator computes the following for **each architecture × each scale**:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Top-1 classification accuracy |
| **Macro F1** | Unweighted mean F1 across all 3 classes |
| **Mean Entropy** | Average Shannon entropy of softmax outputs — lower = more confident |
| **ECE** | Expected Calibration Error — how well confidence matches accuracy (lower = better calibrated) |
| **PSI** | Prediction Stability Index — fraction of samples where all 4 scales agree on the same class (1.0 = perfect scale-invariance) |
| **CSA** | Cross-Scale Agreement — pairwise agreement rate between each pair of scales |

---

## Setup & Usage

### Google Colab (recommended)

1. Open the notebook in Colab and set **Runtime → Change runtime type → T4 GPU**
2. Run **Cell 1** to install `scikit-learn`
3. Run **Cell 2** — upload your `kaggle.json` when prompted (get it from [kaggle.com/settings](https://www.kaggle.com/settings))
4. Run all remaining cells top-to-bottom

> The notebook auto-detects the dataset folder structure and moves files into the expected layout automatically.

### Training order

The notebook trains architectures sequentially with GPU memory cleared between each:

```python
ARCHS_TO_TRAIN = ["resnet", "vit", "vgg"]  # change to ["resnet"] for a single model
```

Best checkpoints are saved automatically to `checkpoints/best_{arch}.pth` whenever validation accuracy improves.

### Outputs

After training and evaluation, download everything with **Cell 11**:
- `results.zip` — per-scale JSON metrics + training curve PNGs
- `logs.zip` — training history JSON per architecture
- `checkpoints.zip` — best model weights per architecture

---

## Configuration

All hyperparameters are centralised in `config.py`:

```python
# Resolution
BASE_SIZE      = 224
SCALES         = [1.0, 0.75, 0.50, 0.25]

# Training
BATCH_SIZE     = 16          # 16 recommended for T4; use 8 for VGG if OOM
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
SCHEDULER      = "cosine"    # cosine annealing from 1e-3 → 1e-6

# Consistency loss
LAMBDA_CONSIST = 0.5
TEMP           = 2.0

# ViT
VIT_PATCH_SIZE = 16
VIT_DIM        = 384
VIT_DEPTH      = 6
VIT_HEADS      = 6
VIT_MLP_DIM    = 1024
```

---

## Results

After training, the summary table (Cell 10) shows accuracy and PSI broken down by architecture and input scale. Key things to look for:

- **Accuracy drop across scales** — how much does each model degrade as resolution decreases?
- **PSI score** — which architecture is most self-consistent across resolutions?
- **CSA between 1.00 and 0.25** — the hardest pair; large agreement here means strong scale-invariance.
- **ECE** — is the model well-calibrated, or overconfident?

A well-trained model under this framework should show: high PSI (>0.85), small accuracy drop between scale 1.0 and 0.5, and a consistently low consistency loss during training.

---

## Requirements

| Package | Version |
|---------|---------|
| Python  | 3.10+   |
| PyTorch | 2.0+    |
| torchvision | 0.15+ |
| scikit-learn | 1.0+ |
| numpy   | 1.23+   |
| Pillow  | 9.0+    |
| matplotlib | 3.5+ |
| pandas  | 1.4+    |

All packages except `scikit-learn` are pre-installed in Google Colab. The notebook installs `scikit-learn` automatically in Cell 1.

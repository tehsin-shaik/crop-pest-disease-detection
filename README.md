# CropDiseaseNet: Crop Pest and Disease Detection for Maize and Cashew

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)]()
[![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00.svg)]()
[![Status](https://img.shields.io/badge/Model-Custom%20CNN%20from%20Scratch-success.svg)]()

A PyTorch deep learning project for **automated crop pest and disease classification** from plant leaf images using **CropDiseaseNet**, a custom convolutional neural network built entirely from scratch.

This project focuses on **Maize** and **Cashew** disease/pest classification and provides a complete training pipeline with reproducible splits, safe image handling, augmentation, imbalance-aware training, checkpointing, and rich evaluation.

---

## Overview
Early detection of crop pests and diseases is important for improving yield, reducing financial loss, and supporting food security. Manual diagnosis can be slow, inconsistent, and difficult to scale. This project explores how **computer vision** and **deep learning** can support crop-health monitoring from leaf images.

The notebook supports two training modes:
- **Separate crop mode**: train one model for Maize and one for Cashew
- **Combined mode**: train a single model across all Maize and Cashew classes

---

## Dataset

The dataset contains raw crop images for:
### Cashew (5 classes)
- Anthracnose
- Gummosis
- Healthy
- Leaf Miner
- Red Rust

### Maize (7 classes)
- Fall Armyworm
- Grasshopper
- Healthy
- Leaf Beetle
- Leaf Blight
- Leaf Spot
- Streak Virus
 
In **combined mode**, each label is prefixed with the crop name, for example:
- `Cashew___Anthracnose`
- `Maize___Leaf Blight`

### Why raw images instead of pre-augmented data?
Pre-augmented datasets may contain multiple transformed versions of the same original image. If related images leak across train, validation, and test splits, model performance can be artificially inflated.

To avoid this, this project:
- uses **raw images**
- applies **stratified splitting before augmentation**
- performs augmentation **only on the training set**

---

## CropDiseaseNet
The main model in this project is **CropDiseaseNet**, a custom CNN implemented manually in PyTorch and trained **from scratch**, without pretrained weights.

### Architecture highlights
- Conv-BatchNorm-SiLU blocks
- Residual connections
- Depthwise separable convolutions
- Squeeze-and-Excitation channel attention
- Global average pooling
- Dropout classifier head

This architecture was designed to stay efficient enough for Colab GPU training while still using modern CNN design ideas.

---

## Training Pipeline
### Data handling
- Google Colab + Google Drive support
- Safe corrupted-image detection
- Bad images are **skipped and logged**, not deleted
- Reproducible **stratified 70/15/15 train/validation/test split**
- Split CSV files saved for reproducibility
- Random seeds set for Python, NumPy, and PyTorch

### Preprocessing and augmentation
- Configurable image size, default: **224 × 224**
- Normalization statistics computed from the **training split**
- Training augmentations include:
  - RandomResizedCrop
  - RandomHorizontalFlip
  - RandomVerticalFlip
  - RandomRotation
  - ColorJitter
  - RandomAffine
  - RandomPerspective
  - RandomErasing

Validation and test sets use deterministic transforms only.

### Handling Class Imbalance
To handle class imbalance, the pipeline supports:
- **WeightedRandomSampler**, or
- **class-weighted loss**

### Optimization
- Optimizer: **AdamW**
- Loss: **CrossEntropyLoss with label smoothing**
- Scheduler: **OneCycleLR**
- Mixed precision training with `torch.cuda.amp`
- Gradient clipping
- Early stopping based on **validation macro F1**
- Best checkpoint saved automatically

---

## Evaluation

The model is evaluated using:

- Accuracy
- Macro precision / recall / F1
- Weighted precision / recall / F1
- Per-class precision / recall / F1
- Confusion matrix
- Normalized confusion matrix
- ROC-AUC (One-vs-Rest, multiclass)
- Top-2 / Top-3 accuracy
- Misclassified image analysis

---

## Results

### Combined Maize + Cashew Classification
**Custom CNN from scratch: CropDiseaseNet**

| Metric | Score |
|--------|-------|
| Test Accuracy | **91.0%** |
| Macro Precision | **0.902** |
| Macro Recall | **0.915** |
| Macro F1-score | **0.908** |
| Weighted F1-score | **0.910** |
| Macro ROC-AUC (OvR) | **0.991** |
| Top-2 Accuracy | **98.3%** |
| Top-3 Accuracy | **99.4%** |

### Cashew-Only Classification

| Metric | Score |
|--------|-------|
| Test Accuracy | **94.9%** |
| Macro Precision | **0.953** |
| Macro Recall | **0.949** |
| Macro F1-score | **0.951** |
| Weighted F1-score | **0.949** |
| Macro ROC-AUC (OvR) | **0.991** |
| Top-2 Accuracy | **98.8%** |
| Top-3 Accuracy | **99.6%** |

### Maize-Only Classification

| Metric | Score |
|--------|-------|
| Test Accuracy | **84.1%** |
| Macro F1-score | **0.851** |
| Macro ROC-AUC (OvR) | **0.976** |

---

## Key Findings

- The custom CNN performed strongly on both separate and combined crop-classification tasks
- The **Cashew-only model** achieved the strongest single-crop result with **94.9% accuracy** and **0.951 macro-F1**
- The **combined 12-class model** achieved **91.0% accuracy** and showed strong multi-crop generalization
- The **Maize-only task** was more challenging due to higher visual similarity between certain disease classes
- The hardest remaining maize distinction was between **Leaf Blight** and **Leaf Spot**
- Very high **Top-2** and **Top-3** accuracy suggests the model often ranks the correct class among its top predictions even when Top-1 is wrong

---

## Optional Baselines
For comparison, the notebook also supports optional pretrained baselines:
- ResNet18
- EfficientNet-B0
- VGG16

These baselines are included only for benchmarking. The main final model of this project is **CropDiseaseNet**.

---

## Tech Stack

- Python
- PyTorch
- torchvision
- scikit-learn
- NumPy
- pandas
- matplotlib
- Google Colab
- Jupyter Notebook

---
## How to Run

1. Upload the dataset to Google Drive  
   Dataset source: https://data.mendeley.com/datasets/bwh3zbpkpv/1
2. Open the notebook in Google Colab
3. Mount Google Drive
4. Set the dataset path in the configuration section
5. Run the notebook cells to:
   - scan and validate the dataset
   - create reproducible splits
   - train `CropDiseaseNet`
   - evaluate results
   - run inference on new images

---


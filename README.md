# Crop Pest and Disease Detection with CropDiseaseNet (Maize & Cashew)
A PyTorch deep learning project for maize and cashew pest/disease classification from plant leaf images using a custom CNN built from scratch.

## Overview
This project focuses on multi-class crop pest and disease classification using a manually designed convolutional neural network called CropDiseaseNet.

- Separate crop mode: train one model for Maize and one for Cashew
- Combined mode: train a single model across all Maize and Cashew classes together

Early detection of crop diseases is important for improving yield, reducing financial loss, and supporting food security. Traditional diagnosis often depends on manual inspection, which can be slow, inconsistent, and difficult to scale.

This project uses **computer vision** and **deep learning** to classify crop diseases from images of maize and cashew plants.

## Dataset

The project uses raw crop image datasets for:

- **Cashew (5 classes):**
  - Anthracnose
  - Gumosis
  - Healthy
  - Leaf Miner
  - Red Rust

- **Maize (7 classes):**
  - Fall Armyworm
  - Grasshopper
  - Healthy
  - Leaf Beetle
  - Leaf Blight
  - Leaf Spot
  - Streak Virus
 
In combined mode, labels are prefixed with crop name, for example:
- Cashew___Anthracnose
- Maize___Leaf Blight

### Why raw images instead of pre-augmented data?
Pre-augmented datasets can contain multiple transformed versions of the same original image. If similar images appear in both training and test splits, this can cause data leakage and inflate performance.

To avoid this, the project uses raw images and applies augmentation only after splitting the dataset.

## Pipeline

### Data Handling
- Google Drive / Colab-compatible setup
- Safe corrupted-image detection
- Bad images are skipped and logged, not automatically deleted
- Reproducible stratified 70/15/15 train/validation/test split
- Split files saved for reproducibility

### Preprocessing and Augmentation
- Resized all images to **224 × 224**
- Training-set normalization statistics computed from the dataset
- Training augmentations include: RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomAffine, RandomPerspective
RandomErasing

Validation and test images use deterministic transforms only.

### Handling Class Imbalance
The training pipeline supports imbalance-aware learning using:
- WeightedRandomSampler, or
- class-weighted loss

## Models

This project compares a custom CNN with transfer learning models:

- **Custom CNN**
- **ResNet18**
- **EfficientNet-B0**
- **VGG16**

### Custom CNN Architecture
The custom model contains:
- 4 convolution blocks
- Conv2D + BatchNorm + ReLU + MaxPool in each block
- Channel depth progression: **32 → 64 → 128 → 256**
- Adaptive Global Average Pooling
- Fully connected classifier with Dropout

## Training Setup

- **Image size:** 224
- **Batch size:** 32
- **Epochs:** 20
- **Optimizer:** Adam / AdamW
- **Learning rate:**  
  - Custom CNN: 0.001  
  - Pretrained models: 0.0005
- **Loss:** Weighted CrossEntropyLoss
- **Early stopping:** Used based on validation performance

## Evaluation Metrics

Performance was evaluated using:
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- ROC Curve / AUC

Macro F1-score was especially important because the datasets were imbalanced.

## Results

### Best Model
**EfficientNet-B0** achieved the best overall performance:

- **Cashew Accuracy:** 95.73%
- **Maize Accuracy:** 85.89%

### Custom CNN Performance
- **Cashew Accuracy:** 90.03%
- **Maize Accuracy:** 70.91%

## Key Findings
- The custom CNN trained from scratch performed strongly on both separate and combined tasks
- The combined 12-class model achieved the best overall result



- Transfer learning outperformed the custom CNN overall.
- **EfficientNet-B0** was the strongest model on both crops.
- Cashew classification was easier than maize classification.
- Maize classes were more visually similar, leading to more confusion between diseases.
- The custom CNN still performed competitively on cashew, showing that a lightweight model can learn strong crop-specific features.

## Tech Stack
Python, PyTorch, torchvision, scikit-learn, NumPy, Matplotlib, Google Colab

## How to Run
1. Upload the dataset from https://data.mendeley.com/datasets/bwh3zbpkpv/1 to Google Drive
2. Open the notebook in Google Colab
3. Mount Google Drive
4. Set the dataset path in the configuration section
5. Run the notebook cells to train and evaluate the model


# Crop Pest and Disease Detection (Maize & Cashew)
A deep learning project for automated crop pest and disease classification from plant images, focused on **maize** and **cashew** crops.

## Overview

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

### Why raw images instead of pre-augmented data?
Pre-augmented datasets may contain multiple transformed versions of the same original image. This increases the risk of **data leakage** if similar images appear across training, validation, and test sets, which can artificially inflate performance.

## Approach

### Preprocessing
- Resized all images to **224 × 224**
- Applied **ImageNet normalization**
- Removed corrupted image files if present

### Data Augmentation
Applied only to the training set:
- Random horizontal flip
- Random rotation (10°)
- Color jitter

### Handling Class Imbalance
Used **class-weighted cross-entropy loss** so that underrepresented classes receive higher importance during training.

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

- Transfer learning outperformed the custom CNN overall.
- **EfficientNet-B0** was the strongest model on both crops.
- Cashew classification was easier than maize classification.
- Maize classes were more visually similar, leading to more confusion between diseases.
- The custom CNN still performed competitively on cashew, showing that a lightweight model can learn strong crop-specific features.

## Tech Stack
Python, PyTorch, Torchvision, scikit-learn, NumPy, Matplotlib

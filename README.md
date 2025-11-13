# IE4483 - Image Classification Project

This project implements image classification using ResNet18 for two different datasets:
1. **Cats vs Dogs** (Binary Classification)
2. **CIFAR-10** (Multi-class Classification with 10 classes)

## Project Overview

The project uses transfer learning with a pretrained ResNet18 model to classify images. The model architecture includes a custom classifier head that adapts the pretrained features to the specific classification task.

## Features

- **Cats vs Dogs Classification**: Binary classification (2 classes)
- **CIFAR-10 Classification**: Multi-class classification (10 classes)
- Transfer learning with pretrained ResNet18
- Custom 3-layer MLP classifier head
- Data augmentation for improved generalization
- Early stopping to prevent overfitting
- Model checkpointing and training history tracking

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Setup

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Cats vs Dogs Dataset

The dataset should be organized in the following structure:
```
datasets/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
```

- **Training set**: 20,000 images (10,000 cat + 10,000 dog)
- **Validation set**: 5,000 images (2,500 cat + 2,500 dog)
- **Test set**: 500 images

### CIFAR-10 Dataset

The CIFAR-10 dataset will be automatically downloaded when you run the training script. No manual setup required.

## Usage

### Training Models

#### 1. Train Cats vs Dogs Model

```bash
python train_model.py
```

This will:
- Load and preprocess the cats vs dogs dataset
- Train a ResNet18 model with transfer learning
- Save the best model as `best_cats_dogs_model.pth`
- Generate training history plot as `training_history.png`

#### 2. Train CIFAR-10 Model

First, download the CIFAR-10 dataset:
```bash
python download_cifar10.py
```

Then train the model:
```bash
python train_cifar10.py
```

This will:
- Load CIFAR-10 dataset (20,000 train / 5,000 val / 500 test)
- Train a ResNet18 model adapted for 10 classes
- Save the best model as `best_cifar10_model.pth`
- Generate training history plot as `cifar10_training_history.png`

### Making Predictions

#### 1. Predict on Cats vs Dogs Test Set

```bash
python predict_test.py
```

This generates `submission.csv` with predictions for the test set.

#### 2. Predict on CIFAR-10 Test Set

```bash
python predict_cifar10.py
```

This generates `cifar10_predictions.csv` with predictions and accuracy metrics.

## File Structure

```
IE4483/
├── train_model.py              # Main training script for cats vs dogs
├── train_cifar10.py            # Training script for CIFAR-10
├── predict_test.py             # Prediction script for cats vs dogs
├── predict_cifar10.py          # Prediction script for CIFAR-10
├── download_cifar10.py         # Script to download CIFAR-10 dataset
├── resnet18_model.py           # ResNet18 model definition
├── model_training.py            # Training pipeline and utilities
├── model_prediction.py         # Prediction utilities
├── data_loader.py              # Data loading and preprocessing
├── Data_Preprocessing.py       # Data preprocessing utilities
├── requirements.txt            # Python dependencies
├── best_cats_dogs_model.pth    # Trained cats vs dogs model
├── best_cifar10_model.pth      # Trained CIFAR-10 model
├── submission.csv              # Predictions for cats vs dogs test set
├── cifar10_predictions.csv      # Predictions for CIFAR-10 test set
└── README.md                   # This file
```

## Model Architecture

### ResNet18 Backbone
- Pretrained on ImageNet
- Feature extraction layers (frozen or fine-tuned)
- Output: 512-dimensional feature vector

### Custom Classifier Head
- **Layer 1**: 512 → 256 (ReLU + Dropout)
- **Layer 2**: 256 → 128 (ReLU + Dropout)
- **Layer 3**: 128 → num_classes (Output layer)
- **Dropout rate**: 0.5

### For Cats vs Dogs
- Output: 2 classes (cat, dog)

### For CIFAR-10
- Output: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## Training Configuration

### Default Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Weight Decay**: 1e-4
- **Batch Size**: 32
- **Epochs**: 5 (quick training) or more for full training
- **Early Stopping**: Patience of 2-3 epochs
- **Learning Rate Scheduler**: StepLR (reduces LR by 0.1)

## Results

### Cats vs Dogs Classification
- **Best Validation Accuracy**: 96.18%
- **Final Training Accuracy**: 97.06%
- **Final Validation Accuracy**: 95.60%

### CIFAR-10 Classification
- **Best Validation Accuracy**: 79.44%
- **Final Training Accuracy**: 77.53%
- **Final Validation Accuracy**: 79.44%

## Notes

- Images are resized to 224×224 pixels for ResNet18 compatibility
- CIFAR-10 images (32×32) are upscaled to 224×224
- Data augmentation includes random horizontal flips
- Models use transfer learning from ImageNet-pretrained weights
- Training history is saved in model checkpoint files

## Requirements

See `requirements.txt` for full list of dependencies. Main packages include:
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- PIL/Pillow


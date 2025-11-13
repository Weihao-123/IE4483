"""
Training Script for CIFAR-10 Classification

This script trains a ResNet18 model on CIFAR-10 dataset for comparison
with the cats vs dogs classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from resnet18_model import ResNet18Classifier
from model_training import ModelTrainer
import time
import numpy as np

def get_cifar10_data_loaders(batch_size=32, num_workers=2, 
                              train_size=20000, val_size=5000, test_size=500):
    """
    Create data loaders for CIFAR-10 dataset (same split as cats vs dogs)
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        train_size (int): Number of training images (20,000)
        val_size (int): Number of validation images (5,000)
        test_size (int): Number of test images (500)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    print("Loading CIFAR-10 dataset (same split as cats vs dogs)...")
    print(f"  Training: {train_size} images")
    print(f"  Validation: {val_size} images")
    print(f"  Test: {test_size} images")
    
    # Simplified transforms for faster processing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Upscale from 32x32 to 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=False,
        transform=train_transform
    )
    
    # Load full test dataset (we'll split this into val and test)
    full_testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=False,
        transform=val_test_transform
    )
    
    # Create training subset (20,000 images)
    if train_size < len(full_trainset):
        print(f"Creating training subset: {train_size} images from {len(full_trainset)} total")
        np.random.seed(42)  # For reproducibility
        train_indices = np.random.choice(len(full_trainset), train_size, replace=False)
        trainset = Subset(full_trainset, train_indices)
    else:
        trainset = full_trainset
    
    # Split test set into validation (5,000) and test (500)
    total_val_test = val_size + test_size
    if total_val_test <= len(full_testset):
        print(f"Splitting test set: {val_size} validation + {test_size} test from {len(full_testset)} total")
        np.random.seed(42)  # For reproducibility
        val_test_indices = np.random.choice(len(full_testset), total_val_test, replace=False)
        
        # Split into validation and test
        val_indices = val_test_indices[:val_size]
        test_indices = val_test_indices[val_size:val_size+test_size]
        
        valset = Subset(full_testset, val_indices)
        testset = Subset(full_testset, test_indices)
    else:
        # If not enough images, use what we have
        valset = full_testset
        testset = full_testset
    
    # Create data loaders
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Get class names
    class_names = full_trainset.classes
    
    print(f"\nDataset Summary:")
    print(f"  Training set: {len(trainset)} images")
    print(f"  Validation set: {len(valset)} images")
    print(f"  Test set: {len(testset)} images")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Classes: {class_names}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, class_names

def create_cifar10_model(pretrained=True, freeze_backbone=False):
    """
    Create ResNet18 model for CIFAR-10 (10 classes)
    
    Args:
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone
    
    Returns:
        ResNet18Classifier: Model instance
    """
    # Create model with 10 classes for CIFAR-10
    # Images are upscaled to 224x224 in data transforms to use pretrained weights
    model = ResNet18Classifier(
        num_classes=10,  # CIFAR-10 has 10 classes
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=0.5
    )
    
    return model

def main():
    """Main training function for CIFAR-10 (Quick Training)"""
    print("=" * 80)
    print("CIFAR-10 CLASSIFICATION - QUICK TRAINING (5 EPOCHS)")
    print("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders (same split as cats vs dogs)
    print("\n1. Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader, class_names = get_cifar10_data_loaders(
        batch_size=32,  # Smaller batch size for CPU training
        num_workers=2,  # Fewer workers for CPU
        train_size=20000,  # Same as cats vs dogs
        val_size=5000,      # Same as cats vs dogs
        test_size=500       # Same as cats vs dogs
    )
    
    # Create model
    print("\n2. Creating ResNet18 model...")
    model = create_cifar10_model(
        pretrained=True,
        freeze_backbone=False
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"Number of Classes: {model_info['num_classes']}")
    
    # Create trainer
    print("\n3. Setting up trainer...")
    trainer = ModelTrainer(model, device=device)
    
    # Train model
    print("\n4. Training model...")
    print("Using same dataset split as cats vs dogs for fair comparison...")
    
    start_time = time.time()
    history = trainer.train_model(
        train_loader, val_loader,  # Use validation set, not test set
        num_epochs=5,  # Quick training (5 epochs)
        learning_rate=0.001,
        optimizer_type='adam',
        weight_decay=1e-4,
        scheduler_step=5,  # Reduce LR every 5 epochs
        early_stopping_patience=2,  # Quick early stopping for testing
        min_accuracy_threshold=50.0
    )
    
    training_time = time.time() - start_time
    
    # Save model
    print("\n5. Saving model...")
    trainer.save_model("best_cifar10_model.pth")
    
    # Plot training history
    print("\n6. Plotting training history...")
    trainer.plot_training_history("cifar10_training_history.png")
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    trainer.model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = trainer.model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_accuracy = 100.0 * test_correct / test_total
    print(f"Test set accuracy: {test_accuracy:.2f}% ({test_correct}/{test_total})")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Best model saved as: best_cifar10_model.pth")
    print(f"Training history plot saved as: cifar10_training_history.png")
    print(f"Best validation accuracy: {trainer.best_accuracy:.2f}%")
    print(f"Test set accuracy: {test_accuracy:.2f}%")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Number of epochs: {history.get('epochs_completed', 'N/A')}")
    print("\nNOTE: Dataset split matches cats vs dogs:")
    print("      - Training: 20,000 images")
    print("      - Validation: 5,000 images (used during training)")
    print("      - Test: 500 images (for final predictions)")
    print("=" * 80)
    
    # Print comparison info
    print("\n" + "=" * 80)
    print("COMPARISON INFORMATION")
    print("=" * 80)
    print("Dataset: CIFAR-10")
    print(f"  - Training samples: {len(train_loader.dataset):,}")
    print(f"  - Validation samples: {len(val_loader.dataset):,}")
    print(f"  - Test samples: {len(test_loader.dataset):,}")
    print(f"  - Number of classes: 10")
    print(f"  - Image size: 32x32 pixels (upscaled to 224x224)")
    print(f"  - Best validation accuracy: {trainer.best_accuracy:.2f}%")
    print(f"  - Test accuracy: {test_accuracy:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()

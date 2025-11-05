"""
Data Loading for Cats vs Dogs Classification

This module provides optimized data loading with different computational budgets
and comprehensive data augmentation strategies.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image
import random

class CatsDogsDataset(Dataset):
    """
    Custom dataset for cats vs dogs classification
    """
    
    def __init__(self, data_dir, transform=None, class_names=None):
        """
        Initialize dataset
        
        Args:
            data_dir (str): Path to dataset directory
            transform: Image transforms to apply
            class_names (list): List of class names
        """
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = class_names or ['cat', 'dog']
        
        # Get all image files
        self.image_files = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_files.append(os.path.join(class_dir, filename))
                        self.labels.append(class_idx)
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        print(f"Class distribution: {dict(zip(self.class_names, [self.labels.count(i) for i in range(len(self.class_names))]))}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get image and label"""
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms(computational_budget='medium', image_size=224):
    """
    Get data transforms based on computational budget
    
    Args:
        computational_budget (str): 'low', 'medium', or 'high'
        image_size (int): Target image size
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    if computational_budget == 'low':
        # Minimal transforms for quick training
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif computational_budget == 'medium':
        # Balanced transforms for good performance
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif computational_budget == 'high':
        # Comprehensive transforms for maximum performance
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 64, image_size + 64)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:
        raise ValueError(f"Unknown computational budget: {computational_budget}")
    
    return train_transform, val_transform

def create_optimized_data_loaders(train_dir, val_dir, computational_budget='medium', 
                                batch_size=None, num_workers=None, image_size=224):
    """
    Create optimized data loaders based on computational budget
    
    Args:
        train_dir (str): Path to training directory
        val_dir (str): Path to validation directory
        computational_budget (str): 'low', 'medium', or 'high'
        batch_size (int): Batch size (auto-determined if None)
        num_workers (int): Number of workers (auto-determined if None)
        image_size (int): Target image size
    
    Returns:
        tuple: (train_loader, val_loader, class_names, config)
    """
    
    # Auto-determine batch size based on computational budget
    if batch_size is None:
        if computational_budget == 'low':
            batch_size = 64
        elif computational_budget == 'medium':
            batch_size = 32
        else:  # high
            batch_size = 16
    
    # Auto-determine number of workers
    if num_workers is None:
        num_workers = min(4, os.cpu_count())
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(computational_budget, image_size)
    
    # Create datasets
    train_dataset = CatsDogsDataset(train_dir, transform=train_transform)
    val_dataset = CatsDogsDataset(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Configuration info
    config = {
        'computational_budget': computational_budget,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'image_size': image_size,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    print(f"\nData Loader Configuration:")
    print(f"  Computational Budget: {computational_budget}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {image_size}x{image_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset.class_names, config

def create_quick_data_loaders(train_dir, val_dir, batch_size=32):
    """
    Create quick data loaders for testing
    
    Args:
        train_dir (str): Path to training directory
        val_dir (str): Path to validation directory
        batch_size (int): Batch size
    
    Returns:
        tuple: (train_loader, val_loader, class_names, config)
    """
    return create_optimized_data_loaders(
        train_dir, val_dir, 
        computational_budget='low',
        batch_size=batch_size,
        num_workers=2,
        image_size=224
    )

def test_data_loader():
    """Test data loader functionality"""
    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    # Test different computational budgets
    budgets = ['low', 'medium', 'high']
    
    for budget in budgets:
        print(f"\nTesting {budget.upper()} budget:")
        print("-" * 30)
        
        try:
            # Create data loaders
            train_loader, val_loader, class_names, config = create_optimized_data_loaders(
                train_dir="datasets/train",
                val_dir="datasets/val",
                computational_budget=budget
            )
            
            # Test loading a batch
            for batch_idx, (data, target) in enumerate(train_loader):
                print(f"  Batch shape: {data.shape}")
                print(f"  Target shape: {target.shape}")
                print(f"  Data type: {data.dtype}")
                print(f"  Target type: {target.dtype}")
                print(f"  ✓ Data loader works correctly")
                break
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    # Test the data loader
    test_data_loader()
    
    print("\n" + "=" * 60)
    print("DATA LOADER USAGE:")
    print("=" * 60)
    print("# Create data loaders")
    print("train_loader, val_loader, class_names, config = create_optimized_data_loaders(")
    print("    train_dir='datasets/train',")
    print("    val_dir='datasets/val',")
    print("    computational_budget='medium'")
    print(")")
    print()
    print("# Quick data loaders for testing")
    print("train_loader, val_loader, class_names, config = create_quick_data_loaders(")
    print("    train_dir='datasets/train',")
    print("    val_dir='datasets/val'")
    print(")")






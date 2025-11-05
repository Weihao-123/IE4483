"""
Data Preprocessing Configuration for Cats vs Dogs Classification

This module provides comprehensive preprocessing configurations optimized for different
computational budgets and training scenarios.
"""

import torch
from torchvision import transforms

class PreprocessingConfig:
    """
    Configuration class for data preprocessing with different computational budgets
    """
    
    def __init__(self, computational_budget='medium'):
        """
        Initialize preprocessing configuration
        
        Args:
            computational_budget (str): 'low', 'medium', 'high'
        """
        self.budget = computational_budget
        self.config = self._get_config()
    
    def _get_config(self):
        """Get configuration based on computational budget"""
        configs = {
            'low': {
                'image_size': 128,
                'batch_size': 16,
                'augmentation_level': 'light',
                'train_subset': 1000,
                'val_subset': 250,
                'num_workers': 2,
                'description': 'Optimized for limited computational resources'
            },
            'medium': {
                'image_size': 224,
                'batch_size': 32,
                'augmentation_level': 'medium',
                'train_subset': 2000,
                'val_subset': 500,
                'num_workers': 4,
                'description': 'Balanced performance and resource usage'
            },
            'high': {
                'image_size': 224,
                'batch_size': 64,
                'augmentation_level': 'heavy',
                'train_subset': None,  # Use all data
                'val_subset': None,
                'num_workers': 8,
                'description': 'High-performance training with full dataset'
            }
        }
        
        return configs.get(self.budget, configs['medium'])
    
    def get_augmentation_transforms(self):
        """
        Get augmentation transforms based on the configuration
        
        Returns:
            tuple: (train_transforms, val_transforms)
        """
        image_size = self.config['image_size']
        augmentation_level = self.config['augmentation_level']
        
        # Base normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # Validation transforms (no augmentation)
        val_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
        
        # Training transforms with augmentation
        if augmentation_level == 'none':
            train_transforms = val_transforms
        elif augmentation_level == 'light':
            train_transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                normalize
            ])
        elif augmentation_level == 'medium':
            train_transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize
            ])
        elif augmentation_level == 'heavy':
            train_transforms = transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                normalize
            ])
        else:
            raise ValueError(f"Unknown augmentation_level: {augmentation_level}")
        
        return train_transforms, val_transforms
    
    def get_memory_estimate(self):
        """
        Estimate memory usage for the configuration
        
        Returns:
            dict: Memory estimates in MB
        """
        image_size = self.config['image_size']
        batch_size = self.config['batch_size']
        
        # Estimate memory per image (3 channels, float32)
        bytes_per_pixel = 4  # float32
        channels = 3
        pixels_per_image = image_size * image_size
        bytes_per_image = pixels_per_image * channels * bytes_per_pixel
        
        # Memory estimates
        memory_per_batch = (bytes_per_image * batch_size) / (1024 * 1024)  # MB
        memory_per_epoch = memory_per_batch * 100  # Rough estimate for one epoch
        
        return {
            'memory_per_batch_mb': round(memory_per_batch, 2),
            'memory_per_epoch_mb': round(memory_per_epoch, 2),
            'recommended_gpu_memory_gb': max(2, round(memory_per_batch / 1000, 1))
        }
    
    def print_config(self):
        """Print the current configuration"""
        print("=" * 60)
        print(f"PREPROCESSING CONFIGURATION: {self.budget.upper()}")
        print("=" * 60)
        print(f"Description: {self.config['description']}")
        print(f"Image size: {self.config['image_size']}x{self.config['image_size']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Augmentation level: {self.config['augmentation_level']}")
        print(f"Train subset: {self.config['train_subset'] or 'All data'}")
        print(f"Validation subset: {self.config['val_subset'] or 'All data'}")
        print(f"Number of workers: {self.config['num_workers']}")
        
        # Memory estimates
        memory = self.get_memory_estimate()
        print(f"\nMemory estimates:")
        print(f"  Memory per batch: {memory['memory_per_batch_mb']} MB")
        print(f"  Memory per epoch: {memory['memory_per_epoch_mb']} MB")
        print(f"  Recommended GPU memory: {memory['recommended_gpu_memory_gb']} GB")
        print("=" * 60)

def compare_configurations():
    """Compare different preprocessing configurations"""
    print("PREPROCESSING CONFIGURATION COMPARISON")
    print("=" * 80)
    
    for budget in ['low', 'medium', 'high']:
        config = PreprocessingConfig(budget)
        config.print_config()
        print()

def get_recommended_config(available_memory_gb=4, training_time_hours=2):
    """
    Get recommended configuration based on available resources
    
    Args:
        available_memory_gb (float): Available GPU memory in GB
        training_time_hours (float): Available training time in hours
    
    Returns:
        str: Recommended computational budget
    """
    if available_memory_gb < 4:
        return 'low'
    elif available_memory_gb < 8:
        return 'medium'
    else:
        return 'high'

if __name__ == "__main__":
    # Compare all configurations
    compare_configurations()
    
    # Show recommendations
    print("RECOMMENDATIONS:")
    print("=" * 40)
    print("For 2GB GPU memory: low")
    print("For 4-6GB GPU memory: medium") 
    print("For 8GB+ GPU memory: high")
    print("=" * 40)
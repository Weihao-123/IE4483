"""
Model Training for Cats vs Dogs Classification

This module implements comprehensive training pipeline with parameter testing,
validation, and model optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from resnet18_model import create_resnet18_model, get_training_config

class ModelTrainer:
    """
    Comprehensive model trainer with parameter testing and validation
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.training_history = defaultdict(list)
        self.best_accuracy = 0.0
        self.best_model_state = None
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """
        Train model for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
        
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate_epoch(self, val_loader, criterion):
        """
        Validate model for one epoch
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            dict: Validation metrics for the epoch
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001, 
                   optimizer_type='adam', weight_decay=1e-4, scheduler_step=5, 
                   early_stopping_patience=3, min_accuracy_threshold=60.0):
        """
        Train model with comprehensive parameter testing and early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            weight_decay: Weight decay for regularization
            scheduler_step: Step size for learning rate scheduler
            early_stopping_patience: Number of epochs to wait before early stopping
            min_accuracy_threshold: Minimum accuracy threshold to continue training
        
        Returns:
            dict: Training history and best model info
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Setup optimizer
        optimizer = self._get_optimizer(optimizer_type, learning_rate, weight_decay)
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)
        
        # Early stopping variables
        best_val_acc = 0.0
        epochs_without_improvement = 0
        early_stopped = False
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Record metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
            
            # Early stopping logic
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Check for early stopping conditions
            if val_metrics['accuracy'] < min_accuracy_threshold:
                print(f"\n⚠️  EARLY STOPPING: Validation accuracy ({val_metrics['accuracy']:.2f}%) "
                      f"below threshold ({min_accuracy_threshold}%)")
                early_stopped = True
                break
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️  EARLY STOPPING: No improvement for {early_stopping_patience} epochs")
                early_stopped = True
                break
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
                  f"Time: {epoch_time:.2f}s, "
                  f"No improvement: {epochs_without_improvement}/{early_stopping_patience}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        
        if early_stopped:
            print(f"Training was stopped early due to poor performance or lack of improvement")
        else:
            print(f"Training completed all {num_epochs} epochs")
        
        return {
            'training_history': dict(self.training_history),
            'best_accuracy': self.best_accuracy,
            'total_time': total_time,
            'early_stopped': early_stopped,
            'epochs_completed': epoch + 1
        }
    
    def _get_optimizer(self, optimizer_type, learning_rate, weight_decay):
        """Get optimizer based on type"""
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def test_parameters(self, train_loader, val_loader, parameter_configs):
        """
        Test different parameter configurations
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            parameter_configs: List of parameter configurations to test
        
        Returns:
            dict: Results for each configuration
        """
        results = {}
        
        for i, config in enumerate(parameter_configs):
            print(f"\n{'='*60}")
            print(f"TESTING CONFIGURATION {i+1}/{len(parameter_configs)}")
            print(f"{'='*60}")
            print(f"Learning Rate: {config['learning_rate']}")
            print(f"Optimizer: {config['optimizer']}")
            print(f"Weight Decay: {config['weight_decay']}")
            print(f"Epochs: {config['epochs']}")
            
            # Create new model for each test
            model = create_resnet18_model(
                pretrained=config.get('pretrained', True),
                freeze_backbone=config.get('freeze_backbone', False),
                dropout_rate=config.get('dropout_rate', 0.5)
            )
            
            # Initialize trainer
            trainer = ModelTrainer(model, self.device)
            
            # Train model
            start_time = time.time()
            history = trainer.train_model(
                train_loader, val_loader,
                num_epochs=config['epochs'],
                learning_rate=config['learning_rate'],
                optimizer_type=config['optimizer'],
                weight_decay=config['weight_decay'],
                early_stopping_patience=config.get('early_stopping_patience', 3),
                min_accuracy_threshold=config.get('min_accuracy_threshold', 60.0)
            )
            
            # Record results
            results[f"config_{i+1}"] = {
                'parameters': config,
                'best_accuracy': trainer.best_accuracy,
                'training_time': time.time() - start_time,
                'final_train_acc': history['training_history']['train_acc'][-1],
                'final_val_acc': history['training_history']['val_acc'][-1],
                'early_stopped': history.get('early_stopped', False),
                'epochs_completed': history.get('epochs_completed', config['epochs'])
            }
            
            print(f"Best Accuracy: {trainer.best_accuracy:.2f}%")
            print(f"Training Time: {time.time() - start_time:.2f}s")
        
        return results
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not self.training_history:
            print("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['train_acc'], label='Train Accuracy')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """Save model and training info"""
        torch.save({
            'model_state_dict': self.best_model_state,
            'model_info': self.model.get_model_info(),
            'best_accuracy': self.best_accuracy,
            'training_history': dict(self.training_history)
        }, filepath)
        print(f"Model saved to {filepath}")

def get_parameter_test_configs():
    """
    Get different parameter configurations for testing with improved settings
    
    Returns:
        list: List of parameter configurations
    """
    return [
        # Configuration 1: Conservative transfer learning (recommended)
        {
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'epochs': 10,
            'pretrained': True,
            'freeze_backbone': False,
            'dropout_rate': 0.5,
            'early_stopping_patience': 4,
            'min_accuracy_threshold': 65.0,
            'description': 'Conservative transfer learning (recommended)'
        },
        
        # Configuration 2: Fine-tuning with lower learning rate
        {
            'learning_rate': 0.0005,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'epochs': 12,
            'pretrained': True,
            'freeze_backbone': True,
            'dropout_rate': 0.3,
            'early_stopping_patience': 3,
            'min_accuracy_threshold': 60.0,
            'description': 'Fine-tuning with frozen backbone'
        },
        
        # Configuration 3: Moderate learning rate (fixed from problematic 0.01)
        {
            'learning_rate': 0.003,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'epochs': 12,
            'pretrained': True,
            'freeze_backbone': False,
            'dropout_rate': 0.4,
            'early_stopping_patience': 3,
            'min_accuracy_threshold': 65.0,
            'description': 'Moderate learning rate (improved)'
        },
        
        # Configuration 4: SGD optimizer with momentum
        {
            'learning_rate': 0.001,
            'optimizer': 'sgd',
            'weight_decay': 1e-4,
            'epochs': 10,
            'pretrained': True,
            'freeze_backbone': False,
            'dropout_rate': 0.5,
            'early_stopping_patience': 4,
            'min_accuracy_threshold': 65.0,
            'description': 'SGD optimizer with momentum'
        },
        
        # Configuration 5: AdamW optimizer (modern choice)
        {
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'weight_decay': 1e-3,
            'epochs': 10,
            'pretrained': True,
            'freeze_backbone': False,
            'dropout_rate': 0.5,
            'early_stopping_patience': 4,
            'min_accuracy_threshold': 65.0,
            'description': 'AdamW optimizer (modern choice)'
        }
    ]

def compare_parameter_results(results):
    """Compare results from different parameter configurations"""
    print("\n" + "="*80)
    print("PARAMETER TESTING RESULTS COMPARISON")
    print("="*80)
    
    # Sort by best accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Config':<12} {'Best Acc':<10} {'Time':<8} {'Epochs':<7} {'Early Stop':<12} {'Description'}")
    print("-" * 100)
    
    for rank, (config_name, result) in enumerate(sorted_results, 1):
        params = result['parameters']
        early_stop_status = "Yes" if result['early_stopped'] else "No"
        print(f"{rank:<4} {config_name:<12} {result['best_accuracy']:<10.2f}% "
              f"{result['training_time']:<8.1f}s {result['epochs_completed']:<7} "
              f"{early_stop_status:<12} {params['description']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    best_config = sorted_results[0][1]
    print(f"🥇 BEST: {best_config['parameters']['description']}")
    print(f"   Accuracy: {best_config['best_accuracy']:.2f}%")
    print(f"   Time: {best_config['training_time']:.1f}s")
    
    if len(sorted_results) > 1:
        second_best = sorted_results[1][1]
        print(f"🥈 SECOND: {second_best['parameters']['description']}")
        print(f"   Accuracy: {second_best['best_accuracy']:.2f}%")
        print(f"   Time: {second_best['training_time']:.1f}s")

if __name__ == "__main__":
    print("MODEL TRAINING SYSTEM")
    print("="*60)
    print("This module provides comprehensive training with parameter testing.")
    print("Use with your data loaders to train and optimize your ResNet18 model.")
    print("="*60)




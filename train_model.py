#!/usr/bin/env python3
"""
Training Script for Cats vs Dogs Classification

This script demonstrates how to train the ResNet18 model with different
parameter configurations and compare results.
"""

import torch
import os
from data_loader import create_optimized_data_loaders
from resnet18_model import create_resnet18_model
from model_training import ModelTrainer, get_parameter_test_configs, compare_parameter_results

def main():
    """Main training function"""
    print("=" * 80)
    print("CATS VS DOGS CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)
    
    # Check if dataset exists
    import os
    if not os.path.exists("datasets/train"):
        print("Error: Dataset not found!")
        print("Please make sure the dataset is extracted in the 'datasets' folder.")
        return
    
    # Create data loaders
    print("\n1. Loading and preparing data...")
    train_loader, val_loader, class_names, config = create_optimized_data_loaders(
        train_dir="datasets/train",
        val_dir="datasets/val",
        computational_budget='medium'  # Adjust based on your resources
    )
    
    print(f"Training set: {len(train_loader.dataset)} images")
    print(f"Validation set: {len(val_loader.dataset)} images")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Image size: {config['image_size']}x{config['image_size']}")
    
    # Get parameter test configurations
    print("\n2. Setting up parameter testing...")
    parameter_configs = get_parameter_test_configs()
    print(f"Testing {len(parameter_configs)} different parameter configurations:")
    
    for i, config in enumerate(parameter_configs, 1):
        print(f"  {i}. {config['description']}")
    
    # Test different parameters
    print("\n3. Testing different parameter configurations...")
    print("This will take some time depending on your hardware...")
    
    # Create a model for testing (we'll create new ones for each test)
    test_model = create_resnet18_model()
    trainer = ModelTrainer(test_model)
    
    # Run parameter tests
    results = trainer.test_parameters(train_loader, val_loader, parameter_configs)
    
    # Compare results
    print("\n4. Analyzing results...")
    compare_parameter_results(results)
    
    # Train final model with best parameters
    print("\n5. Training final model with best parameters...")
    
    # Find best configuration
    best_config_name = max(results.keys(), key=lambda x: results[x]['best_accuracy'])
    best_config = results[best_config_name]['parameters']
    
    print(f"Using best configuration: {best_config['description']}")
    
    # Create final model
    final_model = create_resnet18_model(
        pretrained=best_config['pretrained'],
        freeze_backbone=best_config['freeze_backbone'],
        dropout_rate=best_config['dropout_rate']
    )
    
    # Train final model
    final_trainer = ModelTrainer(final_model)
    final_history = final_trainer.train_model(
        train_loader, val_loader,
        num_epochs=best_config['epochs'],
        learning_rate=best_config['learning_rate'],
        optimizer_type=best_config['optimizer'],
        weight_decay=best_config['weight_decay']
    )
    
    # Save model
    print("\n6. Saving model...")
    final_trainer.save_model("best_cats_dogs_model.pth")
    
    # Plot training history
    print("\n7. Plotting training history...")
    final_trainer.plot_training_history("training_history.png")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Best model saved as: best_cats_dogs_model.pth")
    print(f"Training history plot saved as: training_history.png")
    print(f"Final accuracy: {final_trainer.best_accuracy:.2f}%")
    print("=" * 80)

def quick_train():
    """Quick training with default parameters (for testing)"""
    print("QUICK TRAINING (Default Parameters)")
    print("=" * 50)
    
    # Create data loaders
    train_loader, val_loader, class_names, config = create_optimized_data_loaders(
        train_dir="datasets/train",
        val_dir="datasets/val",
        computational_budget='low'  # Use low budget for quick training
    )
    
    # Create model
    model = create_resnet18_model()
    trainer = ModelTrainer(model)
    
    # Train with default parameters and early stopping
    history = trainer.train_model(
        train_loader, val_loader,
        num_epochs=5,  # Reduced epochs for quick training
        learning_rate=0.001,
        optimizer_type='adam',
        early_stopping_patience=2,  # Quick early stopping for testing
        min_accuracy_threshold=60.0
    )
    
    print(f"Quick training completed!")
    print(f"Best accuracy: {trainer.best_accuracy:.2f}%")
    
    # Save the model for prediction
    print("Saving model for prediction...")
    try:
        trainer.save_model("best_cats_dogs_model.pth")
        
        # Verify the model was saved
        if os.path.exists("best_cats_dogs_model.pth"):
            print("✓ Model successfully saved!")
        else:
            print("✗ Error: Model file was not created!")
            
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    return trainer

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick training mode
        quick_train()
    else:
        # Full parameter testing mode
        main()

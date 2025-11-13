"""
Prediction Script for CIFAR-10 Classification

This script loads a trained CIFAR-10 model and makes predictions on the test set.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from resnet18_model import ResNet18Classifier
import os

def load_test_data(batch_size=32, num_workers=2, test_size=500):
    """
    Load CIFAR-10 test data (same as training script)
    
    Args:
        batch_size (int): Batch size
        num_workers (int): Number of workers
        test_size (int): Number of test images (500)
    
    Returns:
        tuple: (test_loader, class_names)
    """
    print("Loading CIFAR-10 test data...")
    
    # Same transform as training
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full test dataset
    full_testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=False,
        transform=test_transform
    )
    
    # Get the same test subset as training (for consistency)
    # Use the same seed (42) and same split as training: 5,000 val + 500 test
    np.random.seed(42)
    val_size = 5000
    total_val_test = val_size + test_size
    val_test_indices = np.random.choice(len(full_testset), total_val_test, replace=False)
    test_indices = val_test_indices[val_size:val_size+test_size]  # Last 500 = test set
    
    testset = Subset(full_testset, test_indices)
    
    # Get class names
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    class_names = full_trainset.classes
    
    # Create data loader
    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Test set: {len(testset)} images")
    print(f"Classes: {class_names}")
    
    return test_loader, class_names

def load_model(model_path, device):
    """
    Load trained CIFAR-10 model
    
    Args:
        model_path (str): Path to trained model
        device: Device to load model on
    
    Returns:
        ResNet18Classifier: Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model (10 classes for CIFAR-10)
    model = ResNet18Classifier(
        num_classes=10,
        pretrained=False,  # We're loading trained weights
        freeze_backbone=False,
        dropout_rate=0.5
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    best_accuracy = checkpoint.get('best_accuracy', 'Unknown')
    print(f"Model loaded successfully!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    return model, best_accuracy

def predict_test_set(model, test_loader, class_names, device):
    """
    Make predictions on test set and calculate accuracy
    
    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to use
    
    Returns:
        dict: Prediction results
    """
    print("\nMaking predictions on test set...")
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            # Store predictions
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # Calculate accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {total}/{len(test_loader.dataset)} images...")
    
    accuracy = 100.0 * correct / total
    
    print(f"\nTest Set Results:")
    print(f"  Total images: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def save_predictions_to_csv(results, class_names, output_file='cifar10_predictions.csv'):
    """
    Save predictions to CSV file
    
    Args:
        results: Prediction results dictionary
        class_names: List of class names
        output_file: Output CSV filename
    """
    print(f"\nSaving predictions to {output_file}...")
    
    # Create DataFrame
    df = pd.DataFrame({
        'Image_ID': range(1, len(results['predictions']) + 1),
        'True_Label': [class_names[label] for label in results['labels']],
        'Predicted_Label': [class_names[pred] for pred in results['predictions']],
        'Confidence': [f"{conf:.4f}" for conf in results['confidences']],
        'Correct': [pred == label for pred, label in zip(results['predictions'], results['labels'])]
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print(f"  File: {output_file}")
    
    return df

def print_class_wise_accuracy(results, class_names):
    """
    Print accuracy for each class
    
    Args:
        results: Prediction results dictionary
        class_names: List of class names
    """
    print("\n" + "=" * 80)
    print("CLASS-WISE ACCURACY")
    print("=" * 80)
    
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = labels == class_idx
        if class_mask.sum() > 0:
            class_correct = (predictions[class_mask] == labels[class_mask]).sum()
            class_total = class_mask.sum()
            class_accuracy = 100.0 * class_correct / class_total
            print(f"  {class_name:15s}: {class_accuracy:6.2f}% ({class_correct}/{class_total})")
    
    print("=" * 80)

def main():
    """Main prediction function"""
    print("=" * 80)
    print("CIFAR-10 CLASSIFICATION - TEST SET PREDICTION")
    print("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check for trained model
    model_path = "best_cifar10_model.pth"
    
    if not os.path.exists(model_path):
        print(f"\nError: Trained model not found at {model_path}")
        print("Please train a model first using train_cifar10.py")
        return
    
    # Load model
    print("\n1. Loading trained model...")
    model, best_val_accuracy = load_model(model_path, device)
    
    # Load test data
    print("\n2. Loading test data...")
    test_loader, class_names = load_test_data(
        batch_size=32,
        num_workers=2,
        test_size=500
    )
    
    # Make predictions
    print("\n3. Making predictions...")
    results = predict_test_set(model, test_loader, class_names, device)
    
    # Print class-wise accuracy
    print_class_wise_accuracy(results, class_names)
    
    # Save predictions to CSV
    print("\n4. Saving predictions...")
    df = save_predictions_to_csv(results, class_names)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETED!")
    print("=" * 80)
    print(f"Model validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Test set accuracy: {results['accuracy']:.2f}%")
    print(f"Predictions saved to: cifar10_predictions.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()

"""
Model Prediction for Cats vs Dogs Classification

This module implements prediction system for test set classification
and generates submission.csv file with predictions.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
from resnet18_model import create_resnet18_model

class TestDataset(Dataset):
    """
    Dataset class for test images
    """
    
    def __init__(self, test_dir, transform=None):
        """
        Initialize test dataset
        
        Args:
            test_dir (str): Path to test directory
            transform: Image transforms to apply
        """
        self.test_dir = test_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(glob.glob(os.path.join(test_dir, ext)))
        
        # Sort files for consistent ordering
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} test images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get image and filename"""
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, filename

class ModelPredictor:
    """
    Model predictor for test set classification
    """
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to trained model
            device: Device to use for prediction
        """
        self.device = device
        self.model = None
        self.class_names = ['cat', 'dog']  # 0=cat, 1=dog
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No model loaded. Please train a model first or provide model path.")
    
    def load_model(self, model_path):
        """
        Load trained model
        
        Args:
            model_path (str): Path to saved model
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            self.model = create_resnet18_model()
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded from {model_path}")
            print(f"Model accuracy: {checkpoint.get('best_accuracy', 'Unknown'):.2f}%")
            
            # Store model accuracy for later use
            self.model_accuracy = checkpoint.get('best_accuracy', 'Unknown')
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model for prediction...")
            self.model = create_resnet18_model()
            self.model.to(self.device)
            self.model.eval()
    
    def predict_single_image(self, image_tensor):
        """
        Predict single image
        
        Args:
            image_tensor: Preprocessed image tensor
        
        Returns:
            tuple: (predicted_class, confidence)
        """
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def predict_test_set(self, test_dir, batch_size=32, save_predictions=True):
        """
        Predict on entire test set
        
        Args:
            test_dir (str): Path to test directory
            batch_size (int): Batch size for prediction
            save_predictions (bool): Whether to save predictions to CSV
        
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            print("Error: No model loaded. Cannot make predictions.")
            return None
        
        print(f"Predicting on test set in {test_dir}")
        print(f"Using device: {self.device}")
        
        # Create test dataset
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = TestDataset(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Make predictions
        predictions = []
        filenames = []
        confidences = []
        
        print("Making predictions...")
        for batch_idx, (images, batch_filenames) in enumerate(test_loader):
            images = images.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidences_batch, predicted_batch = torch.max(probabilities, 1)
                
                predictions.extend(predicted_batch.cpu().numpy())
                filenames.extend(batch_filenames)
                confidences.extend(confidences_batch.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Create results
        results = {
            'filenames': filenames,
            'predictions': predictions,
            'confidences': confidences,
            'class_names': self.class_names,
            'model_accuracy': getattr(self, 'model_accuracy', 'Unknown')
        }
        
        # Print prediction summary
        self._print_prediction_summary(results)
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(results)
        
        return results
    
    def _print_prediction_summary(self, results):
        """Print prediction summary"""
        predictions = results['predictions']
        confidences = results['confidences']
        
        # Count predictions
        cat_count = sum(1 for p in predictions if p == 0)
        dog_count = sum(1 for p in predictions if p == 1)
        
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total images: {len(predictions)}")
        print(f"Cat predictions: {cat_count} ({cat_count/len(predictions)*100:.1f}%)")
        print(f"Dog predictions: {dog_count} ({dog_count/len(predictions)*100:.1f}%)")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print(f"Min confidence: {np.min(confidences):.3f}")
        print(f"Max confidence: {np.max(confidences):.3f}")
        print("="*60)
    
    def _save_predictions(self, results):
        """Save predictions to submission.csv"""
        # Create submission dataframe
        submission_data = []
        
        for filename, prediction in zip(results['filenames'], results['predictions']):
            # Extract ID from filename (assuming format like "1.jpg", "2.jpg", etc.)
            file_id = os.path.splitext(filename)[0]
            
            # Convert prediction to label (1=dog, 0=cat)
            label = prediction  # prediction is already 0 or 1
            
            submission_data.append({
                'ID': file_id,
                'Label': label
            })
        
        # Create DataFrame and save
        submission_df = pd.DataFrame(submission_data)
        submission_df = submission_df.sort_values('ID')  # Sort by ID
        submission_df.to_csv('submission.csv', index=False)
        
        print(f"\nPredictions saved to submission.csv")
        print(f"Submission file contains {len(submission_df)} predictions")
        
        # Show sample of predictions
        print("\nSample predictions:")
        print(submission_df.head(10))
        
        return submission_df

def create_sample_submission(test_dir, output_file='submission.csv'):
    """
    Create sample submission file with random predictions
    
    Args:
        test_dir (str): Path to test directory
        output_file (str): Output file path
    """
    # Get test images
    test_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_files.extend(glob.glob(os.path.join(test_dir, ext)))
    
    test_files.sort()
    
    # Create random predictions
    np.random.seed(42)  # For reproducible results
    random_predictions = np.random.randint(0, 2, len(test_files))
    
    # Create submission data
    submission_data = []
    for i, (file_path, prediction) in enumerate(zip(test_files, random_predictions)):
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        
        submission_data.append({
            'ID': file_id,
            'Label': prediction
        })
    
    # Save submission file
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df.sort_values('ID')
    submission_df.to_csv(output_file, index=False)
    
    print(f"Sample submission file created: {output_file}")
    print(f"Contains {len(submission_df)} predictions")
    print("\nSample:")
    print(submission_df.head())
    
    return submission_df

def predict_with_trained_model(model_path, test_dir, batch_size=32):
    """
    Predict using trained model
    
    Args:
        model_path (str): Path to trained model
        test_dir (str): Path to test directory
        batch_size (int): Batch size for prediction
    """
    # Create predictor
    predictor = ModelPredictor(model_path)
    
    # Make predictions
    results = predictor.predict_test_set(test_dir, batch_size=batch_size)
    
    return results

if __name__ == "__main__":
    print("MODEL PREDICTION SYSTEM")
    print("="*60)
    
    # Check if test directory exists
    test_dir = "datasets/test"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        print("Please make sure the dataset is extracted properly.")
        exit(1)
    
    # Check if trained model exists
    model_path = "best_cats_dogs_model.pth"
    if os.path.exists(model_path):
        print(f"Found trained model: {model_path}")
        print("Making predictions with trained model...")
        
        # Predict with trained model
        results = predict_with_trained_model(model_path, test_dir)
        
    else:
        print(f"No trained model found at {model_path}")
        print("Creating sample submission with random predictions...")
        
        # Create sample submission
        create_sample_submission(test_dir)
    
    print("\nPrediction system ready!")
    print("="*60)





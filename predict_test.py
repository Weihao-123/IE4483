#!/usr/bin/env python3
"""
Prediction Script for Cats vs Dogs Classification

This script generates predictions for the test set and creates submission.csv
"""

import os
import sys
import glob
import random

from model_prediction import ModelPredictor, predict_with_trained_model, create_sample_submission

def create_real_predictions(test_dir, model_path, output_file="submission.csv"):
    """Create real predictions using trained model"""
    print(f"Creating real predictions from trained model...")
    
    try:
        # Use the trained model to make predictions
        results = predict_with_trained_model(model_path, test_dir, batch_size=32)
        
        if results:
            print(f"✓ Successfully created predictions using trained model!")
            print(f"✓ Model accuracy: {results.get('model_accuracy', 'Unknown')}%")
            return True
        else:
            print("✗ Failed to create predictions")
            return False
            
    except Exception as e:
        print(f"✗ Error creating predictions: {e}")
        return False

def main():
    """Main prediction function"""
    print("=" * 80)
    print("CATS VS DOGS CLASSIFICATION - TEST SET PREDICTION")
    print("=" * 80)
    
    # Check if test directory exists
    test_dir = "datasets/test"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        print("Please make sure the dataset is extracted properly.")
        return
    
    # Check for trained model
    model_path = "best_cats_dogs_model.pth"
    
    if os.path.exists(model_path):
        print(f"✓ Found trained model: {model_path}")
        print("Creating real predictions using trained model...")
        
        # Create real predictions
        success = create_real_predictions(test_dir, model_path)
        if not success:
            print("Failed to create predictions. Please check model file.")
            return
        
    else:
        print(f"⚠ No trained model found at {model_path}")
        print("Please train a model first to generate predictions.")
        return
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETED!")
    print("=" * 80)
    print("Files created:")
    print("  - submission.csv (predictions for test set)")
    print("=" * 80)

def quick_predict():
    """Quick prediction for testing"""
    print("QUICK PREDICTION TEST")
    print("=" * 40)
    
    test_dir = "datasets/test"
    model_path = "best_cats_dogs_model.pth"
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Please train a model first.")
        return
    
    # Create real predictions
    success = create_real_predictions(test_dir, model_path, "quick_submission.csv")
    if success:
        print("Quick prediction completed!")
    else:
        print("Quick prediction failed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick prediction mode
        quick_predict()
    else:
        # Full prediction mode
        main()





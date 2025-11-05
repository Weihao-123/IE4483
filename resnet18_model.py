"""
ResNet18 Model for Cats vs Dogs Classification

Simplified version focusing only on ResNet18 with transfer learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18Classifier(nn.Module):
    """
    ResNet18 model with custom classifier for cats vs dogs classification
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False, dropout_rate=0.5):
        """
        Initialize ResNet18 model
        
        Args:
            num_classes (int): Number of output classes (2 for cats vs dogs)
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze ResNet18 parameters
            dropout_rate (float): Dropout rate for classifier
        """
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Remove the original classifier (last layer)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze ResNet18 parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Forward pass"""
        # Extract features using ResNet18
        features = self.backbone(x)
        
        # Apply custom classifier
        output = self.classifier(features)
        
        return output
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet18',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'num_classes': 2
        }

def create_resnet18_model(pretrained=True, freeze_backbone=False, dropout_rate=0.5):
    """
    Create ResNet18 model for cats vs dogs classification
    
    Args:
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone parameters
        dropout_rate (float): Dropout rate for classifier
    
    Returns:
        ResNet18Classifier: Model instance
    """
    return ResNet18Classifier(
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )

def get_training_config(strategy='transfer_learning'):
    """
    Get training configuration for ResNet18
    
    Args:
        strategy (str): Training strategy ('transfer_learning', 'fine_tuning', 'from_scratch')
    
    Returns:
        dict: Training configuration
    """
    configs = {
        'transfer_learning': {
            'pretrained': True,
            'freeze_backbone': False,
            'dropout_rate': 0.5,
            'learning_rate': 0.001,
            'description': 'ResNet18 with transfer learning (recommended)',
            'training_time': '1-2 hours',
            'expected_accuracy': '90-95%'
        },
        'fine_tuning': {
            'pretrained': True,
            'freeze_backbone': True,
            'dropout_rate': 0.3,
            'learning_rate': 0.0001,
            'description': 'ResNet18 with frozen backbone + fine-tuning',
            'training_time': '30-60 minutes',
            'expected_accuracy': '85-90%'
        },
        'from_scratch': {
            'pretrained': False,
            'freeze_backbone': False,
            'dropout_rate': 0.5,
            'learning_rate': 0.01,
            'description': 'ResNet18 trained from scratch',
            'training_time': '4-8 hours',
            'expected_accuracy': '80-90%'
        }
    }
    
    return configs.get(strategy, configs['transfer_learning'])

def test_resnet18_model():
    """Test ResNet18 model creation and forward pass"""
    print("=" * 60)
    print("TESTING RESNET18 MODEL")
    print("=" * 60)
    
    # Test different strategies
    strategies = ['transfer_learning', 'fine_tuning', 'from_scratch']
    
    for strategy in strategies:
        print(f"\n{strategy.upper()}:")
        print("-" * 30)
        
        try:
            # Get configuration
            config = get_training_config(strategy)
            
            # Create model
            model = create_resnet18_model(
                pretrained=config['pretrained'],
                freeze_backbone=config['freeze_backbone'],
                dropout_rate=config['dropout_rate']
            )
            
            # Get model info
            info = model.get_model_info()
            print(f"  Model: {info['model_name']}")
            print(f"  Total Parameters: {info['total_parameters']:,}")
            print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
            print(f"  Frozen Parameters: {info['frozen_parameters']:,}")
            print(f"  Learning Rate: {config['learning_rate']}")
            print(f"  Expected Accuracy: {config['expected_accuracy']}")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
                print(f"  Output Shape: {output.shape}")
                print(f"  ✓ Model works correctly")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    # Test the model
    test_resnet18_model()
    
    # Show usage example
    print("\n" + "=" * 60)
    print("USAGE EXAMPLE:")
    print("=" * 60)
    print("# Create ResNet18 model (recommended)")
    print("model = create_resnet18_model()")
    print("print(model.get_model_info())")
    print()
    print("# Create model with fine-tuning")
    print("model = create_resnet18_model(freeze_backbone=True)")
    print()
    print("# Get training configuration")
    print("config = get_training_config('transfer_learning')")
    print("print(config)")







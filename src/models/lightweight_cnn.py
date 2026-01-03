#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple

class LightweightCNN(nn.Module):
    """
    Lightweight CNN for CASME-II micro-expression recognition
    Designed for ~250 samples - avoids overfitting
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(LightweightCNN, self).__init__()
        
        # Use ResNet18 as backbone (much lighter than DenseNet121)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Moderate dropout
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
    def _init_classifier_weights(self):
        """Initialize classifier layers"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input images [B, 3, 224, 224]
        Returns:
            logits: Class logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits

def create_lightweight_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Create lightweight model for CASME-II
    Uses ResNet18 backbone as specified in clean config
    """
    model = LightweightCNN(num_classes=num_classes, pretrained=pretrained)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 Lightweight CNN Model (ResNet18 Backbone):")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"   Backbone: ResNet18 (CASME-II appropriate)")
    
    return model

if __name__ == "__main__":
    # Test the model
    model = create_lightweight_model(num_classes=4)
    
    # Test forward pass
    dummy_input = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Model test passed!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output range: logits")

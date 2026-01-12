#!/usr/bin/env python3
"""
Production-Ready Advanced Architecture with ROI CNN Encoders
Fixes all critical issues for real training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import torchvision.models as models

class ROICNNEncoder(nn.Module):
    """
    CNN encoder for ROI flow maps
    Reduces 224×224 ROI to 256-d vector
    """
    
    def __init__(self, input_channels=3, output_dim=256):
        super().__init__()
        
        # Small CNN architecture for ROI encoding
        self.encoder = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Final projection
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) ROI flow map
        Returns:
            (batch, output_dim) feature vector
        """
        return self.encoder(x)

class ProductionTemporalTransformer(nn.Module):
    """
    Production-ready temporal transformer with ROI encoders
    """
    
    def __init__(self, num_rois=3, roi_channels=3, hidden_dim=256, num_classes=4):
        super().__init__()
        
        self.num_rois = num_rois
        self.roi_channels = roi_channels
        
        # ROI CNN encoders (one per ROI)
        self.roi_encoders = nn.ModuleList([
            ROICNNEncoder(input_channels=roi_channels, output_dim=hidden_dim)
            for _ in range(num_rois)
        ])
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * num_rois,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Positional encoding for 4 temporal windows (learnable per ROI)
        self.pos_encoding = nn.Parameter(torch.randn(1, 4, num_rois, hidden_dim))
        
        # Classification head (NO SOFTMAX)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_rois, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def encode_rois(self, roi_flows):
        """
        Encode ROI flows using CNN encoders
        
        Args:
            roi_flows: (batch, seq_len, num_rois, channels, height, width)
        Returns:
            (batch, seq_len, num_rois, hidden_dim) - changed to keep ROI dimension separate
        """
        batch_size, seq_len, num_rois, channels, height, width = roi_flows.shape
        
        # Encode each ROI
        encoded_rois = []
        for roi_idx in range(num_rois):
            roi_flow = roi_flows[:, :, roi_idx, :, :, :]  # (batch, seq_len, channels, height, width)
            roi_flow = roi_flow.view(-1, channels, height, width)  # (batch * seq_len, channels, height, width)
            
            encoded = self.roi_encoders[roi_idx](roi_flow)  # (batch * seq_len, hidden_dim)
            encoded = encoded.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_dim)
            encoded_rois.append(encoded)
        
        # Stack ROI encodings: (batch, seq_len, num_rois, hidden_dim)
        combined = torch.stack(encoded_rois, dim=2)
        
        return combined
    
    def forward(self, roi_flows):
        """
        Args:
            roi_flows: (batch, seq_len, num_rois, channels, height, width)
        """
        # Encode ROIs
        x = self.encode_rois(roi_flows)  # (batch, seq_len, num_rois, hidden_dim)
        
        # Add per-ROI positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :, :]
        
        # Reshape for attention: (batch, seq_len, num_rois * hidden_dim)
        batch, seq, rois, hidden = x.shape
        x = x.view(batch, seq, rois * hidden)
        
        # Temporal attention
        x, _ = self.temporal_attention(x, x, x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, num_rois * hidden_dim)
        
        # Classification (NO SOFTMAX)
        output = self.classifier(x)
        
        return output

class ProductionGraphAttention(nn.Module):
    """
    Production-ready Graph Attention Network for ROI interactions
    """
    
    def __init__(self, num_rois=3, roi_channels=3, hidden_dim=256, num_classes=4):
        super().__init__()
        
        self.num_rois = num_rois
        
        # ROI CNN encoders
        self.roi_encoders = nn.ModuleList([
            ROICNNEncoder(input_channels=roi_channels, output_dim=hidden_dim)
            for _ in range(num_rois)
        ])
        
        # Graph attention layers
        self.graph_attention1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.graph_attention2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Classification head (NO SOFTMAX)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def encode_rois(self, roi_maps):
        """
        Encode ROI maps using CNN encoders
        
        Args:
            roi_maps: (batch, num_rois, channels, height, width)
        Returns:
            (batch, num_rois, hidden_dim)
        """
        batch_size, num_rois, channels, height, width = roi_maps.shape
        
        # Encode each ROI
        encoded_rois = []
        for roi_idx in range(num_rois):
            roi_map = roi_maps[:, roi_idx, :, :, :]  # (batch, channels, height, width)
            encoded = self.roi_encoders[roi_idx](roi_map)  # (batch, hidden_dim)
            encoded_rois.append(encoded)
        
        # Stack ROI encodings
        combined = torch.stack(encoded_rois, dim=1)  # (batch, num_rois, hidden_dim)
        
        return combined
    
    def forward(self, roi_maps):
        """
        Args:
            roi_maps: (batch, num_rois, channels, height, width)
        """
        # Encode ROIs
        x = self.encode_rois(roi_maps)  # (batch, num_rois, hidden_dim)
        
        # Graph attention layer 1
        residual = x
        x, _ = self.graph_attention1(x, x, x)
        x = self.norm1(x + residual)
        
        # Graph attention layer 2
        residual = x
        x, _ = self.graph_attention2(x, x, x)
        x = self.norm2(x + residual)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification (NO SOFTMAX)
        output = self.classifier(x)
        
        return output

class ProductionHybridModel(nn.Module):
    """
    Production-ready hybrid model with ROI attention fusion
    """
    
    def __init__(self, num_rois=3, roi_channels=3, hidden_dim=256, num_classes=4):
        super().__init__()
        
        self.temporal_model = ProductionTemporalTransformer(
            num_rois=num_rois, roi_channels=roi_channels, 
            hidden_dim=hidden_dim, num_classes=num_classes
        )
        
        self.gcn_model = ProductionGraphAttention(
            num_rois=num_rois, roi_channels=roi_channels,
            hidden_dim=hidden_dim, num_classes=num_classes
        )
        
        # ROI Attention weights (learnable importance per ROI)
        self.roi_attention = nn.Parameter(torch.ones(num_rois) / num_rois)  # Initialize to uniform
        
        # Fusion layer (NO SOFTMAX - logits only)
        self.fusion = nn.Linear(num_classes * 2, num_classes)
        
        # Learnable fusion weights
        self.temporal_weight = nn.Parameter(torch.tensor(0.5))
        self.gcn_weight = nn.Parameter(torch.tensor(0.5))
        
        # ROI names for interpretability
        self.roi_names = ['eyes', 'eyebrows', 'mouth']
    
    def forward(self, temporal_input, gcn_input):
        """
        Args:
            temporal_input: (batch, seq_len, num_rois, channels, height, width)
            gcn_input: (batch, num_rois, channels, height, width)
        """
        # Apply ROI attention to both inputs
        # Reshape attention weights to match input dimensions
        roi_attention_weights = F.softmax(self.roi_attention, dim=0)
        
        # Normalize attention weights for numerical stability
        roi_attention_weights = roi_attention_weights / roi_attention_weights.sum()
        
        # Apply attention to temporal input
        temporal_attention = roi_attention_weights.view(1, 1, -1, 1, 1, 1)  # (1, 1, num_rois, 1, 1, 1)
        temporal_input_attended = temporal_input * temporal_attention
        
        # Apply attention to GCN input  
        gcn_attention = roi_attention_weights.view(1, -1, 1, 1, 1)  # (1, num_rois, 1, 1, 1)
        gcn_input_attended = gcn_input * gcn_attention
        
        # Get predictions from both models with attended inputs
        temporal_out = self.temporal_model(temporal_input_attended)
        gcn_out = self.gcn_model(gcn_input_attended)
        
        # Clip fusion weights for stability
        tw = torch.sigmoid(self.temporal_weight)
        gw = torch.sigmoid(self.gcn_weight)
        
        # Weighted fusion
        temporal_out = temporal_out * tw
        gcn_out = gcn_out * gw
        
        # Concatenate and fuse
        combined = torch.cat([temporal_out, gcn_out], dim=1)
        return self.fusion(combined)
    
    def get_roi_importance(self):
        """Get ROI importance weights for interpretability"""
        roi_weights = F.softmax(self.roi_attention, dim=0)
        return {name: weight.item() for name, weight in zip(self.roi_names, roi_weights)}

def create_production_model(model_type='hybrid', num_classes=4, num_rois=3, roi_channels=3):
    """Create production-ready model"""
    if model_type == 'temporal':
        return ProductionTemporalTransformer(
            num_rois=num_rois, roi_channels=roi_channels, num_classes=num_classes
        )
    elif model_type == 'gcn':
        return ProductionGraphAttention(
            num_rois=num_rois, roi_channels=roi_channels, num_classes=num_classes
        )
    elif model_type == 'hybrid' or model_type == 'hybrid_attention':
        return ProductionHybridModel(
            num_rois=num_rois, roi_channels=roi_channels, num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_production_models():
    """Test production models with correct dimensions"""
    print("🚀 Testing Production-Ready Advanced Architectures")
    
    models = ['temporal', 'gcn', 'hybrid']
    
    for model_type in models:
        print(f"\n🧠 Testing {model_type.upper()} Model:")
        
        try:
            model = create_production_model(model_type=model_type)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✅ Created {model_type} model with {param_count:,} parameters")
            
            # Create appropriate dummy input
            if model_type == 'temporal':
                dummy_input = torch.randn(2, 2, 3, 3, 224, 224)  # (batch, seq_len, num_rois, channels, height, width)
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✅ Output shape: {output.shape}")
                print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
                
            elif model_type == 'gcn':
                dummy_input = torch.randn(4, 3, 3, 224, 224)  # (batch, num_rois, channels, height, width)
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✅ Output shape: {output.shape}")
                print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
                
            elif model_type == 'hybrid':
                dummy_temporal = torch.randn(2, 2, 3, 3, 224, 224)
                dummy_gcn = torch.randn(2, 3, 3, 224, 224)
                with torch.no_grad():
                    output = model(dummy_temporal, dummy_gcn)
                print(f"✅ Output shape: {output.shape}")
                print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"✅ Predicted emotions: {torch.argmax(output, dim=1).tolist()}")
                print(f"✅ Probabilities: {F.softmax(output, dim=1)[0].tolist()}")
            
            print(f"✅ {model_type.upper()} working correctly!")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Production Architecture Testing Complete!")

if __name__ == "__main__":
    test_production_models()

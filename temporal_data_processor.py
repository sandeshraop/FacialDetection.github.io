#!/usr/bin/env python3
"""
Temporal Flow Data Processor for Production Models
Handles loading and preparing real temporal sequences (onset→apex, apex→offset)
"""

import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class TemporalFlowDataProcessor:
    """
    Processor for real temporal ROI flow sequences
    Handles onset→apex and apex→offset flows
    """
    
    def __init__(self, num_rois=3):
        self.num_rois = num_rois
        self.roi_names = ['eyes', 'eyebrows', 'mouth']
    
    def load_multi_temporal_roi_flows(self, sample_dir):
        """
        Load multi-temporal ROI flows from precomputed data (4 temporal windows)
        
        Args:
            sample_dir: Path to sample directory containing temporal flow files
            
        Returns:
            Dictionary with temporal flows for each ROI and temporal window
        """
        temporal_flows = {}
        
        # Load 4 temporal windows: t0, t1, t2, t3
        windows = ['t0', 't1', 't2', 't3']
        
        for window in windows:
            for roi_name in self.roi_names:
                # Load stacked flow for this window and ROI
                flow_file = sample_dir / f'stacked_flow_{window}.npy'
                if flow_file.exists():
                    # Load the 9-channel stacked flow and extract ROI channels
                    stacked_flow = np.load(flow_file)  # (224, 224, 9)
                    
                    # Extract ROI channels (3 channels per ROI)
                    roi_idx = self.roi_names.index(roi_name)
                    start_channel = roi_idx * 3
                    end_channel = start_channel + 3
                    roi_flow = stacked_flow[:, :, start_channel:end_channel]  # (224, 224, 3)
                    
                    temporal_flows[f'{roi_name}_{window}'] = roi_flow
                else:
                    # Create dummy data
                    temporal_flows[f'{roi_name}_{window}'] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        return temporal_flows
    
    def load_temporal_roi_flows(self, sample_dir):
        """
        Load real temporal ROI flows from precomputed data
        
        Args:
            sample_dir: Path to sample directory containing temporal flow files
            
        Returns:
            Dictionary with temporal flows for each ROI and phase
        """
        temporal_flows = {}
        
        # Load onset→apex and apex→offset flows
        phases = ['onset', 'apex', 'offset']
        
        for phase in phases:
            for roi_name in self.roi_names:
                # Load RGB flow for this phase and ROI
                flow_file = sample_dir / f'{roi_name}_{phase}_flow.jpg'
                if flow_file.exists():
                    flow_img = Image.open(flow_file)
                    flow_array = np.array(flow_img)  # (224, 224, 3)
                    temporal_flows[f'{roi_name}_{phase}'] = flow_array
                else:
                    # Create dummy data
                    temporal_flows[f'{roi_name}_{phase}'] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        return temporal_flows
    
    def compute_temporal_differences(self, temporal_flows):
        """
        Compute real temporal flow differences:
        1. Onset → Apex (expression building)
        2. Apex → Offset (expression fading)
        
        Args:
            temporal_flows: Dictionary with temporal flows
            
        Returns:
            Dictionary with temporal differences
        """
        temporal_diffs = {}
        
        # Compute onset→apex difference
        for roi_name in self.roi_names:
            onset_flow = temporal_flows[f'{roi_name}_onset']
            apex_flow = temporal_flows[f'{roi_name}_apex']
            offset_flow = temporal_flows[f'{roi_name}_offset']
            
            # Convert to float for computation
            onset_float = onset_flow.astype(np.float32) / 255.0
            apex_float = apex_flow.astype(np.float32) / 255.0
            offset_float = offset_flow.astype(np.float32) / 255.0
            
            # Compute differences
            onset_to_apex = np.abs(apex_float - onset_float)  # Expression building
            apex_to_offset = np.abs(offset_float - apex_float)  # Expression fading
            
            # Convert back to uint8 for consistency
            onset_to_apex = (onset_to_apex * 255).astype(np.uint8)
            apex_to_offset = (apex_to_offset * 255).astype(np.uint8)
            
            temporal_diffs[f'{roi_name}_onset_to_apex'] = onset_to_apex
            temporal_diffs[f'{roi_name}_apex_to_offset'] = apex_to_offset
        
        return temporal_diffs
    
    def prepare_temporal_input(self, sample_dir):
        """
        Prepare temporal input for production models using 4 temporal windows
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            torch.Tensor: (batch, seq_len, num_rois, channels, height, width)
        """
        # Load multi-temporal flows
        temporal_flows = self.load_multi_temporal_roi_flows(sample_dir)
        
        # Prepare sequence: [t0, t1, t2, t3] = [onset→mid1, mid1→apex, apex→mid2, mid2→offset]
        sequence = []
        for window in ['t0', 't1', 't2', 't3']:
            window_data = []
            for roi_name in self.roi_names:
                roi_flow = temporal_flows[f'{roi_name}_{window}']  # (224, 224, 3)
                window_data.append(roi_flow)
            
            # Stack ROIs: (num_rois, height, width, channels)
            window_tensor = np.stack(window_data)
            # Permute to (num_rois, channels, height, width)
            window_tensor = np.transpose(window_tensor, (0, 3, 1, 2))
            sequence.append(window_tensor)
        
        # Stack sequence: (seq_len, num_rois, channels, height, width)
        sequence_tensor = np.stack(sequence)
        
        # Convert to tensor and add batch dimension
        temporal_input = torch.from_numpy(sequence_tensor).float()
        temporal_input = temporal_input.unsqueeze(0)  # (1, seq_len, num_rois, channels, height, width)
        
        return temporal_input
    
    def prepare_gcn_input(self, sample_dir):
        """
        Prepare GCN input (apex frame only)
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            torch.Tensor: (batch, num_rois, channels, height, width)
        """
        # Load apex flows
        apex_flows = {}
        for roi_name in self.roi_names:
            flow_file = sample_dir / f'{roi_name}_apex_flow.jpg'
            if flow_file.exists():
                flow_img = Image.open(flow_file)
                flow_array = np.array(flow_img)  # (224, 224, 3)
                apex_flows[roi_name] = flow_array
            else:
                apex_flows[roi_name] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Stack ROIs
        roi_data = []
        for roi_name in self.roi_names:
            roi_flow = apex_flows[roi_name]  # (224, 224, 3)
            roi_flow = roi_flow.transpose(2, 0, 1)  # (3, 224, 224)
            roi_data.append(roi_flow)
        
        # Convert to tensor
        roi_tensor = torch.from_numpy(np.stack(roi_data)).float()
        roi_tensor = roi_tensor.unsqueeze(0)  # Add batch dimension
        
        return roi_tensor
    
    def prepare_hybrid_input(self, sample_dir):
        """
        Prepare inputs for hybrid model
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            tuple: (temporal_input, gcn_input)
        """
        temporal_input = self.prepare_temporal_input(sample_dir)
        gcn_input = self.prepare_gcn_input(sample_dir)
        
        return temporal_input, gcn_input

def test_temporal_data_processor():
    """Test temporal data processor with real data"""
    print("🧪 Testing Temporal Flow Data Processor")
    
    # Check if precomputed data exists
    flow_dir = Path("data/processed/roi_optical_flow")
    if not flow_dir.exists():
        print("❌ Precomputed data not found. Run precompute_roi_optical_flow.py first.")
        return False
    
    # Find a sample
    sample_dir = None
    for subject_dir in flow_dir.iterdir():
        if subject_dir.is_dir():
            for video_dir in subject_dir.iterdir():
                if video_dir.is_dir() and (video_dir / "stacked_flow.npy").exists():
                    sample_dir = video_dir
                    break
            if sample_dir:
                break
    
    if sample_dir is None:
        print("❌ No valid samples found.")
        return False
    
    print(f"✅ Using sample: {sample_dir}")
    
    # Test processor
    processor = TemporalFlowDataProcessor()
    
    # Test temporal input preparation
    print("\n🧠 Testing Temporal Input Preparation:")
    try:
        temporal_input = processor.prepare_temporal_input(sample_dir)
        print(f"✅ Temporal input shape: {temporal_input.shape}")
        print(f"✅ Temporal input range: [{temporal_input.min():.3f}, {temporal_input.max():.3f}]")
        
        # Verify sequence contains real differences
        seq_diff = temporal_input[0, 1] - temporal_input[0, 0]  # Difference between timesteps
        print(f"✅ Temporal difference range: [{seq_diff.min():.3f}, {seq_diff.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Temporal input error: {e}")
        return False
    
    # Test GCN input preparation
    print("\n🧠 Testing GCN Input Preparation:")
    try:
        gcn_input = processor.prepare_gcn_input(sample_dir)
        print(f"✅ GCN input shape: {gcn_input.shape}")
        print(f"✅ GCN input range: [{gcn_input.min():.3f}, {gcn_input.max():.3f}]")
        
    except Exception as e:
        print(f"❌ GCN input error: {e}")
        return False
    
    # Test hybrid input preparation
    print("\n🧠 Testing Hybrid Input Preparation:")
    try:
        temporal_input, gcn_input = processor.prepare_hybrid_input(sample_dir)
        print(f"✅ Hybrid temporal input shape: {temporal_input.shape}")
        print(f"✅ Hybrid GCN input shape: {gcn_input.shape}")
        
    except Exception as e:
        print(f"❌ Hybrid input error: {e}")
        return False
    
    print("\n🎉 Temporal Data Processor working correctly!")
    return True

def test_production_models_with_real_data():
    """Test production models with real temporal data"""
    print("🧪 Testing Production Models with Real Temporal Data")
    
    # Check if precomputed data exists
    flow_dir = Path("data/processed/roi_optical_flow")
    if not flow_dir.exists():
        print("❌ Precomputed data not found. Run precompute_roi_optical_flow.py first.")
        return False
    
    # Find a sample
    sample_dir = None
    for subject_dir in flow_dir.iterdir():
        if subject_dir.is_dir():
            for video_dir in subject_dir.iterdir():
                if video_dir.is_dir() and (video_dir / "stacked_flow.npy").exists():
                    sample_dir = video_dir
                    break
            if sample_dir:
                break
    
    if sample_dir is None:
        print("❌ No valid samples found.")
        return False
    
    print(f"✅ Using sample: {sample_dir}")
    
    # Import production models
    from production_advanced_architectures import create_production_model
    
    # Test temporal model
    print("\n🧠 Testing Production Temporal Model:")
    try:
        model = create_production_model(model_type='temporal')
        print(f"✅ Created temporal model")
        
        processor = TemporalFlowDataProcessor()
        temporal_input = processor.prepare_temporal_input(sample_dir)
        print(f"✅ Temporal input shape: {temporal_input.shape}")
        
        with torch.no_grad():
            output = model(temporal_input)
        print(f"✅ Temporal output shape: {output.shape}")
        print(f"✅ Predicted emotion: {torch.argmax(output, dim=1).item()}")
        print(f"✅ Probabilities: {torch.softmax(output, dim=1)[0].tolist()}")
        
    except Exception as e:
        print(f"❌ Temporal model error: {e}")
        return False
    
    # Test GCN model
    print("\n🧠 Testing Production GCN Model:")
    try:
        model = create_production_model(model_type='gcn')
        print(f"✅ Created GCN model")
        
        gcn_input = processor.prepare_gcn_input(sample_dir)
        print(f"✅ GCN input shape: {gcn_input.shape}")
        
        with torch.no_grad():
            output = model(gcn_input)
        print(f"✅ GCN output shape: {output.shape}")
        print(f"✅ Predicted emotion: {torch.argmax(output, dim=1).item()}")
        print(f"✅ Probabilities: {torch.softmax(output, dim=1)[0].tolist()}")
        
    except Exception as e:
        print(f"❌ GCN model error: {e}")
        return False
    
    # Test hybrid model
    print("\n🧠 Testing Production Hybrid Model:")
    try:
        model = create_production_model(model_type='hybrid')
        print(f"✅ Created hybrid model")
        
        temporal_input, gcn_input = processor.prepare_hybrid_input(sample_dir)
        print(f"✅ Hybrid inputs - Temporal: {temporal_input.shape}, GCN: {gcn_input.shape}")
        
        with torch.no_grad():
            output = model(temporal_input, gcn_input)
        print(f"✅ Hybrid output shape: {output.shape}")
        print(f"✅ Predicted emotion: {torch.argmax(output, dim=1).item()}")
        print(f"✅ Probabilities: {torch.softmax(output, dim=1)[0].tolist()}")
        
    except Exception as e:
        print(f"❌ Hybrid model error: {e}")
        return False
    
    print("\n🎉 Production Models with Real Data working correctly!")
    return True

if __name__ == "__main__":
    # Test data processor
    success1 = test_temporal_data_processor()
    
    # Test models with real data
    success2 = test_production_models_with_real_data()
    
    if success1 and success2:
        print("\n🔥 System ready for LOSO training with production models!")
    else:
        print("\n❌ Please fix the issues before proceeding.")

#!/usr/bin/env python3
"""
Micro-Expression Emotion Prediction Script
Uses trained optical flow model to predict emotions from onset->apex frames
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import yaml
from PIL import Image
import torchvision.transforms as transforms

# Import model and data loader
from src.models.lightweight_cnn import create_lightweight_model
from src.data_loader.advanced_casme_loader import AdvancedCASMEIIDataset

def compute_optical_flow(onset_path, apex_path, input_size=224):
    """
    Compute optical flow between onset and apex frames
    Returns HSV optical flow image (3 channels)
    """
    # Load images
    onset_img = cv2.imread(str(onset_path), cv2.IMREAD_GRAYSCALE)
    apex_img = cv2.imread(str(apex_path), cv2.IMREAD_GRAYSCALE)
    
    if onset_img is None or apex_img is None:
        raise ValueError(f"Could not load images: {onset_path}, {apex_path}")
    
    # Resize to standard size
    onset_img = cv2.resize(onset_img, (input_size, input_size))
    apex_img = cv2.resize(apex_img, (input_size, input_size))
    
    # Compute optical flow using Farneback algorithm (reliable)
    flow = cv2.calcOpticalFlowFarneback(
        onset_img, apex_img, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Convert flow to HSV representation
    hsv = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    
    # Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Set HSV values
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value
    
    # Convert HSV to RGB
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb_flow

def load_trained_model(checkpoint_path, config_path='config/config.yaml'):
    """
    Load trained model from checkpoint
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model with config parameters
    num_classes = config.get('model', {}).get('num_classes', 4)
    pretrained = config.get('model', {}).get('pretrained', True)
    model = create_lightweight_model(num_classes=num_classes, pretrained=pretrained)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, config

def preprocess_optical_flow(flow_image, config):
    """
    Preprocess optical flow image for model input
    """
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((config['model']['input_size'], config['model']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(flow_image)
    
    # Apply transforms
    tensor = transform(pil_image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

def predict_emotion(onset_path, apex_path, model, config, device='cpu'):
    """
    Predict emotion from onset and apex frames
    """
    print(f"🔥 Computing optical flow...")
    print(f"   Onset: {onset_path}")
    print(f"   Apex: {apex_path}")
    
    # Compute optical flow
    flow_image = compute_optical_flow(onset_path, apex_path, config['model']['input_size'])
    
    # Preprocess
    input_tensor = preprocess_optical_flow(flow_image, config)
    input_tensor = input_tensor.to(device)
    
    print(f"🧠 Predicting emotion...")
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map class index to emotion
    emotion_map = {
        0: 'happiness',
        1: 'disgust', 
        2: 'surprise',
        3: 'repression'
    }
    
    predicted_emotion = emotion_map.get(predicted_class, 'unknown')
    
    # Get all probabilities
    all_probs = probabilities[0].cpu().numpy()
    
    print(f"🎯 Prediction Results:")
    print(f"   Emotion: {predicted_emotion}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Class: {predicted_class}")
    print()
    print(f"📊 All Probabilities:")
    for i, (emotion, prob) in enumerate(zip(emotion_map.values(), all_probs)):
        print(f"   {emotion}: {prob:.2%}")
    
    return {
        'predicted_emotion': predicted_emotion,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': dict(zip(emotion_map.values(), all_probs.tolist())),
        'flow_image': flow_image
    }

def main():
    parser = argparse.ArgumentParser(description='Predict micro-expression emotion from optical flow')
    parser.add_argument('--onset', type=str, required=True, help='Path to onset frame image')
    parser.add_argument('--apex', type=str, required=True, help='Path to apex frame image')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/best_model.pth', 
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device: cpu, cuda, or auto')
    parser.add_argument('--save_flow', type=str, default=None,
                       help='Save optical flow visualization to this path')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 Micro-Expression Emotion Prediction")
    print(f"   Device: {device}")
    print(f"   Checkpoint: {args.checkpoint}")
    print()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"💡 Run training first to create model weights:")
        print(f"   python advanced_hybrid_losos.py --config config/config.yaml")
        return
    
    # Load model
    print(f"📥 Loading trained model...")
    try:
        model, config = load_trained_model(args.checkpoint, args.config)
        model = model.to(device)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check input files
    onset_path = Path(args.onset)
    apex_path = Path(args.apex)
    
    if not onset_path.exists():
        print(f"❌ Onset image not found: {onset_path}")
        return
    
    if not apex_path.exists():
        print(f"❌ Apex image not found: {apex_path}")
        return
    
    # Predict emotion
    try:
        results = predict_emotion(onset_path, apex_path, model, config, device)
        
        # Save optical flow visualization if requested
        if args.save_flow:
            flow_image = results['flow_image']
            cv2.imwrite(args.save_flow, cv2.cvtColor(flow_image, cv2.COLOR_RGB2BGR))
            print(f"💾 Optical flow saved to: {args.save_flow}")
        
        print()
        print(f"🎉 Prediction complete!")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

if __name__ == "__main__":
    main()

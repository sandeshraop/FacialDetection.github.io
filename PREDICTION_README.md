# 🎯 Micro-Expression Emotion Prediction

## 📋 Overview
Predict micro-expression emotions using trained optical flow model. Takes onset and apex frames as input and outputs emotion classification.

## 🚀 Usage

### Basic Prediction
```bash
python predict_emotion.py --onset onset_frame.jpg --apex apex_frame.jpg
```

### With Custom Checkpoint
```bash
python predict_emotion.py --onset onset.jpg --apex apex.jpg --checkpoint output/checkpoints/best_model.pth
```

### Save Optical Flow Visualization
```bash
python predict_emotion.py --onset onset.jpg --apex apex.jpg --save_flow optical_flow.jpg
```

## 📊 Output
```
🎯 Prediction Results:
   Emotion: happiness
   Confidence: 67.34%
   Class: 0

📊 All Probabilities:
   happiness: 67.34%
   disgust: 12.45%
   surprise: 15.23%
   repression: 4.98%
```

## 🎯 Emotion Classes
- **0**: happiness
- **1**: disgust  
- **2**: surprise
- **3**: repression

## 📁 Requirements
- Trained model checkpoint (`output/checkpoints/best_model.pth`)
- Config file (`config/config.yaml`)
- Input images (onset and apex frames)

## 🔧 Options
- `--onset`: Path to onset frame image (required)
- `--apex`: Path to apex frame image (required)
- `--checkpoint`: Model checkpoint path (default: `output/checkpoints/best_model.pth`)
- `--config`: Config file path (default: `config/config.yaml`)
- `--device`: Device (cpu/cuda/auto, default: auto)
- `--save_flow`: Save optical flow visualization

## 🧪 Test Prediction
```bash
python test_prediction.py
```

## 🎯 How It Works
1. **Load Model**: Trained ResNet18 with optical flow input
2. **Compute Optical Flow**: Farneback algorithm between onset→apex
3. **Preprocess**: Resize, normalize, convert to tensor
4. **Predict**: Softmax classification with confidence scores
5. **Output**: Emotion prediction with probabilities

## 📈 Performance
- **Expected UAR**: 60-70% (optical flow improvement)
- **Input**: 224×224 RGB optical flow
- **Speed**: ~10ms per prediction (GPU)
- **Model**: 11M parameters (ResNet18)

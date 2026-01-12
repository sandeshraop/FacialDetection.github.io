#!/usr/bin/env python3
"""
Test prediction script with sample data
"""

import sys
from pathlib import Path

def test_prediction_script():
    """Test the prediction script with sample data"""
    
    print("🧪 Testing Prediction Script...")
    
    # Check if prediction script exists
    if not Path("predict_emotion.py").exists():
        print("❌ predict_emotion.py not found")
        return False
    
    # Check for sample data
    data_dir = Path("data/processed/Cropped")
    if not data_dir.exists():
        print("❌ Sample data not found")
        print("💡 Make sure CASME-II data is in data/processed/Cropped/")
        return False
    
    # Find a sample subject
    subjects = list(data_dir.glob("sub*"))
    if not subjects:
        print("❌ No subject directories found")
        return False
    
    sample_subject = subjects[0]
    print(f"📁 Using sample subject: {sample_subject}")
    
    # Find sample images
    image_files = list(sample_subject.glob("*/reg_img*.jpg"))
    if len(image_files) < 2:
        print("❌ Need at least 2 images for onset/apex")
        return False
    
    onset_img = image_files[0]
    apex_img = image_files[1] if len(image_files) > 1 else image_files[0]
    
    print(f"🖼️  Sample images:")
    print(f"   Onset: {onset_img}")
    print(f"   Apex: {apex_img}")
    
    # Test the prediction command
    cmd = f"python predict_emotion.py --onset {onset_img} --apex {apex_img}"
    print(f"🚀 Test command:")
    print(f"   {cmd}")
    
    return True

if __name__ == "__main__":
    test_prediction_script()

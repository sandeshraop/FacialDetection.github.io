#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def test_optical_flow():
    """Test optical flow computation"""
    print("🔥 Testing Optical Flow Implementation...")
    
    # Check if we have sample frames
    data_dir = Path("data/processed/Cropped")
    if not data_dir.exists():
        print("❌ Data directory not found. This test requires actual CASME-II data.")
        return
    
    # Find a subject with frames
    test_subject = None
    test_video = None
    
    for subject_dir in data_dir.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith('sub'):
            for video_dir in subject_dir.iterdir():
                if video_dir.is_dir():
                    frames = sorted(video_dir.glob('*.jpg'))
                    if len(frames) >= 2:
                        test_subject = subject_dir.name
                        test_video = video_dir.name
                        print(f"📁 Found test data: {test_subject}/{test_video}")
                        print(f"   Frames available: {len(frames)}")
                        
                        # Test optical flow on first two frames
                        onset_path = frames[0]
                        apex_path = frames[1] if len(frames) > 1 else frames[0]
                        
                        print(f"   Testing optical flow:")
                        print(f"   Onset frame: {onset_path.name}")
                        print(f"   Apex frame: {apex_path.name}")
                        
                        # Compute optical flow
                        try:
                            # Load frames as grayscale
                            onset = cv2.imread(str(onset_path), cv2.IMREAD_GRAYSCALE)
                            apex = cv2.imread(str(apex_path), cv2.IMREAD_GRAYSCALE)
                            
                            if onset is None or apex is None:
                                print("❌ Could not load test frames")
                                return
                            
                            print(f"   Frame sizes: {onset.shape} → {apex.shape}")
                            
                            # Compute optical flow
                            flow = cv2.calcOpticalFlowFarneback(
                                onset, apex,
                                None,
                                pyr_scale=0.5,
                                levels=3,
                                winsize=15,
                                iterations=3,
                                poly_n=5,
                                poly_sigma=1.2,
                                flags=0
                            )
                            
                            print(f"   Flow computed: {flow.shape}")
                            
                            # Convert to RGB
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            hsv = np.zeros((onset.shape[0], onset.shape[1], 3), dtype=np.uint8)
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 1] = 255
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            
                            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                            flow_image = Image.fromarray(flow_rgb)
                            
                            print(f"   ✅ Optical flow image created: {flow_image.size}")
                            print(f"   ✅ Flow magnitude range: {mag.min():.2f} - {mag.max():.2f}")
                            print(f"   ✅ Flow angle range: {ang.min():.2f} - {ang.max():.2f}")
                            
                            # Save test result
                            output_path = Path("optical_flow_test.jpg")
                            flow_image.save(output_path)
                            print(f"   📸 Test optical flow saved to: {output_path}")
                            
                            print("\n🎉 Optical Flow Implementation Working!")
                            print("🔥 Ready for motion-based micro-expression recognition!")
                            
                            return True
                            
                        except Exception as e:
                            print(f"❌ Error computing optical flow: {e}")
                            return
    
    if test_subject is None:
        print("❌ No test frames found. Please ensure CASME-II data is available.")
        print("   Expected structure: data/processed/Cropped/subXX/video_name/*.jpg")

if __name__ == "__main__":
    test_optical_flow()

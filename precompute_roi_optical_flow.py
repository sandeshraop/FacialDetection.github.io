#!/usr/bin/env python3
"""
ROI-Based Optical Flow Computation with MediaPipe
Computes separate optical flow for eyes, eyebrows, and mouth regions using precise face detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import re
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    print(" MediaPipe not found. Using Haar cascade fallback.")
    HAS_MEDIAPIPE = False

class ROIOpticalFlowComputer:
    def __init__(self):
        # Initialize MediaPipe Face Detection if available
        self.has_mediapipe = HAS_MEDIAPIPE
        if self.has_mediapipe:
            try:
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except AttributeError:
                print("⚠️ MediaPipe version incompatible. Using Haar cascade fallback.")
                self.has_mediapipe = False
        
        # ROI definitions based on facial landmarks
        self.roi_landmarks = {
            'eyes': [33, 133, 7, 163, 144, 145, 153, 154, 155, 156],  # Eye corners and surrounding points
            'eyebrows': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],  # Eyebrow points
            'mouth': [61, 84, 17, 314, 405, 291, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318],  # Mouth contour
            'full_face': list(range(468))  # All landmarks
        }
        
        # Fallback ROI regions (if landmarks fail)
        self.roi_regions = {
            'eyes': (0.1, 0.1, 0.8, 0.3),
            'eyebrows': (0.1, 0.05, 0.8, 0.15),
            'mouth': (0.2, 0.6, 0.6, 0.3),
            'full_face': (0.0, 0.0, 1.0, 1.0)
        }
    
    def normalize_ep(self, name):
        """
        Convert EP variations to common form: EPxx_xx
        Examples:
          EP01_2      -> EP01_02
          EP01_02f    -> EP01_02
          EP06_02_01  -> EP06_02
          EP03_14f    -> EP03_14
        """
        name = name.replace('f', '')
        m = re.search(r'EP(\d+)[_\-]?(\d+)', name.upper())
        if m:
            ep = int(m.group(1))
            clip = int(m.group(2))
            return f"EP{ep:02d}_{clip:02d}"
        return None
    
    def detect_face_with_mediapipe(self, frame):
        """
        Detect face and landmarks using MediaPipe
        Returns face bounding box and landmarks
        """
        if not self.has_mediapipe:
            # Fallback to Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                return faces[0], None
            else:
                return None, None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Convert normalized landmarks to pixel coordinates
            h_img, w_img, _ = frame.shape
            landmark_coords = np.array([
                [int(lm.x * w_img), int(lm.y * h_img)] for lm in landmarks.landmark
            ])
            x, y, w, h = cv2.boundingRect(landmark_coords)
            
            return (x, y, w, h), landmarks.landmark
        else:
            # Fallback to Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                return faces[0], None
            else:
                return None, None
    
    def extract_roi_from_landmarks(self, frame, landmarks, roi_name):
        """
        Extract ROI using facial landmarks for precise regions
        """
        if landmarks is None:
            return None, None
        
        # Get landmark coordinates
        landmark_coords = np.array([
            [int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in landmarks
        ])
        
        # Get ROI landmark indices
        roi_indices = self.roi_landmarks[roi_name]
        
        # Extract ROI coordinates
        roi_coords = landmark_coords[roi_indices]
        
        # Get bounding box for ROI
        x, y, w, h = cv2.boundingRect(roi_coords)
        
        # Add some padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        
        return roi, (x, y, w, h)
    
    def extract_roi_fallback(self, frame, face_rect, roi_name):
        """
        Fallback ROI extraction using face bounding box
        """
        x, y, w, h = face_rect
        roi_params = self.roi_regions[roi_name]
        
        # Calculate ROI coordinates
        roi_x = int(x + roi_params[0] * w)
        roi_y = int(y + roi_params[1] * h)
        roi_w = int(roi_params[2] * w)
        roi_h = int(roi_params[3] * h)
        
        # Extract ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        return roi, (roi_x, roi_y, roi_w, roi_h)
    
    def compute_flow_from_rois(self, roi1, roi2):
        """
        Compute optical flow from already extracted ROIs
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        input_size = 224
        gray1 = cv2.resize(gray1, (input_size, input_size))
        gray2 = cv2.resize(gray2, (input_size, input_size))
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate flow magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        
        # Convert to HSV for visualization
        hsv = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb_flow, avg_magnitude, magnitude, angle
    
    def compute_optical_flow_roi(self, frame1, frame2, face_rect, roi_name='full_face'):
        """
        Compute optical flow for specific ROI (legacy - kept for compatibility)
        """
        # Extract ROI from both frames
        roi1, _ = self.extract_roi_fallback(frame1, face_rect, roi_name)
        roi2, _ = self.extract_roi_fallback(frame2, face_rect, roi_name)
        
        return self.compute_flow_from_rois(roi1, roi2)
    
    def compute_multi_roi_flow(self, frame1, frame2, face_rect, landmarks=None, landmarks1=None):
        """
        Compute optical flow for multiple ROIs using MediaPipe landmarks
        Supports frame-specific landmarks for better alignment
        Returns dictionary of flow images and magnitudes
        """
        roi_flows = {}
        
        for roi_name in ['eyes', 'eyebrows', 'mouth']:
            try:
                # Try landmark-based extraction with frame-specific landmarks
                if landmarks1 is not None and landmarks is not None:
                    # Use frame-specific landmarks for each frame
                    roi1, _ = self.extract_roi_from_landmarks(frame1, landmarks1, roi_name)
                    roi2, _ = self.extract_roi_from_landmarks(frame2, landmarks, roi_name)
                elif landmarks is not None:
                    # Fallback to single landmarks for both frames
                    roi1, _ = self.extract_roi_from_landmarks(frame1, landmarks, roi_name)
                    roi2, _ = self.extract_roi_from_landmarks(frame2, landmarks, roi_name)
                else:
                    # Fallback to bounding box
                    roi1, _ = self.extract_roi_fallback(frame1, face_rect, roi_name)
                    roi2, _ = self.extract_roi_fallback(frame2, face_rect, roi_name)
                
                if roi1 is not None and roi2 is not None:
                    flow_image, avg_magnitude, magnitude, angle = self.compute_flow_from_rois(roi1, roi2)
                    roi_flows[roi_name] = {
                        'image': flow_image,
                        'magnitude': avg_magnitude,
                        'magnitude_map': magnitude,
                        'angle_map': angle
                    }
                else:
                    # Create dummy flow
                    dummy_flow = np.zeros((224, 224, 3), dtype=np.uint8)
                    dummy_mag = np.zeros((224, 224), dtype=np.float32)
                    dummy_angle = np.zeros((224, 224), dtype=np.float32)
                    roi_flows[roi_name] = {
                        'image': Image.fromarray(dummy_flow),
                        'magnitude': 0.0,
                        'magnitude_map': dummy_mag,
                        'angle_map': dummy_angle
                    }
                    
            except Exception as e:
                print(f"Error computing {roi_name} flow: {e}")
                # Create dummy flow
                dummy_flow = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_mag = np.zeros((224, 224), dtype=np.float32)
                dummy_angle = np.zeros((224, 224), dtype=np.float32)
                roi_flows[roi_name] = {
                    'image': Image.fromarray(dummy_flow),
                    'magnitude': 0.0,
                    'magnitude_map': dummy_mag,
                    'angle_map': dummy_angle
                }
        
        return roi_flows
    
    def stack_roi_flows(self, roi_flows):
        """
        Stack ROI flows into multi-channel image
        Returns 9-channel image (3 ROIs × 3 channels)
        """
        # Convert to numpy arrays
        eyes_flow = np.array(roi_flows['eyes']['image'])
        brows_flow = np.array(roi_flows['eyebrows']['image'])
        mouth_flow = np.array(roi_flows['mouth']['image'])
        
        # Stack channels: [R,G,B] for each ROI
        stacked_flow = np.concatenate([
            eyes_flow,      # 3 channels
            brows_flow,     # 3 channels
            mouth_flow      # 3 channels
        ], axis=2)  # Shape: (224, 224, 9)
        
        # Return numpy array directly (don't convert to PIL Image)
        return stacked_flow

def map_frame_number_to_index(target_frame_num, frame_paths):
    """Map Excel frame number to actual frame index"""
    # Extract frame numbers from filenames
    frame_numbers = []
    for frame_path in frame_paths:
        # Extract number from filename (safer method)
        filename = frame_path.stem
        import re
        nums = re.findall(r'\d+', filename)
        frame_num = int(nums[-1]) if nums else 0
        frame_numbers.append(frame_num)
    
    # Find closest frame number to target
    if not frame_numbers:
        return len(frame_paths) // 2  # Default to middle
    
    closest_idx = min(range(len(frame_numbers)), 
                    key=lambda i: abs(frame_numbers[i] - target_frame_num))
    return closest_idx

def precompute_roi_optical_flow(data_dir, force_recompute=False):
    """
    Precompute ROI-based optical flow for CASME-II dataset
    """
    data_dir = Path(data_dir)
    processed_dir = Path('data/processed')  # Fixed path
    
    # Create output directory
    flow_dir = processed_dir / 'roi_optical_flow'
    flow_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Precomputing ROI-based optical flow for CASME-II dataset")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {flow_dir}")
    
    # Initialize ROI flow computer
    flow_computer = ROIOpticalFlowComputer()
    
    # Load metadata
    coding_path = processed_dir / 'Cropped' / 'CASME2-coding-20140508.xlsx'
    df = pd.read_excel(coding_path)
    
    # Rename columns properly
    df = df.rename(columns={
        df.columns[0]: 'subject',
        df.columns[1]: 'filename',
        df.columns[3]: 'onset_frame',
        df.columns[4]: 'apex_frame',
        df.columns[5]: 'offset_frame',
        df.columns[8]: 'emotion'
    })
    
    # Select only needed columns
    df = df[['subject', 'filename', 'onset_frame', 'apex_frame', 'offset_frame', 'emotion']]
    
    # Clean data
    df = df.dropna(subset=['subject', 'filename', 'emotion'])
    df['subject'] = df['subject'].astype(int).astype(str).str.zfill(2)
    df['subject'] = 'sub' + df['subject']
    df['filename'] = df['filename'].astype(str).str.strip()
    df['emotion'] = df['emotion'].astype(str).str.lower().str.strip()
    
    # Filter to 4 main emotions
    df = df[df['emotion'].isin(['happiness', 'disgust', 'surprise', 'repression'])]
    print(f"📊 Found {len(df)} valid samples with 4 main emotions")
    
    # Process each sample
    processed_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing ROI optical flow"):
        try:
            subject = row['subject']
            video_name = row['filename'].split('.')[0]
            
            # Create output directory for this sample
            sample_flow_dir = flow_dir / subject / video_name
            sample_flow_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already computed
            stacked_flow_file = sample_flow_dir / 'stacked_flow.npy'
            if stacked_flow_file.exists() and not force_recompute:
                processed_count += 1
                continue
            
            # Find frame directory using normalized EP matching
            subject_dir = processed_dir / 'Cropped' / subject
            video_dir = None
            
            target_ep = flow_computer.normalize_ep(video_name)
            
            if subject_dir.exists() and target_ep is not None:
                for d in subject_dir.iterdir():
                    if d.is_dir():
                        if flow_computer.normalize_ep(d.name) == target_ep:
                            video_dir = d
                            break
            
            if not video_dir or not video_dir.exists():
                print(f"❌ No video directory found for {subject}/{video_name}")
                skipped_count += 1
                continue
            
            # Find frames
            frames = sorted(video_dir.glob('*.jpg'))
            if len(frames) < 2:
                skipped_count += 1
                continue
            
            # Map frame numbers to indices (use Excel frames!)
            onset_frame_num = int(row['onset_frame']) if pd.notna(row['onset_frame']) else 1
            apex_frame_num = int(row['apex_frame']) if pd.notna(row['apex_frame']) else len(frames) // 2
            offset_frame_num = int(row['offset_frame']) if pd.notna(row['offset_frame']) else len(frames) - 1
            
            # Calculate temporal window frames
            onset_idx = map_frame_number_to_index(onset_frame_num, frames)
            apex_idx = map_frame_number_to_index(apex_frame_num, frames)
            offset_idx = map_frame_number_to_index(offset_frame_num, frames)
            
            # Create temporal windows
            mid1_idx = (onset_idx + apex_idx) // 2
            mid2_idx = (apex_idx + offset_idx) // 2
            
            temporal_windows = [
                (onset_idx, mid1_idx, "t0"),
                (mid1_idx, apex_idx, "t1"), 
                (apex_idx, mid2_idx, "t2"),
                (mid2_idx, offset_idx, "t3")
            ]
            
            # Process each temporal window
            temporal_flows = {}
            
            for start_idx, end_idx, window_name in temporal_windows:
                # Load frames for this window
                start_frame = cv2.imread(str(frames[start_idx]))
                end_frame = cv2.imread(str(frames[end_idx]))
                
                if start_frame is None or end_frame is None:
                    print(f"⚠️ Missing frames for {window_name} in {subject}/{video_name}")
                    continue
                
                # Detect face using MediaPipe on both frames separately
                face_rect1, landmarks1 = flow_computer.detect_face_with_mediapipe(start_frame)
                face_rect2, landmarks2 = flow_computer.detect_face_with_mediapipe(end_frame)
                
                # Use fallback if either detection fails
                if face_rect1 is None or face_rect2 is None:
                    h, w, _ = end_frame.shape
                    face_rect1 = face_rect2 = (0, 0, w, h)
                    landmarks1 = landmarks2 = None
                    print(f"⚠️ Face detection failed for {subject}/{video_name} {window_name}, using full frame ROI")
                
                # Use frame-specific landmarks for better ROI alignment
                start_landmarks = landmarks1 if landmarks1 is not None else landmarks2
                end_landmarks = landmarks2 if landmarks2 is not None else landmarks1
                
                # Compute ROI flows for this window with frame-specific landmarks
                roi_flows = flow_computer.compute_multi_roi_flow(start_frame, end_frame, face_rect2, end_landmarks, start_landmarks)
                
                # Stack flows into 9-channel image
                stacked_flow = flow_computer.stack_roi_flows(roi_flows)
                
                # Save temporal window
                temporal_flows[window_name] = {
                    'stacked_flow': stacked_flow,
                    'roi_flows': roi_flows
                }
                
                # Save as numpy array (preserves 9 channels)
                np.save(sample_flow_dir / f'stacked_flow_{window_name}.npy', stacked_flow)
                
                # Save individual ROI data for advanced analysis
                for roi_name, flow_data in roi_flows.items():
                    # Save magnitude
                    mag_file = sample_flow_dir / f'{roi_name}_magnitude_{window_name}.npy'
                    np.save(mag_file, flow_data['magnitude'])
                    
                    # Save magnitude map
                    mag_map_file = sample_flow_dir / f'{roi_name}_magnitude_map_{window_name}.npy'
                    np.save(mag_map_file, flow_data['magnitude_map'])
                    
                    # Save angle map
                    angle_map_file = sample_flow_dir / f'{roi_name}_angle_map_{window_name}.npy'
                    np.save(angle_map_file, flow_data['angle_map'])
                    
                    # Save RGB flow for visualization
                    rgb_file = sample_flow_dir / f'{roi_name}_flow_{window_name}.jpg'
                    if isinstance(flow_data['image'], Image.Image):
                        flow_data['image'].save(rgb_file)
                    else:
                        # Convert numpy array to PIL Image
                        Image.fromarray(flow_data['image']).save(rgb_file)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            skipped_count += 1
    
    print(f"\n🎉 ROI optical flow precomputation complete!")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Skipped: {skipped_count}")
    print(f"📁 ROI flows saved to: {flow_dir}")
    print(f"🧠 Ready for multi-branch CNN architecture!")

def main():
    parser = argparse.ArgumentParser(description='Precompute ROI-based optical flow for CASME-II dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--force_recompute', action='store_true', help='Force recompute existing flows')
    
    args = parser.parse_args()
    
    precompute_roi_optical_flow(args.data_dir, args.force_recompute)

if __name__ == "__main__":
    main()

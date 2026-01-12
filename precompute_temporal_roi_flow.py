#!/usr/bin/env python3
"""
Temporal ROI-Based Optical Flow (3-Frame)
Computes optical flow for onset→apex and apex→offset
Creates 18-channel input for advanced temporal modeling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the base ROI computer
from precompute_roi_optical_flow import ROIOpticalFlowComputer

class TemporalROIOpticalFlowComputer(ROIOpticalFlowComputer):
    def __init__(self):
        super().__init__()
    
    def compute_temporal_roi_flow(self, onset_frame, apex_frame, offset_frame, face_rect, landmarks=None):
        """
        Compute optical flow for three temporal phases:
        1. Onset → Apex (expression building)
        2. Apex → Offset (expression fading)
        
        Returns 18-channel stacked flow
        """
        temporal_flows = {}
        
        for roi_name in ['eyes', 'eyebrows', 'mouth']:
            try:
                # Extract ROIs for all three frames
                if landmarks is not None:
                    roi_onset, _ = self.extract_roi_from_landmarks(onset_frame, landmarks, roi_name)
                    roi_apex, _ = self.extract_roi_from_landmarks(apex_frame, landmarks, roi_name)
                    roi_offset, _ = self.extract_roi_from_landmarks(offset_frame, landmarks, roi_name)
                else:
                    # Fallback to bounding box
                    roi_onset, _ = self.extract_roi_fallback(onset_frame, face_rect, roi_name)
                    roi_apex, _ = self.extract_roi_fallback(apex_frame, face_rect, roi_name)
                    roi_offset, _ = self.extract_roi_fallback(offset_frame, face_rect, roi_name)
                
                if all(roi is not None for roi in [roi_onset, roi_apex, roi_offset]):
                    # Compute onset→apex flow
                    flow1, mag1, mag_map1, angle1 = self.compute_flow_from_rois(roi_onset, roi_apex)
                    
                    # Compute apex→offset flow
                    flow2, mag2, mag_map2, angle2 = self.compute_flow_from_rois(roi_apex, roi_offset)
                    
                    temporal_flows[roi_name] = {
                        'onset_flow': flow1,
                        'apex_flow': flow2,
                        'onset_magnitude': mag1,
                        'apex_magnitude': mag2,
                        'onset_mag_map': mag_map1,
                        'apex_mag_map': mag_map2,
                        'onset_angle_map': angle1,
                        'apex_angle_map': angle2
                    }
                else:
                    # Create dummy flows
                    dummy_flow = np.zeros((224, 224, 3), dtype=np.uint8)
                    dummy_mag = np.zeros((224, 224), dtype=np.float32)
                    dummy_angle = np.zeros((224, 224), dtype=np.float32)
                    
                    temporal_flows[roi_name] = {
                        'onset_flow': Image.fromarray(dummy_flow),
                        'apex_flow': Image.fromarray(dummy_flow),
                        'onset_magnitude': 0.0,
                        'apex_magnitude': 0.0,
                        'onset_mag_map': dummy_mag,
                        'apex_mag_map': dummy_mag,
                        'onset_angle_map': dummy_angle,
                        'apex_angle_map': dummy_angle
                    }
                    
            except Exception as e:
                print(f"Error computing {roi_name} temporal flow: {e}")
                # Create dummy flows
                dummy_flow = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_mag = np.zeros((224, 224), dtype=np.float32)
                dummy_angle = np.zeros((224, 224), dtype=np.float32)
                
                temporal_flows[roi_name] = {
                    'onset_flow': Image.fromarray(dummy_flow),
                    'apex_flow': Image.fromarray(dummy_flow),
                    'onset_magnitude': 0.0,
                    'apex_magnitude': 0.0,
                    'onset_mag_map': dummy_mag,
                    'apex_mag_map': dummy_mag,
                    'onset_angle_map': dummy_angle,
                    'apex_angle_map': dummy_angle
                }
        
        return temporal_flows
    
    def stack_temporal_flows(self, temporal_flows):
        """
        Stack temporal ROI flows into 18-channel image
        Order: [eyes_onset, eyes_apex, brows_onset, brows_apex, mouth_onset, mouth_apex]
        """
        # Convert to numpy arrays
        eyes_onset = np.array(temporal_flows['eyes']['onset_flow'])
        eyes_apex = np.array(temporal_flows['eyes']['apex_flow'])
        brows_onset = np.array(temporal_flows['eyebrows']['onset_flow'])
        brows_apex = np.array(temporal_flows['eyebrows']['apex_flow'])
        mouth_onset = np.array(temporal_flows['mouth']['onset_flow'])
        mouth_apex = np.array(temporal_flows['mouth']['apex_flow'])
        
        # Stack channels: 6 flows × 3 channels = 18 channels
        stacked_flow = np.concatenate([
            eyes_onset,    # 3 channels
            eyes_apex,     # 3 channels
            brows_onset,   # 3 channels
            brows_apex,    # 3 channels
            mouth_onset,   # 3 channels
            mouth_apex     # 3 channels
        ], axis=2)  # Shape: (224, 224, 18)
        
        return Image.fromarray(stacked_flow)

def precompute_temporal_roi_flow(data_dir, force_recompute=False):
    """
    Precompute temporal ROI-based optical flow for all samples
    """
    data_dir = Path(data_dir)
    processed_dir = data_dir / 'processed'
    
    # Create output directory
    flow_dir = data_dir / 'processed' / 'temporal_roi_optical_flow'
    flow_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Precomputing Temporal ROI-based optical flow for CASME-II dataset")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Output directory: {flow_dir}")
    print(f"🔥 Creating 18-channel temporal flows!")
    
    # Initialize temporal flow computer
    flow_computer = TemporalROIOpticalFlowComputer()
    
    # Load metadata
    coding_path = processed_dir / 'Cropped' / 'CASME2-coding-20140508.xlsx'
    df = pd.read_excel(coding_path, header=0)
    
    # Extract data
    data = {
        'subject': df.iloc[1:, 0].astype(str).str.strip(),
        'filename': df.iloc[1:, 1].astype(str).str.strip(),
        'onset_frame': pd.to_numeric(df.iloc[1:, 3], errors='coerce'),
        'apex_frame': pd.to_numeric(df.iloc[1:, 4], errors='coerce'),
        'offset_frame': pd.to_numeric(df.iloc[1:, 5], errors='coerce'),
        'emotion': df.iloc[1:, 8].astype(str).str.strip()
    }
    df = pd.DataFrame(data)
    
    # Clean and filter
    df = df[df['subject'].notna() & df['filename'].notna() & df['emotion'].notna()]
    df['subject'] = 'sub' + df['subject'].str.zfill(2)
    df['emotion'] = df['emotion'].str.lower().str.strip()
    
    # Map emotions to 4-class standard
    emotion_mapping = {
        'happiness': 'happiness',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'repression': 'repression'
    }
    df = df[df['emotion'].isin(['happiness', 'disgust', 'surprise', 'repression'])]
    df['emotion'] = df['emotion'].map(emotion_mapping)
    
    print(f"📊 Found {len(df)} valid samples")
    
    # Process each sample
    processed_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing Temporal ROI flows"):
        try:
            subject = row['subject']
            video_name = row['filename'].split('.')[0]
            
            # Create output directory for this sample
            sample_flow_dir = flow_dir / subject / video_name
            sample_flow_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already computed
            stacked_flow_file = sample_flow_dir / 'temporal_stacked_flow.npy'
            if stacked_flow_file.exists() and not force_recompute:
                processed_count += 1
                continue
            
            # Find frame directory
            subject_dir = processed_dir / 'Cropped' / subject
            video_dir = None
            
            if subject_dir.exists():
                for d in subject_dir.iterdir():
                    if d.is_dir() and d.name.upper() == video_name.upper():
                        video_dir = d
                        break
            
            if not video_dir or not video_dir.exists():
                skipped_count += 1
                continue
            
            # Find frames
            frames = sorted(video_dir.glob('*.jpg'))
            if len(frames) < 3:
                skipped_count += 1
                continue
            
            # Map frame numbers to indices (use Excel frames!)
            onset_frame_num = int(row['onset_frame']) if pd.notna(row['onset_frame']) else 1
            apex_frame_num = int(row['apex_frame']) if pd.notna(row['apex_frame']) else len(frames) // 2
            offset_frame_num = int(row['offset_frame']) if pd.notna(row['offset_frame']) else len(frames) - 1
            
            def map_frame_number_to_index(target_frame_num, frame_paths):
                """Map Excel frame number to actual frame index"""
                frame_numbers = []
                for frame_path in frame_paths:
                    filename = frame_path.stem
                    import re
                    nums = re.findall(r'\d+', filename)
                    frame_num = int(nums[-1]) if nums else 0
                    frame_numbers.append(frame_num)
                
                if not frame_numbers:
                    return len(frame_paths) // 2
                
                closest_idx = min(range(len(frame_numbers)), 
                                key=lambda i: abs(frame_numbers[i] - target_frame_num))
                return closest_idx
            
            onset_idx = map_frame_number_to_index(onset_frame_num, frames)
            apex_idx = map_frame_number_to_index(apex_frame_num, frames)
            offset_idx = map_frame_number_to_index(offset_frame_num, frames)
            
            # Clamp indices
            onset_idx = max(0, min(onset_idx, len(frames) - 1))
            apex_idx = max(0, min(apex_idx, len(frames) - 1))
            offset_idx = max(0, min(offset_idx, len(frames) - 1))
            
            # Load frames
            onset_frame = cv2.imread(str(frames[onset_idx]))
            apex_frame = cv2.imread(str(frames[apex_idx]))
            offset_frame = cv2.imread(str(frames[offset_idx]))
            
            if any(frame is None for frame in [onset_frame, apex_frame, offset_frame]):
                skipped_count += 1
                continue
            
            # Detect face using MediaPipe (or fallback)
            face_rect, landmarks = flow_computer.detect_face_with_mediapipe(apex_frame)
            
            if face_rect is None:
                skipped_count += 1
                continue
            
            # Compute temporal ROI flows
            temporal_flows = flow_computer.compute_temporal_roi_flow(
                onset_frame, apex_frame, offset_frame, face_rect, landmarks
            )
            
            # Stack flows into 18-channel image
            stacked_flow = flow_computer.stack_temporal_flows(temporal_flows)
            
            # Save as numpy array (preserves 18 channels)
            stacked_flow_array = np.array(stacked_flow)
            np.save(sample_flow_dir / 'temporal_stacked_flow.npy', stacked_flow_array)
            
            # Save individual ROI temporal data
            for roi_name, flow_data in temporal_flows.items():
                # Save onset and apex magnitudes
                np.save(sample_flow_dir / f'{roi_name}_onset_magnitude.npy', flow_data['onset_magnitude'])
                np.save(sample_flow_dir / f'{roi_name}_apex_magnitude.npy', flow_data['apex_magnitude'])
                
                # Save magnitude maps
                np.save(sample_flow_dir / f'{roi_name}_onset_mag_map.npy', flow_data['onset_mag_map'])
                np.save(sample_flow_dir / f'{roi_name}_apex_mag_map.npy', flow_data['apex_mag_map'])
                
                # Save angle maps
                np.save(sample_flow_dir / f'{roi_name}_onset_angle_map.npy', flow_data['onset_angle_map'])
                np.save(sample_flow_dir / f'{roi_name}_apex_angle_map.npy', flow_data['apex_angle_map'])
                
                # Save RGB flows for visualization
                flow_data['onset_flow'].save(sample_flow_dir / f'{roi_name}_onset_flow.jpg')
                flow_data['apex_flow'].save(sample_flow_dir / f'{roi_name}_apex_flow.jpg')
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            skipped_count += 1
    
    print(f"\n🎉 Temporal ROI optical flow precomputation complete!")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Skipped: {skipped_count}")
    print(f"📁 Temporal flows saved to: {flow_dir}")
    print(f"🚀 Ready for 18-channel temporal CNN architecture!")

def main():
    parser = argparse.ArgumentParser(description='Precompute temporal ROI-based optical flow for CASME-II dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--force_recompute', action='store_true', help='Force recompute existing flows')
    
    args = parser.parse_args()
    
    precompute_temporal_roi_flow(args.data_dir, args.force_recompute)

if __name__ == "__main__":
    main()

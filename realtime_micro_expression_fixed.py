#!/usr/bin/env python3
"""
Fixed Real-time Micro-Expression Recognition with ROI Attention
Corrected version addressing all critical issues
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
import time
from pathlib import Path

# Import your production model
from production_advanced_architectures import create_production_model
from precompute_roi_optical_flow import ROIOpticalFlowComputer

class RealTimeMicroExpressionRecognizer:
    def __init__(self, model_path=None):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize ROI flow computer
        self.flow_computer = ROIOpticalFlowComputer()
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_production_model('hybrid_attention')
        self.model.to(self.device)
        self.model.eval()
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from {model_path}")
        
        # Initialize buffers
        self.roi_flows_buffer = deque(maxlen=4)  # Store last 4 flows
        
        # ROI names
        self.roi_names = ['eyes', 'eyebrows', 'mouth']
        
        # Emotion labels
        self.emotions = ['disgust', 'happiness', 'repression', 'surprise']
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Disable gradients for performance
        torch.set_grad_enabled(False)
        
    def extract_roi_from_frame(self, frame, landmarks):
        """Extract ROI regions from frame using landmarks"""
        try:
            roi_flows = {}
            
            if landmarks is None:
                print("❌ No landmarks provided")
                return None
            
            for roi_name in self.roi_names:
                roi_flow, _ = self.flow_computer.extract_roi_from_landmarks(
                    frame, landmarks, roi_name
                )
                if roi_flow is not None:
                    roi_flows[roi_name] = roi_flow
            
            return roi_flows
        except Exception as e:
            print(f"❌ ROI extraction error: {e}")
            return None
    
    def compute_optical_flow_between_frames(self, prev_rois, curr_rois):
        """Compute optical flow between ROI images"""
        try:
            flow_data = {}
            for roi_name in self.roi_names:
                if roi_name in prev_rois and roi_name in curr_rois:
                    # Compute flow between consecutive ROI images
                    flow_img, magnitude, _, _ = self.flow_computer.compute_flow_from_rois(
                        prev_rois[roi_name], curr_rois[roi_name]
                    )
                    flow_data[roi_name] = flow_img
                else:
                    # Create dummy data
                    flow_data[roi_name] = np.zeros((224, 224, 3), dtype=np.uint8)
            
            return flow_data
        except Exception as e:
            print(f"❌ Flow computation error: {e}")
            return None
    
    def prepare_model_input(self):
        """Prepare input tensor for model from ROI flows buffer"""
        if len(self.roi_flows_buffer) < 4:
            return None, None
        
        # Stack 4 temporal windows
        temporal_sequence = []
        gcn_data = []
        
        for flow_data in self.roi_flows_buffer:
            window_data = []
            for roi_name in self.roi_names:
                if roi_name in flow_data:
                    roi_flow = flow_data[roi_name]
                    if isinstance(roi_flow, np.ndarray):
                        roi_flow = cv2.resize(roi_flow, (224, 224))
                        if roi_flow.shape[-1] != 3:
                            roi_flow = np.stack([roi_flow] * 3, axis=-1)[:,:,:3]
                        window_data.append(roi_flow)
                    else:
                        # Create dummy data
                        window_data.append(np.zeros((224, 224, 3), dtype=np.uint8))
                else:
                    window_data.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            temporal_sequence.append(window_data)
        
        # Use latest flow for GCN input
        latest_flow = self.roi_flows_buffer[-1]
        gcn_sequence = []
        for roi_name in self.roi_names:
            if roi_name in latest_flow:
                roi_flow = latest_flow[roi_name]
                if isinstance(roi_flow, np.ndarray):
                    roi_flow = cv2.resize(roi_flow, (224, 224))
                    if roi_flow.shape[-1] != 3:
                        roi_flow = np.stack([roi_flow] * 3, axis=-1)[:,:,:3]
                    gcn_sequence.append(roi_flow)
                else:
                    gcn_sequence.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                gcn_sequence.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Convert to tensors
        temporal_array = np.array(temporal_sequence)  # (4, 3, 224, 224, 3)
        temporal_array = np.transpose(temporal_array, (0, 1, 4, 2, 3))  # (4, 3, 3, 224, 224)
        temporal_tensor = torch.from_numpy(temporal_array).float().unsqueeze(0)  # (1, 4, 3, 3, 224, 224)
        
        gcn_array = np.array(gcn_sequence)  # (3, 224, 224, 3)
        gcn_array = np.transpose(gcn_array, (0, 3, 1, 2))  # (3, 3, 224, 224)
        gcn_tensor = torch.from_numpy(gcn_array).float().unsqueeze(0)  # (1, 3, 3, 224, 224)
        
        # Normalize
        temporal_tensor = temporal_tensor / 255.0
        gcn_tensor = gcn_tensor / 255.0
        
        return temporal_tensor.to(self.device), gcn_tensor.to(self.device)
    
    def predict_emotion(self, temporal_input, gcn_input):
        """Run model inference"""
        try:
            with torch.no_grad():
                output = self.model(temporal_input, gcn_input)
                probabilities = torch.softmax(output, dim=1)
                return probabilities.cpu().numpy()[0]
        except Exception as e:
            print(f"❌ Inference error: {e}")
            return np.array([0.25, 0.25, 0.25, 0.25])
    
    def draw_results(self, frame, probabilities, landmarks):
        """Draw emotion probabilities and ROI information"""
        h, w = frame.shape[:2]
        
        # Draw emotion probabilities
        y_start = 30
        for i, (emotion, prob) in enumerate(zip(self.emotions, probabilities)):
            # Bar background
            bar_width = int(prob * 200)
            cv2.rectangle(frame, (10, y_start), (10 + bar_width, y_start + 25), 
                           (0, 255, 0), -1)
            # Text
            cv2.putText(frame, f"{emotion}: {prob:.2f}", 
                       (15, y_start + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_start += 30
        
        # Draw ROI importance if available
        if hasattr(self.model, 'get_roi_importance'):
            roi_importance = self.model.get_roi_importance()
            y_start = h - 100
            
            cv2.putText(frame, "ROI Importance:", (10, y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_start += 25
            
            for roi_name, importance in roi_importance.items():
                cv2.putText(frame, f"{roi_name}: {importance:.3f}", 
                           (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_start += 20
        
        # Draw FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            fps = 30 / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.fps_start_time = time.time()
        
        return frame
    
    def run(self):
        """Main real-time recognition loop"""
        print("🎥 Starting Real-time Micro-Expression Recognition")
        print("Press 'q' to quit, 's' to save current frame")
        print("✅ Fixed version - correct ROI flow computation")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize buffers
        prev_rois = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame")
                break
            
            frame_count += 1
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.mp_face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract ROI from current frame
                curr_rois = self.extract_roi_from_frame(frame, landmarks)
                
                if curr_rois and prev_rois:
                    # Compute optical flow between consecutive ROI images
                    flow_data = self.compute_optical_flow_between_frames(prev_rois, curr_rois)
                    
                    if flow_data:
                        # Add to buffer
                        self.roi_flows_buffer.append(flow_data)
                        
                        # Run inference every 5 frames for performance
                        if frame_count % 5 == 0 and len(self.roi_flows_buffer) >= 4:
                            # Prepare model input
                            temporal_input, gcn_input = self.prepare_model_input()
                            
                            if temporal_input is not None:
                                # Predict emotion
                                probabilities = self.predict_emotion(temporal_input, gcn_input)
                                
                                # Draw results
                                frame = self.draw_results(frame, probabilities, landmarks)
                
                prev_rois = curr_rois
            
            # Display
            cv2.imshow('Real-time Micro-Expression Recognition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'snapshot_{int(time.time())}.jpg', frame)
                print("📸 Snapshot saved")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.mp_face_mesh.close()
        print("🎉 Real-time recognition stopped")

def main():
    """Main function"""
    # Check for trained model
    model_path = "checkpoints/best_hybrid_attention_model.pth"
    
    recognizer = RealTimeMicroExpressionRecognizer(model_path)
    recognizer.run()

if __name__ == "__main__":
    main()

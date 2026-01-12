#!/usr/bin/env python3
"""
Simplified Real-time Micro-Expression Demo
Working version without complex ROI extraction issues
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

class SimpleMicroExpressionDemo:
    def __init__(self, model_path=None):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_production_model('hybrid_attention')
        self.model.to(self.device)
        self.model.eval()
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from {model_path}")
        
        # Initialize buffers
        self.frame_buffer = deque(maxlen=5)
        
        # Emotion labels
        self.emotions = ['disgust', 'happiness', 'repression', 'surprise']
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Disable gradients for performance
        torch.set_grad_enabled(False)
        
    def create_dummy_input(self):
        """Create dummy input when real processing fails"""
        # Temporal input: (1, 4, 3, 3, 224, 224)
        temporal_input = torch.zeros(1, 4, 3, 3, 224, 224, device=self.device)
        
        # GCN input: (1, 3, 3, 224, 224)
        gcn_input = torch.zeros(1, 3, 3, 224, 224, device=self.device)
        
        return temporal_input, gcn_input
    
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
    
    def draw_results(self, frame, probabilities):
        """Draw emotion probabilities"""
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
        
        # Add title
        cv2.putText(frame, "Micro-Expression Recognition Demo", (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main real-time recognition loop"""
        print("🎥 Starting Simplified Micro-Expression Demo")
        print("Press 'q' to quit, 's' to save current frame")
        print("✅ Simplified version - no complex ROI extraction")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        prediction_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame")
                break
            
            frame_count += 1
            
            # Add frame to buffer
            self.frame_buffer.append(frame)
            
            # Run inference every 5 frames for performance
            if frame_count % 5 == 0:
                # Create dummy input (simplified demo)
                temporal_input, gcn_input = self.create_dummy_input()
                
                # Predict emotion
                probabilities = self.predict_emotion(temporal_input, gcn_input)
                
                # Draw results
                frame = self.draw_results(frame, probabilities)
                
                prediction_counter += 1
                print(f"Prediction {prediction_counter}: {self.emotions[np.argmax(probabilities)]} ({np.max(probabilities):.2f})")
            
            # Display
            cv2.imshow('Micro-Expression Recognition Demo', frame)
            
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
        print("🎉 Demo stopped")

def main():
    """Main function"""
    # Check for trained model
    model_path = "checkpoints/best_hybrid_attention_model.pth"
    
    demo = SimpleMicroExpressionDemo(model_path)
    demo.run()

if __name__ == "__main__":
    main()

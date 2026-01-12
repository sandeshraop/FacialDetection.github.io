#!/usr/bin/env python3
"""
Production LOSO Training Script
Leave-One-Subject-Out training with advanced ROI-based architectures
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import re
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import production models
from production_advanced_architectures import create_production_model
from temporal_data_processor import TemporalFlowDataProcessor

class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing for better generalization
    """
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.weight = weight
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            targets = smooth_targets
        
        # Calculate focal loss
        ce_loss = F.cross_entropy(inputs, targets.argmax(dim=1) if self.label_smoothing > 0 else targets, 
                                 weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class ROIDataset(Dataset):
    """Dataset for ROI-based optical flow with temporal sequences"""
    
    def __init__(self, sample_paths, labels, label_encoder, model_type='hybrid'):
        self.sample_paths = sample_paths
        self.labels = labels
        self.model_type = model_type
        self.processor = TemporalFlowDataProcessor()
        self.label_encoder = label_encoder
        
        # Use global encoder, don't fit per dataset
        self.encoded_labels = self.label_encoder.transform(labels)
        
    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle failed samples"""
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        result = torch.utils.data.dataloader.default_collate(batch)
        return result

    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        label = self.encoded_labels[idx]
        
        try:
            if self.model_type == 'temporal':
                temporal_input = self.processor.prepare_temporal_input(sample_path)
                # Subject-wise normalization
                temporal_mean = temporal_input.mean()
                temporal_std = temporal_input.std()
                if temporal_std > 0:
                    temporal_input = (temporal_input - temporal_mean) / temporal_std
                return temporal_input, label
            elif self.model_type == 'gcn':
                gcn_input = self.processor.prepare_gcn_input(sample_path)
                # Subject-wise normalization
                gcn_mean = gcn_input.mean()
                gcn_std = gcn_input.std()
                if gcn_std > 0:
                    gcn_input = (gcn_input - gcn_mean) / gcn_std
                return gcn_input, label
            elif self.model_type == 'hybrid' or self.model_type == 'hybrid_attention':
                temporal_input, gcn_input = self.processor.prepare_hybrid_input(sample_path)
                # Subject-wise normalization
                temporal_mean = temporal_input.mean()
                temporal_std = temporal_input.std()
                if temporal_std > 0:
                    temporal_input = (temporal_input - temporal_mean) / temporal_std
                
                gcn_mean = gcn_input.mean()
                gcn_std = gcn_input.std()
                if gcn_std > 0:
                    gcn_input = (gcn_input - gcn_mean) / gcn_std
                
                return (temporal_input, gcn_input), label
        except Exception as e:
            print(f"❌ Error loading sample {sample_path}: {e}")
            # Return None instead of dummy data - will be filtered by collate_fn
            return None

class LOSOTrainer:
    """Leave-One-Subject-Out trainer for production models"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
    
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
        m = re.search(r'EP(\d+)[_\-]?(\d+)', name)
        if m:
            ep = int(m.group(1))
            clip = int(m.group(2))
            return f"EP{ep:02d}_{clip:02d}"
        return None
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_subject_samples(self):
        """Get all samples organized by subject"""
        data_dir = Path("data/processed/roi_optical_flow")
        samples_by_subject = {}
        
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return samples_by_subject
        
        # Load Excel metadata for labels
        excel_path = Path("data/processed/Cropped/CASME2-coding-20140508.xlsx")
        if excel_path.exists():
            df = pd.read_excel(excel_path)
            # Create mapping from sample to emotion using normalized EP
            emotion_mapping = {}
            valid_emotions = {'happiness', 'disgust', 'surprise', 'repression'}
            
            for _, row in df.iterrows():
                emotion = row['Estimated Emotion']
                if emotion in valid_emotions:
                    subject = f"sub{int(row['Subject']):02d}"
                    fname = str(row['Filename']).strip()
                    norm = self.normalize_ep(fname)
                    
                    if norm is None:
                        continue
                    
                    emotion_mapping.setdefault(subject, {})[norm] = emotion
            
            total_excel_matches = sum(len(emotions) for emotions in emotion_mapping.values())
            print(f"📊 Found {total_excel_matches} valid samples with 4 main emotions")
        else:
            print("❌ Excel metadata not found. Cannot proceed without labels.")
            return samples_by_subject
        
        # Scan data directory using normalized EP matching
        for subject_dir in data_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub'):
                subject_id = subject_dir.name
                samples_by_subject[subject_id] = []
                
                for video_dir in subject_dir.iterdir():
                    if video_dir.is_dir() and (video_dir / "stacked_flow.npy").exists():
                        roi_name = video_dir.name
                        norm_roi = self.normalize_ep(roi_name)
                        
                        if subject_id in emotion_mapping and norm_roi in emotion_mapping[subject_id]:
                            emotion = emotion_mapping[subject_id][norm_roi]
                            samples_by_subject[subject_id].append({
                                'path': video_dir,
                                'emotion': emotion
                            })
                        else:
                            print(f"❌ No emotion label found for {subject_id}/{roi_name}")
        
        return samples_by_subject
    
    def prepare_loso_data(self, samples_by_subject, test_subject):
        """Prepare train/test data for LOSO with given test subject"""
        train_samples = []
        train_labels = []
        test_samples = []
        test_labels = []
        
        for subject_id, subject_samples in samples_by_subject.items():
            for sample in subject_samples:
                if subject_id == test_subject:
                    test_samples.append(sample['path'])
                    test_labels.append(sample['emotion'])
                else:
                    train_samples.append(sample['path'])
                    train_labels.append(sample['emotion'])
        
        return train_samples, train_labels, test_samples, test_labels
    
    def train_model(self, model, train_loader, val_loader, model_type):
        """Train model with early stopping"""
        # Calculate class weights for focal loss
        all_labels = []
        for batch in train_loader:
            if batch is None:
                continue
            _, labels = batch
            all_labels.extend(labels.numpy())
        
        # Calculate class weights (inverse frequency)
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        num_classes = len(self.global_label_encoder.classes_)
        
        # Create full class weights array
        class_weights = np.ones(num_classes)
        for i, label in enumerate(unique_labels):
            class_weights[label] = len(all_labels) / (len(unique_labels) * counts[i])
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        
        # Use Focal Loss with class weights and label smoothing
        loss_config = self.config.get('loss', {})
        gamma = loss_config.get('gamma', 2.0)
        alpha = loss_config.get('alpha', 0.25)
        label_smoothing = loss_config.get('label_smoothing', 0.1)
        
        criterion = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing, weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.config['training']['lr'])
        
        # Use CosineAnnealingLR for better stability with small datasets
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['training']['num_epochs'],
            eta_min=1e-6
        )
        
        best_val_uar = 0.0
        patience_counter = 0
        max_patience = self.config['training']['early_stop']
        
        for epoch in range(self.config['training']['num_epochs']):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                if batch is None:
                    continue
                
                if model_type == 'hybrid' or model_type == 'hybrid_attention':
                    (temporal_data, gcn_data), labels = batch
                    temporal_data = temporal_data.to(self.device)
                    gcn_data = gcn_data.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Safe squeeze - only remove batch dimension if present
                    if temporal_data.dim() == 7:  # (batch, 1, seq, rois, C, H, W)
                        temporal_data = temporal_data.squeeze(1)
                    if gcn_data.dim() == 6:  # (batch, 1, rois, C, H, W)
                        gcn_data = gcn_data.squeeze(1)
                    
                    outputs = model(temporal_data, gcn_data)
                else:
                    data, labels = batch
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Safe squeeze - only remove batch dimension if present
                    if data.dim() == 6:  # (batch, 1, seq, rois, C, H, W) or (batch, 1, rois, C, H, W)
                        data = data.squeeze(1)
                    
                    outputs = model(data)
                
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                        
                    if model_type == 'hybrid' or model_type == 'hybrid_attention':
                        (temporal_data, gcn_data), labels = batch
                        temporal_data = temporal_data.to(self.device)
                        gcn_data = gcn_data.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Safe squeeze - only remove batch dimension if present
                        if temporal_data.dim() == 7:  # (batch, 1, seq, rois, C, H, W)
                            temporal_data = temporal_data.squeeze(1)
                        if gcn_data.dim() == 6:  # (batch, 1, rois, C, H, W)
                            gcn_data = gcn_data.squeeze(1)
                        
                        outputs = model(temporal_data, gcn_data)
                    else:
                        data, labels = batch
                        data = data.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Safe squeeze - only remove batch dimension if present
                        if data.dim() == 6:  # (batch, 1, seq, rois, C, H, W) or (batch, 1, rois, C, H, W)
                            data = data.squeeze(1)
                        
                        outputs = model(data)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    
                    val_predictions.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            # Calculate UAR
            val_uar = self.calculate_uar(val_true, val_predictions)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {100*train_correct/train_total:.2f}%, "
                  f"Val UAR: {100*val_uar:.2f}%")
            
            # Early stopping
            if val_uar > best_val_uar:
                best_val_uar = val_uar
                patience_counter = 0
                # Save best model
                checkpoint_path = Path(f'checkpoints/best_{model_type}_model.pth')
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
            
            scheduler.step()
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return best_val_uar
    
    def calculate_uar(self, y_true, y_pred):
        """Calculate Unweighted Average Recall"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] == 0:
            return 0.0
        
        per_class_recall = cm.diagonal() / cm.sum(axis=1)
        per_class_recall = per_class_recall[~np.isnan(per_class_recall)]
        
        return np.mean(per_class_recall) if len(per_class_recall) > 0 else 0.0
    
    def evaluate_model(self, model, test_loader, model_type, dataset):
        """Evaluate model on test set"""
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue
                    
                if model_type == 'hybrid' or model_type == 'hybrid_attention':
                    (temporal_data, gcn_data), labels = batch
                    temporal_data = temporal_data.to(self.device)
                    gcn_data = gcn_data.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Safe squeeze - only remove batch dimension if present
                    if temporal_data.dim() == 7:  # (batch, 1, seq, rois, C, H, W)
                        temporal_data = temporal_data.squeeze(1)
                    if gcn_data.dim() == 6:  # (batch, 1, rois, C, H, W)
                        gcn_data = gcn_data.squeeze(1)
                    
                    outputs = model(temporal_data, gcn_data)
                else:
                    data, labels = batch
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Safe squeeze - only remove batch dimension if present
                    if data.dim() == 6:  # (batch, 1, seq, rois, C, H, W) or (batch, 1, rois, C, H, W)
                        data = data.squeeze(1)
                    
                    outputs = model(data)
                
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_uar = self.calculate_uar(true_labels, predictions)
        
        # Convert back to emotion names for report
        emotion_names = dataset.label_encoder.inverse_transform(true_labels)
        pred_emotions = dataset.label_encoder.inverse_transform(predictions)
        
        # Detailed report
        report = classification_report(
            emotion_names, pred_emotions,
            output_dict=True, zero_division=0
        )
        
        # Return both numeric and string labels
        return test_uar, report, emotion_names, pred_emotions, true_labels, predictions
    
    def run_loso_experiment(self, model_type='hybrid'):
        """Run complete LOSO experiment"""
        print(f"🚀 Starting LOSO experiment with {model_type.upper()} model")
        
        samples_by_subject = self.get_subject_samples()
        subjects = list(samples_by_subject.keys())
        
        if not subjects:
            print("❌ No subjects found. Check data directory.")
            return
        
        print(f"📊 Found {len(subjects)} subjects: {subjects}")
        
        # Create GLOBAL label encoder
        all_labels = []
        for subject_samples in samples_by_subject.values():
            for sample in subject_samples:
                all_labels.append(sample['emotion'])
        
        # Ensure all 4 target emotions are included in encoder
        target_emotions = ['happiness', 'disgust', 'surprise', 'repression']
        for emotion in target_emotions:
            if emotion not in all_labels:
                all_labels.append(emotion)  # Add missing emotions
        
        self.global_label_encoder = LabelEncoder()
        self.global_label_encoder.fit(all_labels)
        print(f"🏷️ Global label mapping: {dict(zip(self.global_label_encoder.classes_, self.global_label_encoder.transform(self.global_label_encoder.classes_)))}")
        
        all_results = []
        
        for test_subject in subjects:
            print(f"\n🧪 Testing on subject: {test_subject}")
            
            # Prepare data
            train_samples, train_labels, test_samples, test_labels = self.prepare_loso_data(samples_by_subject, test_subject)
            
            if len(train_samples) == 0 or len(test_samples) == 0:
                print(f"⚠️ No data for subject {test_subject}, skipping...")
                continue
            
            print(f"📈 Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
            
            # Create datasets with GLOBAL encoder
            full_train_dataset = ROIDataset(train_samples, train_labels, self.global_label_encoder, model_type)
            test_dataset = ROIDataset(test_samples, test_labels, self.global_label_encoder, model_type)
            
            # Use random split for validation (more stable for small datasets)
            train_size = int(0.9 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # Reproducible
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.config['data']['batch_size'],
                shuffle=True, num_workers=0, collate_fn=ROIDataset.collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config['data']['batch_size'],
                shuffle=False, num_workers=0, collate_fn=ROIDataset.collate_fn
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.config['data']['batch_size'],
                shuffle=False, num_workers=0, collate_fn=ROIDataset.collate_fn
            )
            
            # Create model
            if model_type == 'hybrid_attention':
                model = create_production_model(model_type='hybrid')  # Uses ROI attention hybrid
            else:
                model = create_production_model(model_type=model_type)
            model = model.to(self.device)
            
            # Train model
            best_val_uar = self.train_model(model, train_loader, val_loader, model_type)
            
            # Load best model for evaluation
            checkpoint_path = Path(f'checkpoints/best_{model_type}_model.pth')
            if checkpoint_path.exists():
                model.load_state_dict(torch.load(checkpoint_path))
            else:
                print(f"⚠️ No checkpoint found, using current model state")
            
            # Evaluate
            test_uar, report, true_emotions, pred_emotions, true_labels, predictions = self.evaluate_model(
                model, test_loader, model_type, test_dataset
            )
            
            # Log ROI importance if using attention model
            if model_type == 'hybrid_attention' and hasattr(model, 'get_roi_importance'):
                roi_importance = model.get_roi_importance()
                print(f"🧠 ROI Importance: {roi_importance}")
                most_important = max(roi_importance, key=roi_importance.get)
                print(f"🎯 Most Important ROI: {most_important} ({roi_importance[most_important]:.3f})")
            
            # Store results
            result = {
                'subject_id': test_subject,
                'train_samples': len(train_samples),
                'test_samples': len(test_samples),
                'best_val_uar': best_val_uar * 100,
                'test_uar': test_uar * 100,
                'report': report,
                'confusion_matrix': confusion_matrix(true_labels, predictions, labels=range(len(self.global_label_encoder.classes_))),
                'true_emotions': true_emotions,
                'pred_emotions': pred_emotions,
                'true_labels': true_labels,
                'pred_labels': predictions
            }
            all_results.append(result)
            
            print(f"✅ {test_subject}: Val UAR: {best_val_uar*100:.2f}%, Test UAR: {test_uar*100:.2f}%")
        
        # Calculate overall statistics
        self.calculate_overall_stats(all_results, model_type)
        
        return all_results
    
    def calculate_overall_stats(self, results, model_type):
        """Calculate overall LOSO statistics"""
        if not results:
            print("❌ No results to analyze")
            return
        
        test_uars = [r['test_uar'] for r in results]
        val_uars = [r['best_val_uar'] for r in results]
        
        # Calculate additional metrics
        all_true_labels = []
        all_pred_labels = []
        for r in results:
            all_true_labels.extend(r['true_labels'])
            all_pred_labels.extend(r['pred_labels'])
        
        # Overall accuracy
        overall_accuracy = np.mean(np.array(all_true_labels) == np.array(all_pred_labels))
        
        # Overall F1 macro
        from sklearn.metrics import f1_score
        overall_f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro')
        
        stats = {
            'model_type': model_type,
            'mean_test_uar': np.mean(test_uars),
            'std_test_uar': np.std(test_uars),
            'mean_val_uar': np.mean(val_uars),
            'best_test_uar': np.max(test_uars),
            'worst_test_uar': np.min(test_uars),
            'num_subjects': len(results),
            'overall_accuracy': overall_accuracy,
            'overall_f1_macro': overall_f1_macro
        }
        
        print(f"\n📊 {model_type.upper()} LOSO Results:")
        print(f"   Mean Test UAR: {stats['mean_test_uar']:.2f}% ± {stats['std_test_uar']:.2f}%")
        print(f"   Best Test UAR: {stats['best_test_uar']:.2f}%")
        print(f"   Worst Test UAR: {stats['worst_test_uar']:.2f}%")
        print(f"   Mean Val UAR: {stats['mean_val_uar']:.2f}%")
        print(f"   Overall Accuracy: {stats['overall_accuracy']:.2f}%")
        print(f"   Overall F1-Macro: {stats['overall_f1_macro']:.3f}")
        print(f"   Subjects: {stats['num_subjects']}")
        
        # Save detailed results
        self.save_results(results, stats, model_type)
    
    def save_results(self, results, stats, model_type):
        """Save LOSO results to CSV files with confusion matrices"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_results = []
        for r in results:
            detailed_results.append({
                'subject_id': r['subject_id'],
                'best_val_uar': r['best_val_uar'],
                'test_uar': r['test_uar'],
                'train_samples': r['train_samples'],
                'test_samples': r['test_samples']
            })
        
        df_detailed = pd.DataFrame(detailed_results)
        df_detailed.to_csv(results_dir / f'loso_{model_type}_results.csv', index=False)
        
        # Save summary statistics
        summary_data = {
            'metric': ['mean_test_uar', 'std_test_uar', 'mean_val_uar', 'best_test_uar', 'worst_test_uar'],
            'value': [stats['mean_test_uar'], stats['std_test_uar'], stats['mean_val_uar'], 
                     stats['best_test_uar'], stats['worst_test_uar']]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(results_dir / f'loso_{model_type}_summary.csv', index=False)
        
        # Save confusion matrices per subject
        confusion_data = []
        for r in results:
            cm = r['confusion_matrix']
            emotion_names = self.global_label_encoder.classes_
            
            # Per-class recall
            per_class_recall = cm.diagonal() / cm.sum(axis=1)
            
            for i, emotion in enumerate(emotion_names):
                if i < len(per_class_recall):
                    confusion_data.append({
                        'subject_id': r['subject_id'],
                        'emotion': emotion,
                        'recall': per_class_recall[i],
                        'support': cm.sum(axis=1)[i]
                    })
        
        df_confusion = pd.DataFrame(confusion_data)
        df_confusion.to_csv(results_dir / f'loso_{model_type}_per_class_recall.csv', index=False)
        
        # Save overall confusion matrix
        all_true = []
        all_pred = []
        for r in results:
            all_true.extend(r['true_emotions'])
            all_pred.extend(r['pred_emotions'])
        
        if len(all_true) > 0 and len(all_pred) > 0:
            overall_cm = confusion_matrix(all_true, all_pred)
            
            # Only save if we have data for all classes
            if overall_cm.shape == (len(self.global_label_encoder.classes_), len(self.global_label_encoder.classes_)):
                # Save confusion matrix as CSV
                cm_df = pd.DataFrame(overall_cm, 
                                   columns=self.global_label_encoder.classes_,
                                   index=self.global_label_encoder.classes_)
                cm_df.to_csv(results_dir / f'loso_{model_type}_confusion_matrix.csv')
                print(f"� Confusion matrix saved to results/loso_{model_type}_confusion_matrix.csv")
            else:
                print(f"⚠️ Confusion matrix shape {overall_cm.shape} doesn't match expected {len(self.global_label_encoder.classes_)}x{len(self.global_label_encoder.classes_)}")
        else:
            print("⚠️ No predictions to save confusion matrix")
        
        print(f"💾 Results saved to results/loso_{model_type}_*.csv")
        print(f"🎯 Per-class recall saved to results/loso_{model_type}_per_class_recall.csv")

def main():
    parser = argparse.ArgumentParser(description='LOSO Training with Production Models')
    parser.add_argument('--config', type=str, default='config/config_optimized.yaml', help='Config file path')
    parser.add_argument('--model_type', type=str, default='hybrid_attention', 
                       choices=['temporal', 'gcn', 'hybrid', 'hybrid_attention'], help='Model type')
    parser.add_argument('--all_models', action='store_true', help='Test all model types')
    
    args = parser.parse_args()
    
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    trainer = LOSOTrainer(args.config)
    
    if args.all_models:
        # Test all models
        model_types = ['temporal', 'gcn', 'hybrid', 'hybrid_attention']
        for model_type in model_types:
            print(f"\n{'='*60}")
            trainer.run_loso_experiment(model_type)
    else:
        # Test single model
        trainer.run_loso_experiment(args.model_type)
    
    print(f"\n🎉 LOSO training complete! Check results/ directory for detailed results.")

if __name__ == "__main__":
    main()

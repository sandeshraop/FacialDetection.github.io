#!/usr/bin/env python3
"""
Create publication-ready visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def create_visualizations():
    """Create all publication figures"""
    
    # Load results
    results_df = pd.read_csv('results/loso_hybrid_attention_results.csv')
    confusion_df = pd.read_csv('results/loso_hybrid_attention_confusion_matrix.csv')
    recall_df = pd.read_csv('results/loso_hybrid_attention_per_class_recall.csv')
    
    # Create figures directory
    Path('figures').mkdir(exist_ok=True)
    
    # 1. Subject-wise Performance
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['subject_id'], results_df['test_uar'], 
            color='steelblue', alpha=0.7, edgecolor='black')
    plt.axhline(y=np.mean(results_df['test_uar']), color='red', 
                linestyle='--', label=f'Mean: {np.mean(results_df["test_uar"]):.1f}%')
    plt.xlabel('Subject ID')
    plt.ylabel('Test UAR (%)')
    plt.title('Subject-wise Test UAR Performance')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/subject_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['test_uar'], bins=10, color='skyblue', 
             alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(results_df['test_uar']), color='red', 
                linestyle='--', label=f'Mean: {np.mean(results_df["test_uar"]):.1f}%')
    plt.xlabel('Test UAR (%)')
    plt.ylabel('Frequency')
    plt.title('Test UAR Distribution Across Subjects')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Emotion-wise Performance
    if 'disgust' in recall_df.columns:
        emotions = ['disgust', 'happiness', 'repression', 'surprise']
        emotion_scores = [recall_df[emotion].mean() for emotion in emotions if emotion in recall_df.columns]
        emotion_labels = [emotion.capitalize() for emotion in emotions if emotion in recall_df.columns]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(emotion_labels, emotion_scores, 
                      color=['red', 'yellow', 'green', 'blue'], alpha=0.7)
        plt.ylabel('Recall (%)')
        plt.title('Emotion-wise Recognition Performance')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, emotion_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('figures/emotion_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Confusion Matrix
    if confusion_df is not None and not confusion_df.empty:
        # Try to extract numeric confusion matrix
        numeric_cols = confusion_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            confusion_matrix = confusion_df[numeric_cols].values
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Disgust', 'Happiness', 'Repression', 'Surprise'],
                       yticklabels=['Disgust', 'Happiness', 'Repression', 'Surprise'])
            plt.xlabel('Predicted Emotion')
            plt.ylabel('True Emotion')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("✅ All visualizations saved to 'figures/' directory")
    print("   - subject_performance.png")
    print("   - performance_distribution.png") 
    print("   - emotion_performance.png")
    print("   - confusion_matrix.png")

if __name__ == "__main__":
    create_visualizations()

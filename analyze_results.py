#!/usr/bin/env python3
"""
Detailed analysis of LOSO results for research publication
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_results():
    # Load results
    results_df = pd.read_csv('results/loso_hybrid_attention_results.csv')
    confusion_df = pd.read_csv('results/loso_hybrid_attention_confusion_matrix.csv')
    recall_df = pd.read_csv('results/loso_hybrid_attention_per_class_recall.csv')

    print('📊 DETAILED PERFORMANCE ANALYSIS')
    print('='*50)

    # Subject-wise analysis
    print('\n🎯 TOP 5 PERFORMING SUBJECTS:')
    top_subjects = results_df.nlargest(5, 'test_uar')
    for _, row in top_subjects.iterrows():
        print(f'  {row["subject_id"]}: Test UAR = {row["test_uar"]:.1f}%')

    print('\n⚠️  CHALLENGING SUBJECTS:')
    bottom_subjects = results_df.nsmallest(5, 'test_uar')
    for _, row in bottom_subjects.iterrows():
        print(f'  {row["subject_id"]}: Test UAR = {row["test_uar"]:.1f}%')

    # Performance distribution
    print('\n📈 PERFORMANCE DISTRIBUTION:')
    uar_values = results_df['test_uar'].values
    print(f'  Mean: {np.mean(uar_values):.1f}%')
    print(f'  Std Dev: {np.std(uar_values):.1f}%')
    print(f'  Median: {np.median(uar_values):.1f}%')
    print(f'  Range: {np.min(uar_values):.1f}% - {np.max(uar_values):.1f}%')

    # Class-wise performance
    if 'disgust' in recall_df.columns:
        print('\n🎭 EMOTION-WISE PERFORMANCE:')
        for emotion in ['disgust', 'happiness', 'repression', 'surprise']:
            if emotion in recall_df.columns:
                avg_recall = recall_df[emotion].mean()
                print(f'  {emotion.capitalize():12}: {avg_recall:.1f}%')

    # Generate performance summary
    print('\n📋 RESEARCH PAPER SUMMARY:')
    print(f'  • Mean Test UAR: {np.mean(uar_values):.1f}% ± {np.std(uar_values):.1f}%')
    print(f'  • Best Test UAR: {np.max(uar_values):.1f}%')
    print(f'  • Subjects: {len(results_df)}')
    print(f'  • ROI Attention: Enabled')
    print(f'  • Multi-Temporal Flow: 4 windows')

if __name__ == "__main__":
    analyze_results()

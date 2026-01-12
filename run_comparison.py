#!/usr/bin/env python3
"""
Run comparison with all model types for research paper
"""

import subprocess
import sys
from pathlib import Path

def run_all_models():
    """Run LOSO training for all model types"""
    
    models = ['temporal', 'gcn', 'hybrid', 'hybrid_attention']
    results = {}
    
    for model_type in models:
        print(f"\n🚀 Running {model_type} model...")
        print("="*50)
        
        # Run training
        cmd = [
            sys.executable, 
            'production_loso_training.py', 
            '--model_type', model_type
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            if result.returncode == 0:
                print(f"✅ {model_type} completed successfully")
                results[model_type] = 'SUCCESS'
            else:
                print(f"❌ {model_type} failed: {result.stderr}")
                results[model_type] = 'FAILED'
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_type} timed out")
            results[model_type] = 'TIMEOUT'
    
    # Summary
    print("\n" + "="*50)
    print("📊 COMPARISON SUMMARY")
    for model_type, status in results.items():
        print(f"  {model_type:15}: {status}")
    
    return results

if __name__ == "__main__":
    run_all_models()

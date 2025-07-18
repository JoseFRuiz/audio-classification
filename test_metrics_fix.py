#!/usr/bin/env python3
"""
Test script to verify that metrics are being logged correctly
"""

import os
import sys
import pandas as pd
import numpy as np

def check_metrics_file(metrics_dir):
    """Check if metrics.csv exists and contains the expected columns"""
    metrics_file = os.path.join(metrics_dir, "metrics", "metrics.csv")
    
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        return False
    
    print(f"✅ Found metrics file: {metrics_file}")
    
    # Read the metrics
    try:
        df = pd.read_csv(metrics_file)
        print(f"✅ Successfully read metrics file with {len(df)} rows")
        
        # Check for expected columns
        expected_columns = [
            'epoch', 'train_loss', 'train_loss_epoch', 'val_loss',
            'val_f1', 'val_map', 'val_auc',
            'train_loss_eval', 'train_f1_eval', 'train_map_eval', 'train_auc_eval'
        ]
        
        print("\n📊 Available columns:")
        for col in df.columns:
            if col in expected_columns:
                print(f"  ✅ {col}")
            else:
                print(f"  📝 {col}")
        
        # Check for missing expected columns
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"\n❌ Missing expected columns: {missing_columns}")
        else:
            print(f"\n✅ All expected columns are present!")
        
        # Show sample data
        print(f"\n📋 Sample data (first 5 rows):")
        print(df.head())
        
        # Check for non-NaN values in key metrics
        key_metrics = ['val_loss', 'val_f1', 'val_map', 'val_auc']
        for metric in key_metrics:
            if metric in df.columns:
                non_nan_count = df[metric].notna().sum()
                print(f"  {metric}: {non_nan_count}/{len(df)} non-NaN values")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading metrics file: {str(e)}")
        return False

def main():
    # Check the most recent experiment directory
    experiment_dirs = [d for d in os.listdir('.') if d.startswith('gru_') and os.path.isdir(d)]
    
    if not experiment_dirs:
        print("❌ No experiment directories found")
        return
    
    # Sort by creation time (most recent first)
    experiment_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    latest_dir = experiment_dirs[0]
    
    print(f"🔍 Checking latest experiment directory: {latest_dir}")
    
    # Check if it has metrics
    metrics_dir = os.path.join(latest_dir, "metrics")
    if os.path.exists(metrics_dir):
        check_metrics_file(latest_dir)
    else:
        print(f"❌ No metrics directory found in {latest_dir}")
        
        # Check if there are any metrics directories in subdirectories
        for root, dirs, files in os.walk(latest_dir):
            if "metrics" in dirs:
                metrics_path = os.path.join(root, "metrics")
                print(f"🔍 Found metrics directory: {metrics_path}")
                check_metrics_file(root)
                break

if __name__ == "__main__":
    main() 
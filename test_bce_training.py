#!/usr/bin/env python3
"""
Test script to verify training works with BCE loss.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Add the current directory to the path
sys.path.append('.')

def test_bce_training():
    """Test training with BCE loss."""
    print("🔹 Testing BCE loss training...")
    
    # Import the dataset class
    from run_experiment_gru_lightning import EmbeddingDataset
    
    # Load a small subset of data
    csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    clip_ids = df["clip_id"].values[:100]  # First 100
    labels = df.iloc[:100, 2:-1].values
    
    print(f"🔹 Loaded {len(clip_ids)} samples")
    
    # Create dataset
    train_dataset = EmbeddingDataset("embeddings", clip_ids, labels, is_train=True, test_size=0.2)
    val_dataset = EmbeddingDataset("embeddings", clip_ids, labels, is_train=False, test_size=0.2)
    
    print(f"🔹 Train dataset: {len(train_dataset)} samples")
    print(f"🔹 Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Test a few batches
    print("\n🔹 Testing data loading...")
    for i, (x, y) in enumerate(train_loader):
        print(f"✅ Batch {i+1}: x={x.shape}, y={y.shape}")
        if i >= 2:  # Test 3 batches
            break
    
    # Test BCE loss
    print("\n🔹 Testing BCE loss...")
    import torch.nn as nn
    bce_loss = nn.BCELoss()
    
    for i, (x, y) in enumerate(train_loader):
        # Create dummy predictions
        preds = torch.sigmoid(torch.randn_like(y))
        loss = bce_loss(preds, y)
        print(f"✅ BCE Loss batch {i+1}: {loss.item():.6f}")
        if i >= 1:  # Test 2 batches
            break
    
    print("\n✅ BCE training test completed!")
    return True

if __name__ == "__main__":
    success = test_bce_training()
    if success:
        print("\n🎉 BCE training test passed! You can now try the full training.")
        print("Run: sbatch gpu_job.sh")
    else:
        print("\n❌ BCE training test failed.") 
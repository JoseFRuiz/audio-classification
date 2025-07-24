#!/usr/bin/env python3
"""
Diagnose data issues before training
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from run_experiment_gru_lightning import EmbeddingDataset

def diagnose_data():
    print("🔍 Diagnosing data issues...")
    
    # Load the CSV
    csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    print(f"📊 Loading CSV from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    clip_ids = df["clip_id"].values
    labels = df.iloc[:, 2:-1].values
    
    print(f"📈 CSV loaded: {len(clip_ids)} clips, {labels.shape[1]} labels")
    
    # Check labels
    print(f"\n🏷️  Label statistics:")
    print(f"   Shape: {labels.shape}")
    print(f"   Range: [{labels.min():.4f}, {labels.max():.4f}]")
    print(f"   Sum: {labels.sum():.0f}")
    print(f"   Total elements: {labels.size}")
    print(f"   Non-zero elements: {(labels > 0).sum()}")
    print(f"   Sparsity: {100 * (labels == 0).sum() / labels.size:.2f}%")
    
    # Check for data corruption
    if labels.sum() > labels.size:
        print("❌ ERROR: Label sum exceeds total elements (data corruption)!")
        return
    
    if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
        print("❌ ERROR: Labels contain NaN or Inf values!")
        return
    
    # Check embeddings
    embedding_dir = "embeddings"
    print(f"\n🔍 Checking embeddings in: {embedding_dir}")
    
    if not os.path.exists(embedding_dir):
        print(f"❌ Embedding directory not found: {embedding_dir}")
        return
    
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    print(f"📁 Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        print("❌ No embedding files found!")
        return
    
    # Check a few embedding files
    print(f"\n🔍 Checking sample embeddings:")
    for i, clip_id in enumerate(clip_ids[:5]):
        embedding_path = os.path.join(embedding_dir, f"{clip_id}.npy")
        if os.path.exists(embedding_path):
            emb = np.load(embedding_path)
            print(f"   {clip_id}: shape={emb.shape}, range=[{emb.min():.4f}, {emb.max():.4f}]")
        else:
            print(f"   {clip_id}: ❌ Not found")
    
    # Create datasets and check
    print(f"\n🔍 Creating datasets...")
    
    # Filter to valid embeddings
    valid_indices = []
    valid_clip_ids = []
    valid_labels = []
    
    for idx, (clip_id, label) in enumerate(zip(clip_ids, labels)):
        embedding_path = os.path.join(embedding_dir, f"{clip_id}.npy")
        if os.path.exists(embedding_path):
            valid_indices.append(idx)
            valid_clip_ids.append(clip_id)
            valid_labels.append(label)
    
    print(f"✅ Valid clips: {len(valid_clip_ids)}/{len(clip_ids)}")
    
    if len(valid_clip_ids) == 0:
        print("❌ No valid embedding files found!")
        return
    
    # Create train/test split
    indices = np.arange(len(valid_clip_ids))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * 0.9)  # 90% train, 10% test
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"📊 Train split: {len(train_indices)}, Validation split: {len(val_indices)}")
    
    # Create datasets
    train_dataset = EmbeddingDataset(embedding_dir, valid_clip_ids, valid_labels, indices=train_indices, is_train=True, test_size=0.0)
    val_dataset = EmbeddingDataset(embedding_dir, valid_clip_ids, valid_labels, indices=val_indices, is_train=False, test_size=0.0)
    
    # Check a few samples
    print(f"\n🔍 Checking dataset samples:")
    
    # Training samples
    for i in range(min(3, len(train_dataset))):
        x, y = train_dataset[i]
        print(f"   Train sample {i}: x shape={x.shape}, y shape={y.shape}")
        print(f"      x range: [{x.min():.4f}, {x.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"      y sum: {y.sum().item()}, y total: {y.numel()}")
    
    # Validation samples
    for i in range(min(3, len(val_dataset))):
        x, y = val_dataset[i]
        print(f"   Val sample {i}: x shape={x.shape}, y shape={y.shape}")
        print(f"      x range: [{x.min():.4f}, {x.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"      y sum: {y.sum().item()}, y total: {y.numel()}")
    
    # Check overall statistics
    print(f"\n📊 Overall dataset statistics:")
    
    # Training set
    train_labels = []
    for i in range(min(100, len(train_dataset))):  # Sample first 100
        _, y = train_dataset[i]
        train_labels.append(y.numpy())
    train_labels = np.array(train_labels)
    
    print(f"   Training labels (sample): shape={train_labels.shape}")
    print(f"      Range: [{train_labels.min():.4f}, {train_labels.max():.4f}]")
    print(f"      Sum: {train_labels.sum():.0f}")
    print(f"      Non-zero: {(train_labels > 0).sum()}")
    
    # Validation set
    val_labels = []
    for i in range(min(100, len(val_dataset))):  # Sample first 100
        _, y = val_dataset[i]
        val_labels.append(y.numpy())
    val_labels = np.array(val_labels)
    
    print(f"   Validation labels (sample): shape={val_labels.shape}")
    print(f"      Range: [{val_labels.min():.4f}, {val_labels.max():.4f}]")
    print(f"      Sum: {val_labels.sum():.0f}")
    print(f"      Non-zero: {(val_labels > 0).sum()}")
    
    print(f"\n✅ Data diagnosis complete!")

if __name__ == "__main__":
    diagnose_data() 
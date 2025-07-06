#!/usr/bin/env python3
"""
Quick test to verify the fixes work.
"""

import os
import sys
import torch

# Test GPU compatibility
print("🔹 Testing GPU compatibility...")
if torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        test_tensor = torch.tensor([1.0], device=device)
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        
        # Test GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU memory: {gpu_memory:.1f} GB")
        
        # Test tensor operations
        x = torch.randn(100, 768).to(device)
        y = torch.randn(100, 200).to(device)
        z = torch.mm(x, x.t())
        print(f"✅ GPU tensor operations work")
        
    except Exception as e:
        print(f"❌ GPU test failed: {str(e)}")
else:
    print("⚠️ No GPU available")

# Test dataset loading
print("\n🔹 Testing dataset loading...")
try:
    from run_experiment_gru_lightning import EmbeddingDataset
    import pandas as pd
    import numpy as np
    
    # Load a small subset
    csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        clip_ids = df["clip_id"].values[:100]  # First 100
        labels = df.iloc[:100, 2:-1].values
        
        # Create dataset
        train_dataset = EmbeddingDataset("embeddings", clip_ids, labels, is_train=True, test_size=0.2)
        print(f"✅ Dataset created: {len(train_dataset)} samples")
        
        # Test loading
        sample_x, sample_y = train_dataset[0]
        print(f"✅ Sample loaded: x={sample_x.shape}, y={sample_y.shape}")
        
    else:
        print("⚠️ CSV file not found")
        
except Exception as e:
    print(f"❌ Dataset test failed: {str(e)}")

print("\n✅ Quick test completed!") 
#!/usr/bin/env python3
"""
Test script to verify the training setup works with existing embeddings.
"""

import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add the current directory to the path so we can import the modules
sys.path.append('.')

def test_training_setup():
    """Test the training setup with existing embeddings."""
    print("🔹 Testing training setup with existing embeddings...")
    
    # Check if embeddings folder exists
    embeddings_dir = "embeddings"
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        return False
    
    # Count embedding files
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
    print(f"🔹 Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        print("❌ No embedding files found")
        return False
    
    # Check if CSV file exists
    csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        clip_ids = df["clip_id"].values
        labels = df.iloc[:, 2:-1].values
        print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"✅ Labels shape: {labels.shape}")
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        return False
    
    # Test the dataset creation logic (without Wav2Vec model)
    try:
        from run_experiment_gru_lightning import EmbeddingDataset
        
        # Test with a small subset
        test_clip_ids = clip_ids[:1000]  # Use first 1000
        test_labels = labels[:1000]
        
        # First, filter clip_ids to only include those with embedding files
        valid_indices = []
        valid_clip_ids = []
        valid_labels = []
        
        for idx, (clip_id, label) in enumerate(zip(test_clip_ids, test_labels)):
            embedding_path = os.path.join(embeddings_dir, f"{clip_id}.npy")
            if os.path.exists(embedding_path):
                valid_indices.append(idx)
                valid_clip_ids.append(clip_id)
                valid_labels.append(label)
        
        if len(valid_clip_ids) == 0:
            print("❌ No valid clip IDs found in test subset")
            return False
        
        print(f"🔹 Valid clip IDs in test subset: {len(valid_clip_ids)}")
        
        # Create train/test split on the filtered data
        indices = np.arange(len(valid_clip_ids))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.8)  # 80% train, 20% val
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        print(f"🔹 Test train split size: {len(train_indices)}")
        print(f"🔹 Test validation split size: {len(val_indices)}")
        
        # Test dataset creation with pre-computed indices
        train_dataset = EmbeddingDataset(embeddings_dir, valid_clip_ids, valid_labels, indices=train_indices, is_train=True)
        val_dataset = EmbeddingDataset(embeddings_dir, valid_clip_ids, valid_labels, indices=val_indices, is_train=False)
        
        print(f"✅ Train dataset size: {len(train_dataset)}")
        print(f"✅ Validation dataset size: {len(val_dataset)}")
        
        # Test loading a sample
        sample_x, sample_y = train_dataset[0]
        print(f"✅ Sample shape: x={sample_x.shape}, y={sample_y.shape}")
        
        # Test dataloader creation
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        print(f"✅ Train loader batches: {len(train_loader)}")
        print(f"✅ Val loader batches: {len(val_loader)}")
        
        # Test batch loading
        for batch_x, batch_y in train_loader:
            print(f"✅ Batch shape: x={batch_x.shape}, y={batch_y.shape}")
            break
            
        print("✅ Dataset creation test passed!")
        
    except Exception as e:
        print(f"❌ Dataset creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Training setup test passed!")
    return True

if __name__ == "__main__":
    success = test_training_setup()
    if success:
        print("\n🎉 All tests passed! The training setup should work correctly.")
        print("\nYou can now try running the training script:")
        print("python run_experiment_gru_lightning.py --save_dir 'test_run' --epochs 2 --eval_interval 1 --lr 1e-3 --batch_size 32 --test_size 0.1 --dropout 0.1 --loss_fn 'bce'")
    else:
        print("\n❌ Tests failed. Please check the setup.") 
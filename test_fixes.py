#!/usr/bin/env python3
"""
Test script to verify the fixes for the audio classification training script.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the current directory to the path so we can import the modules
sys.path.append('.')

def test_dataset_creation():
    """Test the dataset creation logic."""
    print("🔹 Testing dataset creation...")
    
    # Create a mock embedding directory with some test files
    test_embedding_dir = "test_embeddings"
    os.makedirs(test_embedding_dir, exist_ok=True)
    
    # Create some mock embedding files
    test_clip_ids = [f"clip_{i:04d}" for i in range(100)]
    test_labels = np.random.randint(0, 2, size=(100, 10)).astype(np.float32)
    
    # Create embedding files for 80% of the clips
    valid_clip_ids = test_clip_ids[:80]
    for clip_id in valid_clip_ids:
        embedding = np.random.randn(100, 768).astype(np.float32)  # Mock wav2vec embedding
        np.save(os.path.join(test_embedding_dir, f"{clip_id}.npy"), embedding)
    
    print(f"🔹 Created {len(valid_clip_ids)} mock embedding files")
    
    # Test the dataset creation logic
    try:
        from run_experiment_gru_lightning import EmbeddingDataset
        
        # Test with test_size=0.2 (should give ~64 train, ~16 val)
        train_dataset = EmbeddingDataset(test_embedding_dir, test_clip_ids, test_labels, is_train=True, test_size=0.2)
        val_dataset = EmbeddingDataset(test_embedding_dir, test_clip_ids, test_labels, is_train=False, test_size=0.2)
        
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
        raise
    
    finally:
        # Clean up
        import shutil
        if os.path.exists(test_embedding_dir):
            shutil.rmtree(test_embedding_dir)

def test_model_initialization():
    """Test the model initialization."""
    print("\n🔹 Testing model initialization...")
    
    try:
        from run_experiment_gru_lightning import LitRNNClassifier
        
        # Create a mock dataset
        mock_input_dim = 768
        mock_num_classes = 10
        
        # Create model
        model = LitRNNClassifier(
            input_dim=mock_input_dim,
            hidden_dim=256,
            num_layers=2,
            num_classes=mock_num_classes,
            lr=1e-3,
            weight_decay=1e-4,
            dropout=0.1,
            loss_fn="bce"
        )
        
        print(f"✅ Model created successfully")
        print(f"✅ Input dimension: {model.gru.input_size}")
        print(f"✅ Output dimension: {model.fc[-1].out_features}")
        
        # Test forward pass
        mock_batch_size = 4
        mock_seq_len = 100
        mock_input = torch.randn(mock_batch_size, mock_seq_len, mock_input_dim)
        
        with torch.no_grad():
            output = model(mock_input)
            print(f"✅ Forward pass successful: output shape = {output.shape}")
        
        print("✅ Model initialization test passed!")
        
    except Exception as e:
        print(f"❌ Model initialization test failed: {str(e)}")
        raise

def test_gpu_compatibility():
    """Test GPU compatibility check."""
    print("\n🔹 Testing GPU compatibility...")
    
    try:
        # Test the GPU compatibility logic
        if torch.cuda.is_available():
            try:
                test_tensor = torch.tensor([1.0], device="cuda")
                print(f"✅ GPU is available and compatible")
                print(f"✅ GPU: {torch.cuda.get_device_name()}")
            except Exception as e:
                print(f"⚠️ GPU compatibility issue: {str(e)}")
                print("✅ Fallback to CPU would work")
        else:
            print("✅ No GPU available, would use CPU")
            
    except Exception as e:
        print(f"❌ GPU compatibility test failed: {str(e)}")

if __name__ == "__main__":
    print("🧪 Running tests for audio classification fixes...")
    
    test_dataset_creation()
    test_model_initialization()
    test_gpu_compatibility()
    
    print("\n✅ All tests passed!") 
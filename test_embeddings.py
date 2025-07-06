#!/usr/bin/env python3
"""
Test script to verify that embeddings can be loaded correctly.
"""

import os
import numpy as np
import pandas as pd

def test_embeddings():
    """Test loading embeddings from the embeddings folder."""
    print("🔹 Testing embedding loading...")
    
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
    
    # Test loading a few embedding files
    print("🔹 Testing embedding file loading...")
    for i, filename in enumerate(embedding_files[:5]):  # Test first 5 files
        try:
            embedding_path = os.path.join(embeddings_dir, filename)
            embedding = np.load(embedding_path)
            print(f"✅ {filename}: shape = {embedding.shape}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
            return False
    
    # Check if CSV file exists
    csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    if os.path.exists(csv_path):
        print(f"✅ CSV file found: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"✅ Clip IDs column: {df['clip_id'].head().tolist()}")
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            return False
    else:
        print(f"⚠️ CSV file not found: {csv_path}")
    
    # Test matching clip IDs with embedding files
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        clip_ids = df["clip_id"].values
        
        # Check how many clip IDs have corresponding embedding files
        matching_count = 0
        for clip_id in clip_ids[:100]:  # Check first 100
            embedding_path = os.path.join(embeddings_dir, f"{clip_id}.npy")
            if os.path.exists(embedding_path):
                matching_count += 1
        
        print(f"🔹 Clip ID matching test: {matching_count}/100 clip IDs have embeddings")
        
        if matching_count == 0:
            print("❌ No clip IDs match embedding files")
            return False
        elif matching_count < 50:
            print("⚠️ Warning: Less than 50% of clip IDs have embeddings")
        else:
            print("✅ Good matching between clip IDs and embedding files")
    
    print("✅ Embedding test passed!")
    return True

if __name__ == "__main__":
    success = test_embeddings()
    if success:
        print("\n🎉 All tests passed! The embeddings should work correctly.")
    else:
        print("\n❌ Tests failed. Please check the embedding files and paths.") 
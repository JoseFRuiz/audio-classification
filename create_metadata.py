#!/usr/bin/env python3
"""
Script to create metadata.json for existing embeddings.
"""

import os
import json
import numpy as np
import pandas as pd

def create_metadata():
    """Create metadata.json for existing embeddings."""
    print("🔹 Creating metadata for existing embeddings...")
    
    embeddings_dir = "embeddings"
    csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    
    # Count embedding files
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
    print(f"🔹 Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        print("❌ No embedding files found")
        return False
    
    # Load CSV to get label information
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        labels = df.iloc[:, 2:-1].values
        print(f"🔹 CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"🔹 Labels shape: {labels.shape}")
    else:
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    # Get embedding shape from first file
    first_embedding_path = os.path.join(embeddings_dir, embedding_files[0])
    try:
        first_embedding = np.load(first_embedding_path)
        embedding_shape = first_embedding.shape
        print(f"🔹 Embedding shape: {embedding_shape}")
    except Exception as e:
        print(f"❌ Error loading first embedding: {str(e)}")
        return False
    
    # Create metadata
    metadata = {
        "total_samples": len(embedding_files),
        "embedding_shape": embedding_shape,
        "label_shape": labels.shape[1:],
        "embedding_files": embedding_files[:10]  # Store first 10 filenames as example
    }
    
    # Save metadata
    metadata_path = os.path.join(embeddings_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_path}")
    print(f"✅ Total samples: {metadata['total_samples']}")
    print(f"✅ Embedding shape: {metadata['embedding_shape']}")
    print(f"✅ Label shape: {metadata['label_shape']}")
    
    return True

if __name__ == "__main__":
    success = create_metadata()
    if success:
        print("\n🎉 Metadata creation successful!")
    else:
        print("\n❌ Metadata creation failed.") 
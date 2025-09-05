#!/usr/bin/env python3
"""
Quick script to create BirdCLEF CSV from existing downloaded data.
"""

import os
import pandas as pd
from pathlib import Path

def create_birdclef_csv(audio_dir, metadata_file, output_csv):
    """Create CSV file for BirdCLEF dataset."""
    print(f"�� Creating CSV from: {audio_dir}")
    print(f"🔹 Using metadata: {metadata_file}")
    
    # Load metadata
    print("🔹 Loading metadata...")
    metadata_df = pd.read_csv(metadata_file)
    print(f"🔹 Metadata shape: {metadata_df.shape}")
    print(f"🔹 Metadata columns: {list(metadata_df.columns)}")
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
        audio_files.extend(Path(audio_dir).rglob(ext))
    
    print(f"🔹 Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print(f"❌ No audio files found in {audio_dir}")
        return None
    
    # Create mapping from filename to species
    filename_to_species = {}
    for _, row in metadata_df.iterrows():
        filename = row.get('filename', row.get('id', ''))
        species = row.get('primary_label', row.get('species', ''))
        if filename and species:
            # Remove file extension for matching
            filename_no_ext = Path(filename).stem
            filename_to_species[filename_no_ext] = species
    
    print(f"🔹 Created mapping for {len(filename_to_species)} species")
    
    # Get unique species
    unique_species = sorted(list(set(filename_to_species.values())))
    print(f"🔹 Found {len(unique_species)} unique species")
    
    # Create DataFrame
    data = []
    for audio_file in audio_files:
        clip_id = audio_file.stem
        species = filename_to_species.get(clip_id, 'Unknown')
        
        # Create one-hot encoded labels
        labels = [1 if species == s else 0 for s in unique_species]
        
        data.append({
            'clip_id': clip_id,
            'species': species,
            **{species: label for species, label in zip(unique_species, labels)}
        })
    
    df = pd.DataFrame(data)
    
    # Filter out unknown species
    df_filtered = df[df['species'] != 'Unknown']
    
    if len(df_filtered) < len(df):
        print(f"⚠️ Warning: {len(df) - len(df_filtered)} files had unknown species and were filtered out")
    
    if len(df_filtered) == 0:
        print(f"❌ Error: No valid files with species labels found")
        return None
    
    # Remove the species column for the final CSV (keep only clip_id and one-hot labels)
    df_final = df_filtered.drop('species', axis=1)
    df_final.to_csv(output_csv, index=False)
    
    print(f"✅ Created CSV with {len(df_final)} entries: {output_csv}")
    
    # Print species distribution
    species_counts = df_filtered['species'].value_counts()
    print(f"🔹 Top 10 species by count:")
    for species, count in species_counts.head(10).items():
        print(f"   {species}: {count}")
    
    return df_final

if __name__ == "__main__":
    audio_dir = "./birdclef_data/train_audio"
    metadata_file = "./birdclef_data/train_metadata.csv"
    output_csv = "./birdclef_data/birdclef_2023_dataset.csv"
    
    print("🔹 BirdCLEF CSV Creator")
    print("=" * 50)
    
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        exit(1)
    
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        exit(1)
    
    df = create_birdclef_csv(audio_dir, metadata_file, output_csv)
    
    if df is not None:
        print(f"\n🚀 CSV created successfully!")
        print(f"   You can now run your experiment with:")
        print(f"   python run_birdclef_experiment.py --audio_dir {audio_dir} --csv_path {output_csv} --save_dir birdclef_gru_001 --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.2 --dropout 0.1 --loss_fn wu_auc --num_workers 4")
    else:
        print(f"\n❌ Failed to create CSV. Check the error messages above.")
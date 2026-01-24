#!/usr/bin/env python3
"""
Script to load a trained model checkpoint and generate predictions for the validation dataset.
Generates two CSV files: one with actual labels and one with predicted scores.

Example usage:
python generate_predictions.py "wav2vec_032"
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import DataLoader, Dataset
import multiprocessing

# Import the classes from our separate module
from model_classes import EmbeddingDataset, RawAudioDataset, LitRNNClassifier

def load_experiment_config(experiment_dir):
    """Load the configuration from args.json in the experiment directory."""
    args_path = os.path.join(experiment_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {experiment_dir}")
    
    with open(args_path, 'r') as f:
        config = json.load(f)
    
    return config

def create_validation_dataset(config, csv_path, embedding_dir=None, audio_dir=None):
    """Create validation dataset using the same configuration as the original experiment."""
    print(f"🔹 Loading CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    clip_ids = df["clip_id"].values
    labels = df.iloc[:, 2:-1].values  # Skip clip_id and duration columns
    
    print(f"🔹 Number of clips in CSV: {len(clip_ids)}")
    print(f"🔹 Feature extraction mode: {config['feature_mode']}")
    
    if config['feature_mode'] == "wav2vec":
        if embedding_dir is None:
            embedding_dir = config.get('embedding_dir', 'embeddings')
        
        print(f"🔹 Using embedding directory: {embedding_dir}")
        
        # Filter clip_ids to only include those with embedding files
        valid_indices = []
        valid_clip_ids = []
        valid_labels = []
        
        for idx, (clip_id, label) in enumerate(zip(clip_ids, labels)):
            embedding_path = os.path.join(embedding_dir, f"{clip_id}.npy")
            if os.path.exists(embedding_path):
                valid_indices.append(idx)
                valid_clip_ids.append(clip_id)
                valid_labels.append(label)
        
        if len(valid_clip_ids) == 0:
            raise ValueError(f"No valid embedding files found in {embedding_dir}")
        
        print(f"🔹 Valid clip_ids: {len(valid_clip_ids)}")
        
        # Create train/test split using the same random seed as the original experiment
        indices = np.arange(len(valid_clip_ids))
        np.random.seed(42)  # Same seed as original experiment
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - config['test_size']))
        
        val_indices = indices[split_idx:]
        print(f"🔹 Validation split size: {len(val_indices)}")
        
        # Create validation dataset
        val_dataset = EmbeddingDataset(
            embedding_dir, 
            valid_clip_ids, 
            valid_labels, 
            indices=val_indices, 
            is_train=False, 
            test_size=0.0
        )
        
    else:  # raw audio mode
        if audio_dir is None:
            audio_dir = config.get('audio_dir', '../tmp/fsd50k/FSD50K.dev_audio')
        
        print(f"🔹 Using audio directory: {audio_dir}")
        
        # Filter for available audio files
        valid_indices = []
        valid_clip_ids = []
        valid_labels = []
        
        for idx, (clip_id, label) in enumerate(zip(clip_ids, labels)):
            # Try different audio formats
            audio_found = False
            for ext in ['wav', 'mp3', 'flac', 'ogg']:
                audio_path = os.path.join(audio_dir, f"{clip_id}.{ext}")
                if os.path.exists(audio_path):
                    valid_indices.append(idx)
                    valid_clip_ids.append(clip_id)
                    valid_labels.append(label)
                    audio_found = True
                    break
        
        print(f"🔹 Found {len(valid_clip_ids)} audio files out of {len(clip_ids)} total clips")
        
        if len(valid_clip_ids) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        # Create train/test split using the same random seed as the original experiment
        indices = np.arange(len(valid_clip_ids))
        np.random.seed(42)  # Same seed as original experiment
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - config['test_size']))
        
        val_indices = indices[split_idx:]
        print(f"🔹 Validation split size: {len(val_indices)}")
        
        # Create validation dataset
        val_dataset = RawAudioDataset(
            audio_dir, 
            valid_clip_ids, 
            valid_labels, 
            indices=val_indices, 
            is_train=False, 
            test_size=0.0,
            window_size=config.get('window_size', 1024),
            hop_size=config.get('hop_size', 512)
        )
    
    return val_dataset

def load_model_from_checkpoint(experiment_dir, val_dataset):
    """Load the trained model from the best checkpoint."""
    checkpoint_path = os.path.join(experiment_dir, "best-checkpoint.ckpt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"🔹 Loading model from: {checkpoint_path}")
    
    # Load the model
    model = LitRNNClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"🔹 Model loaded on device: {device}")
    print(f"🔹 Model input dimension: {model.gru.input_size}")
    print(f"🔹 Model output dimension: {model.fc[-1].out_features}")
    
    return model, device

def generate_predictions(model, val_dataset, device, batch_size=32, num_workers=1):
    """Generate predictions for the validation dataset."""
    print(f"🔹 Generating predictions with batch_size={batch_size}")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    all_predictions = []
    all_targets = []
    all_clip_ids = []
    
    print(f"🔹 Processing {len(val_dataset)} validation samples...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            
            # Get predictions
            preds = model(x)
            
            # Convert to CPU and store
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            # Get clip_ids for this batch
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(val_dataset))
            batch_clip_ids = [val_dataset.clip_ids[val_dataset.indices[i]] for i in range(batch_start, batch_end)]
            all_clip_ids.extend(batch_clip_ids)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"🔹 Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"🔹 Generated predictions shape: {all_predictions.shape}")
    print(f"🔹 Generated targets shape: {all_targets.shape}")
    print(f"🔹 Number of clip_ids: {len(all_clip_ids)}")
    
    return all_predictions, all_targets, all_clip_ids

def save_predictions_to_csv(predictions, targets, clip_ids, experiment_dir, class_names=None):
    """Save predictions and targets to CSV files."""
    
    # Create DataFrames
    if class_names is None:
        # Generate generic class names if not provided
        num_classes = predictions.shape[1]
        class_names = [f"class_{i:03d}" for i in range(num_classes)]
    
    # Create predictions DataFrame
    pred_data = {'clip_id': clip_ids}
    for i, class_name in enumerate(class_names):
        pred_data[class_name] = predictions[:, i]
    pred_df = pd.DataFrame(pred_data)
    
    # Create targets DataFrame
    target_data = {'clip_id': clip_ids}
    for i, class_name in enumerate(class_names):
        target_data[class_name] = targets[:, i].astype(int)
    target_df = pd.DataFrame(target_data)
    
    # Save to CSV files
    pred_csv_path = os.path.join(experiment_dir, "validation_predictions.csv")
    target_csv_path = os.path.join(experiment_dir, "validation_targets.csv")
    
    pred_df.to_csv(pred_csv_path, index=False)
    target_df.to_csv(target_csv_path, index=False)
    
    print(f"✅ Predictions saved to: {pred_csv_path}")
    print(f"✅ Targets saved to: {target_csv_path}")
    
    # Print some statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   - Number of samples: {len(predictions)}")
    print(f"   - Number of classes: {predictions.shape[1]}")
    print(f"   - Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   - Target range: [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"   - Positive targets: {targets.sum()}/{targets.size} ({100*targets.sum()/targets.size:.2f}%)")
    
    return pred_csv_path, target_csv_path

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for validation dataset from trained model")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory (e.g., 'wav2vec_032')")
    parser.add_argument("--csv_path", type=str, default="../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv", 
                       help="Path to CSV file with labels")
    parser.add_argument("--embedding_dir", type=str, default="embeddings", 
                       help="Directory containing embeddings (for wav2vec mode)")
    parser.add_argument("--audio_dir", type=str, default="../tmp/fsd50k/FSD50K.dev_audio", 
                       help="Directory containing audio files (for raw mode)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--class_names", type=str, nargs='+', default=None, 
                       help="List of class names (if not provided, will use generic names)")
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print(f"🔹 Generating predictions for experiment: {args.experiment_dir}")
    
    # Check if experiment directory exists
    if not os.path.exists(args.experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {args.experiment_dir}")
    
    # Load experiment configuration
    config = load_experiment_config(args.experiment_dir)
    print(f"🔹 Loaded configuration for {config['feature_mode']} mode")
    
    # Create validation dataset
    val_dataset = create_validation_dataset(
        config, 
        args.csv_path, 
        args.embedding_dir, 
        args.audio_dir
    )
    
    # Load trained model
    model, device = load_model_from_checkpoint(args.experiment_dir, val_dataset)
    
    # Generate predictions
    predictions, targets, clip_ids = generate_predictions(
        model, 
        val_dataset, 
        device, 
        args.batch_size, 
        args.num_workers
    )
    
    # Save to CSV files
    pred_csv_path, target_csv_path = save_predictions_to_csv(
        predictions, 
        targets, 
        clip_ids, 
        args.experiment_dir, 
        args.class_names
    )
    
    print(f"\n✅ Successfully generated predictions for {len(predictions)} validation samples")
    print(f"🔹 Files saved in: {args.experiment_dir}")

if __name__ == "__main__":
    main()

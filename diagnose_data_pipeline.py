import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from run_experiment_gru_lightning import EmbeddingDataset, LitRNNClassifier
import argparse

def main():
    # Use the same arguments as your training script
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, default="embeddings")
    parser.add_argument("--csv_path", type=str, default="../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv")
    parser.add_argument("--audio_dir", type=str, default="../tmp/fsd50k/FSD50K.dev_audio")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Load CSV
    print(f"Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    clip_ids = df["clip_id"].values
    labels = df.iloc[:, 2:-1].values

    # Check label stats
    print("Label shape:", labels.shape)
    print("Label unique values:", np.unique(labels))
    print("Label sum (per class):", labels.sum(axis=0))
    print("Label sum (total):", labels.sum())

    # Check for all-zero or all-one labels
    if np.all(labels == 0):
        print("❌ All labels are zero!")
    if np.all(labels == 1):
        print("❌ All labels are one!")

    # Check embeddings
    embedding_dir = args.embedding_dir
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    print(f"Found {len(embedding_files)} embedding files in {embedding_dir}")

    # Filter to valid clip_ids
    valid_clip_ids = []
    valid_labels = []
    for clip_id, label in zip(clip_ids, labels):
        if os.path.exists(os.path.join(embedding_dir, f"{clip_id}.npy")):
            valid_clip_ids.append(clip_id)
            valid_labels.append(label)
    print(f"Valid clip_ids: {len(valid_clip_ids)}")

    # Create dataset and dataloader
    dataset = EmbeddingDataset(embedding_dir, valid_clip_ids, valid_labels, is_train=True, test_size=0.1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get a batch
    for x, y in loader:
        print("Batch embeddings shape:", x.shape)
        print("Batch embeddings min/max:", x.min().item(), x.max().item())
        print("Batch labels shape:", y.shape)
        print("Batch labels unique values:", torch.unique(y))
        print("Batch labels sum:", y.sum())
        break

    # Test model output
    input_dim = x.shape[2]
    num_classes = y.shape[1]
    model = LitRNNClassifier(input_dim=input_dim, hidden_dim=256, num_layers=2, num_classes=num_classes, lr=1e-3, weight_decay=1e-4, dropout=0.1)
    preds = model(x)
    print("Model output shape:", preds.shape)
    print("Model output min/max:", preds.min().item(), preds.max().item())

    # Check for constant output
    if torch.all(preds == preds[0]):
        print("❌ Model output is constant!")

if __name__ == "__main__":
    main() 
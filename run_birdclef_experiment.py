#!/usr/bin/env python3
"""
BirdCLEF Classification Script
Specialized version for BirdCLEF dataset with multilabel species classification.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm
from pytorch_lightning.loggers import CSVLogger
import json
from utils import preprocess_audio, extract_wav2vec_embeddings, SAMPLE_RATE, TARGET_LENGTH, asymmetric_loss, MeanContrastiveRankingLoss, wu_auc_loss, combined_wu_bce_loss, combined_wu_asymmetric_loss, combined_asymmetric_bce_loss
import multiprocessing
import time
import glob
from pathlib import Path

class BirdCLEFDataset(Dataset):
    """Dataset class specifically for BirdCLEF dataset."""
    
    def __init__(self, embedding_dir, clip_ids, labels, indices=None, is_train=True, test_size=0.2, random_state=42):
        self.embedding_dir = embedding_dir
        self.is_train = is_train
        
        print(f"🔹 BirdCLEF dataset: Looking for embeddings in {os.path.abspath(embedding_dir)}")
        
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
            raise ValueError(f"No embedding files found in {os.path.abspath(embedding_dir)}")
        
        self.clip_ids = np.array(valid_clip_ids)
        self.labels = np.array(valid_labels)
        
        # Use provided indices if available, otherwise create train/test split
        if indices is not None:
            self.indices = indices
        else:
            # Create train/test split indices
            indices = np.arange(len(self.clip_ids))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            split_idx = int(len(indices) * (1 - test_size))
            
            if is_train:
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]
        
        print(f"🔹 {'Training' if is_train else 'Validation'} dataset size: {len(self.indices)}")
        
        # Print class distribution
        if len(self.indices) > 0:
            class_counts = self.labels[self.indices].sum(axis=0)
            print(f"🔹 Number of classes: {len(class_counts)}")
            print(f"🔹 Total positive samples: {class_counts.sum()}")
            print(f"🔹 Average samples per class: {class_counts.mean():.1f}")
            print(f"🔹 Min samples per class: {class_counts.min()}")
            print(f"🔹 Max samples per class: {class_counts.max()}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        clip_idx = self.indices[idx]
        clip_id = self.clip_ids[clip_idx]
        label = self.labels[clip_idx]
        
        # Load embedding from file
        embedding_path = os.path.join(self.embedding_dir, f"{clip_id}.npy")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        embedding = np.load(embedding_path)
        
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class BirdCLEFClassifier(pl.LightningModule):
    """Lightning module for BirdCLEF classification."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, lr, weight_decay, dropout, 
                 loss_fn="bce", loss_margin=0.1, gamma_pos=0.0, gamma_neg=4.0, wu_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # GRU layers with dropout
        gru_dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=gru_dropout)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Initialize loss function
        if loss_fn == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_fn == "asymmetric":
            self.loss_fn = lambda preds, targets: asymmetric_loss(
                preds, targets, gamma_pos=gamma_pos, gamma_neg=gamma_neg, margin=0.05
            )
        elif loss_fn == "contrastive":
            self.loss_fn = MeanContrastiveRankingLoss(margin=loss_margin)
        elif loss_fn == "wu_auc":
            self.loss_fn = lambda preds, targets: wu_auc_loss(preds, targets, margin=loss_margin)
        elif loss_fn == "combined_wu_bce":
            self.loss_fn = lambda preds, targets: combined_wu_bce_loss(
                preds, targets, wu_weight=wu_weight, bce_weight=bce_weight, margin=loss_margin
            )
        elif loss_fn == "combined_wu_asymmetric":
            self.loss_fn = lambda preds, targets: combined_wu_asymmetric_loss(
                preds, targets, wu_weight=wu_weight, asymmetric_weight=1-wu_weight, 
                margin=loss_margin, gamma_pos=gamma_pos, gamma_neg=gamma_neg, asymmetric_margin=0.05
            )
        elif loss_fn == "combined_asymmetric_bce":
            self.loss_fn = lambda preds, targets: combined_asymmetric_bce_loss(
                preds, targets, asymmetric_weight=1-bce_weight, bce_weight=bce_weight,
                gamma_pos=gamma_pos, gamma_neg=gamma_neg, margin=0.05
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        # Metrics
        self.f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.map = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        
        self.training_step_outputs = []

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n[-1]  # Last hidden state
        output = self.fc(h_n)
        return output

    def _init_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        
        self.training_step_outputs.append(loss.item())
        
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        avg_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions and targets for epoch-end computation
        if not hasattr(self, 'val_preds'):
            self.val_preds = []
            self.val_targets = []
        
        self.val_preds.append(preds.detach())
        self.val_targets.append(y.detach())
        
        return loss

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_preds') and len(self.val_preds) > 0:
            try:
                all_preds = torch.cat(self.val_preds, dim=0)
                all_targets = torch.cat(self.val_targets, dim=0)
                
                # Apply sigmoid to raw logits
                all_preds_probs = torch.sigmoid(all_preds)
                all_preds_probs = torch.clamp(all_preds_probs, min=1e-7, max=1.0-1e-7)
                all_targets = all_targets.int()
                
                # Compute metrics
                val_f1 = self.f1(all_preds_probs, all_targets)
                val_map = self.map(all_preds_probs, all_targets)
                val_auc = self.auc(all_preds_probs, all_targets)
                
                # Log metrics
                self.log('val_f1', val_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_map', val_map, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_auc', val_auc, on_step=False, on_epoch=True, prog_bar=True)
                
                print(f"✅ Epoch {self.current_epoch}: val_f1={val_f1:.4f}, val_map={val_map:.4f}, val_auc={val_auc:.4f}")
                
            except Exception as e:
                print(f"⚠️ Warning: Error computing validation metrics: {str(e)}")
        
        # Clear stored predictions and targets
        if hasattr(self, 'val_preds'):
            self.val_preds.clear()
            self.val_targets.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

def main():
    parser = argparse.ArgumentParser(description="Train BirdCLEF classification model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--eval_interval", type=int, default=10, help="Interval for evaluation")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--save_dir", type=str, default="birdclef_results", help="Save directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="BirdCLEF audio directory")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file path")
    parser.add_argument("--embedding_dir", type=str, default="birdclef_embeddings", help="Embedding directory")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
    parser.add_argument("--loss_fn", type=str, default="bce", 
                       choices=["bce", "asymmetric", "contrastive", "wu_auc", "combined_wu_bce", "combined_wu_asymmetric", "combined_asymmetric_bce"],
                       help="Loss function")
    parser.add_argument("--loss_margin", type=float, default=0.1, help="Loss margin")
    parser.add_argument("--gamma_pos", type=float, default=0.0, help="Gamma positive")
    parser.add_argument("--gamma_neg", type=float, default=4.0, help="Gamma negative")
    parser.add_argument("--wu_weight", type=float, default=0.5, help="Wu weight")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="BCE weight")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"🔹 Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    print(f"🔹 Loading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    clip_ids = df["clip_id"].values
    labels = df.iloc[:, 1:].values  # Skip clip_id column
    
    print(f"🔹 Dataset: {len(clip_ids)} clips, {labels.shape[1]} species")
    
    # Load Wav2Vec2 model
    MODEL_NAME = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    wav2vec_model.eval()
    wav2vec_model.to(device)
    
    # Create embedding directory
    os.makedirs(args.embedding_dir, exist_ok=True)
    
    # Extract embeddings if needed
    embedding_files = [f for f in os.listdir(args.embedding_dir) if f.endswith('.npy')]
    if len(embedding_files) == 0:
        print("🔹 Extracting embeddings...")
        processed_count = 0
        
        for clip_id, label in tqdm(zip(clip_ids, labels), total=len(clip_ids)):
            # Try different possible audio file extensions and paths
            audio_found = False
            for ext in ['wav', 'mp3', 'flac']:
                # Try different possible paths
                possible_paths = [
                    os.path.join(args.audio_dir, f"{clip_id}.{ext}"),
                    os.path.join(args.audio_dir, "**", f"{clip_id}.{ext}")
                ]
                
                for audio_path in possible_paths:
                    if os.path.exists(audio_path):
                        try:
                            emb = extract_wav2vec_embeddings(audio_path, processor, wav2vec_model, device)
                            embedding_path = os.path.join(args.embedding_dir, f"{clip_id}.npy")
                            np.save(embedding_path, emb)
                            processed_count += 1
                            audio_found = True
                            break
                        except Exception as e:
                            print(f"Warning: Error processing {clip_id}: {str(e)}")
                
                if audio_found:
                    break
            
            if not audio_found:
                print(f"Warning: Could not find audio file for {clip_id}")
        
        print(f"🔹 Processed {processed_count} files")
    
    # Create datasets
    train_dataset = BirdCLEFDataset(args.embedding_dir, clip_ids, labels, is_train=True, test_size=args.test_size)
    val_dataset = BirdCLEFDataset(args.embedding_dir, clip_ids, labels, is_train=False, test_size=args.test_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = BirdCLEFClassifier(
        input_dim=train_dataset[0][0].shape[1],
        hidden_dim=256,
        num_layers=2,
        num_classes=train_dataset[0][1].shape[0],
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_fn=args.loss_fn,
        loss_margin=args.loss_margin,
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        wu_weight=args.wu_weight,
        bce_weight=args.bce_weight
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
        verbose=True,
        mode='min'
    )
    
    # Logger
    csv_logger = CSVLogger(save_dir=args.save_dir, name="metrics")
    
    # Trainer
    trainer_config = {
        'accelerator': 'gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu',
        'devices': 1
    }
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=args.save_dir,
        logger=csv_logger,
        check_val_every_n_epoch=args.eval_interval,
        log_every_n_steps=args.log_interval,
        gradient_clip_val=1.0,
        **trainer_config
    )
    
    # Train
    print("🚀 Starting BirdCLEF classification training...")
    trainer.fit(model, train_loader, val_loader)
    print("✅ Training complete!")

if __name__ == '__main__':
    main()

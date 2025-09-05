#!/usr/bin/env python3

# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_gru_001" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.2 --dropout 0.1 --loss_fn "wu_auc" --num_workers 4

# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_001" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_002" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_003" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_004" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_asymmetric" --wu_weight 0.5 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_005" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_asymmetric_bce" --bce_weight 0.1 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_006" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --wu_weight 0.9 --bce_weight 0.1 --num_workers 4

# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_007" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_008" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_009" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_010" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_asymmetric" --wu_weight 0.5 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_011" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_asymmetric_bce" --bce_weight 0.1 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_012" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --wu_weight 0.9 --bce_weight 0.1 --num_workers 4

# Without scheduler and early stopping
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_013" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_014" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_015" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_016" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_asymmetric" --wu_weight 0.5 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_017" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_asymmetric_bce" --bce_weight 0.1 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_018" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --wu_weight 0.9 --bce_weight 0.1 --num_workers 4

"""
BirdCLEF Classification Script
Specialized version for BirdCLEF dataset with multilabel species classification.
Enhanced with improved architecture, better weight initialization, and comprehensive monitoring.
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

# Set multiprocessing start method early
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

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

class TrainEvalMetricsCallback(Callback):
    """Callback to compute comprehensive training and validation metrics."""
    
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute train metrics on training set
        pl_module.eval()
        device = pl_module.device
        loss_fn = pl_module.loss_fn

        # Train set metrics
        train_total_loss = 0.0
        train_total_samples = 0
        train_all_preds = []
        train_all_targets = []

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                preds = pl_module(x)
                loss = loss_fn(preds, y)
                train_total_loss += loss.item() * x.size(0)
                train_total_samples += x.size(0)
                
                # Store predictions and targets
                train_all_preds.append(preds.detach())
                train_all_targets.append(y.detach())

        if train_total_samples == 0:
            print("⚠️ Warning: No training samples found for metrics computation")
            return

        train_avg_loss = train_total_loss / train_total_samples

        # Compute train metrics
        try:
            # Concatenate all predictions and targets
            train_all_preds = torch.cat(train_all_preds, dim=0)
            train_all_targets = torch.cat(train_all_targets, dim=0)
            
            # Apply sigmoid to raw logits for metrics computation
            train_all_preds_probs = torch.sigmoid(train_all_preds)
            train_all_preds_probs = torch.clamp(train_all_preds_probs, min=1e-7, max=1.0-1e-7)
            train_all_targets = train_all_targets.int()
            
            # Create temporary metrics for training data
            f1 = MultilabelF1Score(num_labels=pl_module.f1.num_labels, average="macro").to(device)
            map_metric = MultilabelAveragePrecision(num_labels=pl_module.map.num_labels, average="macro").to(device)
            auc = MultilabelAUROC(num_labels=pl_module.auc.num_labels, average="macro").to(device)
            
            # Compute train metrics using probabilities
            train_f1 = f1(train_all_preds_probs, train_all_targets)
            train_map = map_metric(train_all_preds_probs, train_all_targets)
            train_auc = auc(train_all_preds_probs, train_all_targets)
            
            print(f"✅ Epoch {trainer.current_epoch}: train_f1={train_f1:.4f}, train_map={train_map:.4f}, train_auc={train_auc:.4f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Error computing train metrics: {str(e)}")
            train_f1 = torch.tensor(0.0)
            train_map = torch.tensor(0.0)
            train_auc = torch.tensor(0.0)

        # Compute all validation metrics on validation set
        val_total_loss = 0.0
        val_total_samples = 0
        val_all_preds = []
        val_all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                preds = pl_module(x)
                loss = loss_fn(preds, y)
                val_total_loss += loss.item() * x.size(0)
                val_total_samples += x.size(0)
                
                # Store predictions and targets
                val_all_preds.append(preds.detach())
                val_all_targets.append(y.detach())

        if val_total_samples == 0:
            print("⚠️ Warning: No validation samples found for validation metrics computation")
            return

        val_avg_loss = val_total_loss / val_total_samples

        # Compute all validation metrics
        try:
            # Concatenate all predictions and targets
            val_all_preds = torch.cat(val_all_preds, dim=0)
            val_all_targets = torch.cat(val_all_targets, dim=0)
            
            # Apply sigmoid to raw logits for metrics computation
            val_all_preds_probs = torch.sigmoid(val_all_preds)
            val_all_preds_probs = torch.clamp(val_all_preds_probs, min=1e-7, max=1.0-1e-7)
            val_all_targets = val_all_targets.int()
            
            # Compute all validation metrics using probabilities
            val_f1 = f1(val_all_preds_probs, val_all_targets)
            val_map = map_metric(val_all_preds_probs, val_all_targets)
            val_auc = auc(val_all_preds_probs, val_all_targets)
            
            print(f"✅ Epoch {trainer.current_epoch}: val_f1={val_f1:.4f}, val_map={val_map:.4f}, val_auc={val_auc:.4f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Error computing validation metrics: {str(e)}")
            val_f1 = torch.tensor(0.0)
            val_map = torch.tensor(0.0)
            val_auc = torch.tensor(0.0)

        # Log all metrics using the trainer's logger
        trainer.logger.log_metrics({
            "epoch": trainer.current_epoch,
            "train_loss_eval": train_avg_loss,
            "train_f1_eval": train_f1.item(),
            "train_map_eval": train_map.item(),
            "train_auc_eval": train_auc.item(),
            "val_loss_eval": val_avg_loss,
            "val_f1_eval": val_f1.item(),
            "val_map_eval": val_map.item(),
            "val_auc_eval": val_auc.item()
        }, step=trainer.current_epoch)

        pl_module.train()  # Switch back to training mode

class WeightNormCallback(Callback):
    """Callback to monitor weight and gradient norms for training stability."""
    
    def __init__(self):
        super().__init__()
        self.grad_norms = []
        self.layer_grad_norms = {}
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Capture gradient norms after each training batch (before optimizer step)
        if batch_idx % trainer.log_every_n_steps == 0:  # Only capture periodically to avoid overhead
            total_grad_norm = 0.0
            grad_count = 0
            
            # Track gradients per layer to identify where vanishing occurs
            layer_grads = {}
            
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_count += 1
                    grad_norm = torch.norm(param.grad.data, p=2).item()
                    total_grad_norm += grad_norm ** 2
                    
                    # Track gradients by layer
                    layer_name = name.split('.')[0]  # Get the main layer name
                    if layer_name not in layer_grads:
                        layer_grads[layer_name] = []
                    layer_grads[layer_name].append(grad_norm)
            
            if grad_count > 0:
                total_grad_norm = total_grad_norm ** 0.5
                self.grad_norms.append(total_grad_norm)
                
                # Store layer gradients
                for layer_name, grads in layer_grads.items():
                    if layer_name not in self.layer_grad_norms:
                        self.layer_grad_norms[layer_name] = []
                    self.layer_grad_norms[layer_name].append(np.mean(grads))
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate overall L2 norm for all weights
        total_weight_norm = 0.0
        param_count = 0
        
        for name, param in pl_module.named_parameters():
            if param.requires_grad:  # Only compute for trainable parameters
                param_count += 1
                # Accumulate weight norm
                total_weight_norm += torch.norm(param.data, p=2).item() ** 2
        
        # Take square root to get the overall L2 norm
        total_weight_norm = total_weight_norm ** 0.5
        
        # Use the average gradient norm from this epoch
        total_grad_norm = np.mean(self.grad_norms) if self.grad_norms else 0.0
        
        # Log the overall norms
        trainer.logger.log_metrics({
            "epoch": trainer.current_epoch,
            "total_weight_norm": total_weight_norm,
            "total_grad_norm": total_grad_norm
        }, step=trainer.current_epoch)
        
        # Add monitoring every 10 epochs
        if trainer.current_epoch % 10 == 0:
            print(f"🔍 Epoch {trainer.current_epoch}: Total weight norm = {total_weight_norm:.6f}, Total gradient norm = {total_grad_norm:.6f}")
            
            # Check for gradient explosion/vanishing
            if total_grad_norm > 10.0:
                print(f"⚠️ Warning: High gradient norm detected: {total_grad_norm:.6f}")
            elif total_grad_norm < 1e-6:
                print(f"⚠️ Warning: Very low gradient norm detected: {total_grad_norm:.6f}")
                
                # Print layer-specific gradient norms to identify the problem
                print("🔍 Layer gradient norms:")
                for layer_name, grads in self.layer_grad_norms.items():
                    avg_grad = np.mean(grads) if grads else 0.0
                    print(f"   {layer_name}: {avg_grad:.6f}")
        
        # Clear the gradient norms for the next epoch
        self.grad_norms.clear()
        self.layer_grad_norms.clear()

class BirdCLEFClassifier(pl.LightningModule):
    """Enhanced Lightning module for BirdCLEF classification with improved architecture."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, lr, weight_decay, dropout, 
                 loss_fn="bce", loss_margin=0.1, gamma_pos=0.0, gamma_neg=4.0, wu_weight=0.5, bce_weight=0.5,
                 use_scheduler=False):
        super().__init__()
        self.save_hyperparameters()
        
        # Enhanced GRU layers with better dropout strategy
        gru_dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=gru_dropout)
        
        # Enhanced fully connected layers with LayerNorm and LeakyReLU
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),  # Use LayerNorm instead of BatchNorm for better gradient flow
            nn.LeakyReLU(0.1),  # Use LeakyReLU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
            # Removed Sigmoid - we'll work with raw logits for better gradient flow
        )
        
        # Initialize weights with improved strategies
        self._init_weights()
        
        # Initialize loss function
        if loss_fn == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for raw logits
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
        # Add safety checks for input data
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print("⚠️ Warning: Input contains NaN or Inf values!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Monitor input to GRU (only occasionally)
        if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
            if torch.all(x == 0) or torch.all(torch.abs(x) < 1e-6):
                print(f"⚠️ Warning: GRU input is near zero! Max abs value: {torch.abs(x).max():.6f}")
        
        _, h_n = self.gru(x)
        h_n = h_n[-1]  # Last hidden state
        
        # Monitor GRU output (only occasionally)
        if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
            if torch.all(h_n == 0) or torch.all(torch.abs(h_n) < 1e-6):
                print(f"⚠️ Warning: GRU output is near zero! Max abs value: {torch.abs(h_n).max():.6f}")
        
        output = self.fc(h_n)
        
        # Add safety check for output (only occasionally)
        if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
            if torch.all(output == 0):
                print(f"⚠️ Warning: All predictions are zero! Input shape: {x.shape}, Output shape: {output.shape}")
                print(f"   GRU output range: [{h_n.min():.4f}, {h_n.max():.4f}]")
            print(f"🔍 Model output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        return output

    def _init_weights(self):
        """Initialize weights with improved strategies to prevent vanishing gradients."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for LeakyReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # Small positive bias to avoid dead neurons
            elif isinstance(module, nn.LayerNorm):
                # Initialize layer norm layers properly
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.GRU):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        # Input-to-hidden weights - use Xavier for better gradient flow
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in param_name:
                        # Hidden-to-hidden weights - use orthogonal initialization for RNNs
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in param_name:
                        # Initialize bias with small positive values to avoid dead neurons
                        nn.init.constant_(param, 0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Validate input data
        if batch_idx == 0:  # Only check first batch to avoid spam
            print(f"🔍 Training batch {batch_idx}: x shape={x.shape}, y shape={y.shape}")
            print(f"   x range: [{x.min():.4f}, {x.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
            print(f"   y sum: {y.sum().item()}, y total: {y.numel()}")
            
            # Check for data corruption
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                print("❌ ERROR: Input data contains NaN or Inf values!")
            if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                print("❌ ERROR: Target data contains NaN or Inf values!")
            if y.sum() > y.numel():
                print("❌ ERROR: Target sum exceeds total elements (data corruption)!")
        
        preds = self(x)
        
        # Add safety checks for loss computation
        try:
            loss = self.loss_fn(preds, y)
            # Check if loss is finite
            if not torch.isfinite(loss):
                print(f"⚠️ Warning: Non-finite loss detected: {loss.item()}")
                # Use a fallback loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        except Exception as e:
            print(f"⚠️ Warning: Error computing loss: {str(e)}")
            # Use a fallback loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        
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
        
        # Validate input data
        if batch_idx == 0:  # Only check first batch to avoid spam
            print(f"🔍 Validation batch {batch_idx}: x shape={x.shape}, y shape={y.shape}")
            print(f"   x range: [{x.min():.4f}, {x.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
            print(f"   y sum: {y.sum().item()}, y total: {y.numel()}")
            
            # Check for data corruption
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                print("❌ ERROR: Validation input data contains NaN or Inf values!")
            if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                print("❌ ERROR: Validation target data contains NaN or Inf values!")
            if y.sum() > y.numel():
                print("❌ ERROR: Validation target sum exceeds total elements (data corruption)!")
        
        preds = self(x)
        
        # Add safety checks for loss computation
        try:
            loss = self.loss_fn(preds, y)
            # Check if loss is finite
            if not torch.isfinite(loss):
                print(f"⚠️ Warning: Non-finite loss detected: {loss.item()}")
                # Use a fallback loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        except Exception as e:
            print(f"⚠️ Warning: Error computing loss: {str(e)}")
            # Use a fallback loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        
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
                
                # Validate data before computing metrics
                if torch.any(torch.isnan(all_preds)) or torch.any(torch.isinf(all_preds)):
                    print("❌ ERROR: Validation predictions contain NaN or Inf values!")
                    return
                
                if torch.any(torch.isnan(all_targets)) or torch.any(torch.isinf(all_targets)):
                    print("❌ ERROR: Validation targets contain NaN or Inf values!")
                    return
                
                # Check for data corruption
                if all_targets.sum() > all_targets.numel():
                    print(f"❌ ERROR: Validation target sum ({all_targets.sum()}) exceeds total elements ({all_targets.numel()})!")
                    return
                
                # Apply sigmoid to raw logits
                all_preds_probs = torch.sigmoid(all_preds)
                all_preds_probs = torch.clamp(all_preds_probs, min=1e-7, max=1.0-1e-7)
                all_targets = all_targets.int()
                
                # Check if we have any positive labels
                if all_targets.sum() == 0:
                    print("⚠️ Warning: No positive labels in validation set - skipping metrics computation")
                    return
                
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
                print(f"   Predictions shape: {all_preds.shape}, Targets shape: {all_targets.shape}")
                print(f"   Predictions range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
                print(f"   Targets sum: {all_targets.sum()}, Targets total: {all_targets.numel()}")
        
        # Clear stored predictions and targets
        if hasattr(self, 'val_preds'):
            self.val_preds.clear()
            self.val_targets.clear()

    def configure_optimizers(self):
        # Use AdamW optimizer with better parameters
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Conditionally add cosine annealing scheduler
        if self.hparams.use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,  # Restart every 50 epochs
                T_mult=2,  # Double the restart interval each time
                eta_min=self.hparams.lr * 0.001  # Minimum LR
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer

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
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value to prevent gradient explosion/vanishing")
    parser.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing scheduler")
    parser.add_argument("--use_early_stopping", action="store_true", help="Use early stopping callback")
    parser.add_argument("--early_stopping_patience", type=int, default=50, help="Patience for early stopping")
    
    args = parser.parse_args()
    
    # Device setup - keep original logic without CPU fallback
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"🔹 Using device: {device}")
    
    # Check GPU compatibility if using GPU
    if args.use_gpu and torch.cuda.is_available():
        try:
            # Test GPU compatibility by creating a small tensor
            test_tensor = torch.tensor([1.0], device=device)
            print(f"🔹 GPU: {torch.cuda.get_device_name()}")
            print(f"🔹 CUDA version: {torch.version.cuda}")
            print(f"🔹 GPU capability: {torch.cuda.get_device_capability()}")
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"🔹 GPU memory: {gpu_memory:.1f} GB")
            
            # Enable Tensor Cores for better performance
            torch.set_float32_matmul_precision('high')
            print("🔹 Enabled Tensor Cores for better performance")
            
        except Exception as e:
            print(f"⚠️ Warning: GPU compatibility issue detected: {str(e)}")
            print("🔹 Continuing with GPU as requested...")
    elif not torch.cuda.is_available() and args.use_gpu:
        print("⚠️ Warning: GPU requested but CUDA is not available")
        print("🔹 Continuing with GPU as requested...")
    
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
    
    # Check embedding completion status
    embedding_files = [f for f in os.listdir(args.embedding_dir) if f.endswith('.npy')]
    existing_embeddings = {f.replace('.npy', '') for f in embedding_files}
    
    # Find missing embeddings
    missing_embeddings = []
    for clip_id in clip_ids:
        if clip_id not in existing_embeddings:
            missing_embeddings.append(clip_id)
    
    total_required = len(clip_ids)
    total_existing = len(existing_embeddings)
    total_missing = len(missing_embeddings)
    
    print(f"🔹 Embedding Status:")
    print(f"   - Total required: {total_required}")
    print(f"   - Already computed: {total_existing}")
    print(f"   - Missing: {total_missing}")
    
    if total_missing > 0:
        print(f"🔹 Computing {total_missing} missing embeddings...")
        processed_count = 0
        
        for clip_id, label in tqdm(zip(missing_embeddings, [labels[list(clip_ids).index(clip_id)] for clip_id in missing_embeddings]), total=len(missing_embeddings)):
            # Try different possible audio file extensions and paths
            audio_found = False
            for ext in ['wav', 'mp3', 'flac', 'ogg']:
                # Try different possible paths for BirdCLEF structure
                possible_paths = [
                    os.path.join(args.audio_dir, f"{clip_id}.{ext}"),
                ]
                
                # Add glob patterns for subdirectory search
                glob_patterns = [
                    os.path.join(args.audio_dir, "**", f"{clip_id}.{ext}")
                ]
                
                # Check direct paths first
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
                
                # Check glob patterns for subdirectory search
                for glob_pattern in glob_patterns:
                    matching_files = glob.glob(glob_pattern, recursive=True)
                    if matching_files:
                        audio_path = matching_files[0]  # Take the first match
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
        
        print(f"🔹 Processed {processed_count} missing embeddings")
        print(f"🔹 Total embeddings now available: {total_existing + processed_count}")
    else:
        print(f"✅ All embeddings are already computed!")
    
    # Create datasets
    train_dataset = BirdCLEFDataset(args.embedding_dir, clip_ids, labels, is_train=True, test_size=args.test_size)
    val_dataset = BirdCLEFDataset(args.embedding_dir, clip_ids, labels, is_train=False, test_size=args.test_size)
    
    # Create dataloaders with better error handling
    # Reduce num_workers if it's too high for the system to prevent worker crashes
    adjusted_num_workers = min(args.num_workers, 2)  # Cap at 2 to prevent worker issues
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=adjusted_num_workers,
        pin_memory=True,
        persistent_workers=True if adjusted_num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=adjusted_num_workers,
        pin_memory=True,
        persistent_workers=True if adjusted_num_workers > 0 else False
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
        bce_weight=args.bce_weight,
        use_scheduler=args.use_scheduler
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    
    # Conditionally add early stopping callback
    callbacks = [checkpoint_callback]
    if args.use_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # Logger
    csv_logger = CSVLogger(save_dir=args.save_dir, name="metrics")
    
    # Add comprehensive metrics callbacks
    train_eval_callback = TrainEvalMetricsCallback(train_loader, val_loader)
    weight_norm_callback = WeightNormCallback()
    
    # Add metrics callbacks to the list
    callbacks.extend([train_eval_callback, weight_norm_callback])
    
    # Trainer configuration - keep GPU if requested
    if args.use_gpu and torch.cuda.is_available():
        trainer_config = {
            'accelerator': 'gpu',
            'devices': 1
        }
    else:
        trainer_config = {
            'accelerator': 'cpu',
            'devices': 1
        }
    
    # Disable sanity check if validation dataset is too small
    num_sanity_val_steps = 0 if len(val_dataset) < args.batch_size else 2
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=args.save_dir,
        logger=csv_logger,
        check_val_every_n_epoch=args.eval_interval,
        log_every_n_steps=args.log_interval,
        gradient_clip_val=args.gradient_clip_val,  # Use command line parameter for gradient clipping
        num_sanity_val_steps=num_sanity_val_steps,  # Disable sanity check for small validation sets
        **trainer_config
    )
    
    # Train
    print("🚀 Starting BirdCLEF classification training...")
    trainer.fit(model, train_loader, val_loader)
    print("✅ Training complete!")

if __name__ == '__main__':
    # Add a simple guard to prevent multiple executions
    start_time = time.time()
    print(f"🔹 Starting training script at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    print(f"🔹 Total training time: {time.time() - start_time:.2f} seconds")

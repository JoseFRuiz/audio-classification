# DINO-based Semi-Supervised Audio Classification with MEL Spectrograms and CRNN
# Based on run_experiment_gru_lightning.py but simplified for:
# - MEL feature mode only
# - CRNN model type only  
# - BCE loss only
# - DINO self-supervised pretraining on unlabeled data
# - Supervised fine-tuning on labeled data

# python run_experiment_dino_lightning.py \
#   --save_dir "complete_dino_001" \
#   --dino_pretrain_epochs 100 \
#   --finetune_epochs 200 \
#   --eval_interval 10 \
#   --log_interval 10 \
#   --lr 1e-4 \
#   --weight_decay 1e-5 \
#   --gradient_clip_val 10.0 \
#   --batch_size 100 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1

# python run_experiment_dino_lightning.py \
#   --save_dir "supervised_only_001" \
#   --supervised_only \
#   --freeze_backbone \
#   --finetune_epochs 500 \
#   --eval_interval 10 \
#   --log_interval 10 \
#   --lr 1e-4 \
#   --weight_decay 1e-5 \
#   --gradient_clip_val 10.0 \
#   --batch_size 100 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1 \
#   --init_weights xavier

# python run_experiment_dino_lightning.py \
#   --save_dir "complete_dino_002" \
#   --dino_pretrain_epochs 200 \
#   --finetune_epochs 500 \
#   --eval_interval 10 \
#   --log_interval 10 \
#   --lr 1e-4 \
#   --weight_decay 1e-5 \
#   --gradient_clip_val 10.0 \
#   --batch_size 100 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1

# python run_experiment_dino_lightning.py \
#   --save_dir "supervised_only_002" \
#   --supervised_only \
#   --finetune_epochs 500 \
#   --eval_interval 10 \
#   --log_interval 10 \
#   --lr 1e-4 \
#   --weight_decay 1e-5 \
#   --gradient_clip_val 10.0 \
#   --batch_size 100 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1 \
#   --init_weights xavier

# python run_experiment_dino_lightning.py \
#   --save_dir "complete_dino_003" \
#   --freeze_backbone \
#   --dino_pretrain_epochs 100 \
#   --finetune_epochs 500 \
#   --eval_interval 10 \
#   --log_interval 10 \
#   --lr 1e-4 \
#   --weight_decay 1e-5 \
#   --gradient_clip_val 10.0 \
#   --batch_size 100 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1

# python run_experiment_dino_lightning.py \
#   --save_dir "complete_dinov2_001" \
#   --use_dinov2 \
#   --dino_pretrain_epochs 100 \
#   --finetune_epochs 200 \
#   --ibot_weight 1.0 \
#   --koleo_weight 0.04 \
#   --mask_ratio 0.6 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1

# python run_experiment_dino_lightning.py \
#   --save_dir "complete_dinov2_001" \
#   --use_dinov2 \
#   --skip_dino_pretrain \
#   --resume_from_checkpoint "complete_dinov2_001/best-checkpoint.ckpt" \
#   --finetune_epochs 500 \
#   --ibot_weight 1.0 \
#   --koleo_weight 0.04 \
#   --mask_ratio 0.6 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1

# python run_experiment_dino_lightning.py \
#   --save_dir "complete_dinov2_002" \
#   --use_dinov2 \
#   --dino_pretrain_epochs 200 \
#   --finetune_epochs 500 \
#   --ibot_weight 1.0 \
#   --koleo_weight 0.04 \
#   --mask_ratio 0.6 \
#   --use_gpu \
#   --test_size 0.1 \
#   --dropout 0.1 \
#   --num_workers 4 \
#   --conv_channels 64 128 256 \
#   --conv_kernel_size 3 \
#   --conv_stride 1 \
#   --dataset_type "complete" \
#   --bptt_length 60 \
#   --labeled_ratio 0.1

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC
import librosa
from tqdm import tqdm
from pytorch_lightning.loggers import CSVLogger
import json
import multiprocessing
import time
import warnings
from transformers import logging as transformers_logging

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

TARGET_LENGTH = 10 * 16000
SAMPLE_RATE = 16000


# ========================
# 1. DINO Augmentation Module
# ========================
class DINOAugmentation:
    """Augmentation module for DINO self-supervised learning on MEL spectrograms."""
    
    def __init__(self, time_mask_param=27, freq_mask_param=12, use_mixup=False, 
                 use_cutmix=False, mixup_alpha=0.4, time_shift_max=0.1):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.time_shift_max = time_shift_max
    
    def time_mask(self, spec, param=None):
        """Apply time masking (SpecAugment-style)."""
        if param is None:
            param = self.time_mask_param
        
        if param == 0:
            return spec
        
        n_mels, time_frames = spec.shape
        t0 = np.random.randint(0, time_frames - param + 1)
        spec[:, t0:t0+param] = 0
        return spec
    
    def freq_mask(self, spec, param=None):
        """Apply frequency masking (SpecAugment-style)."""
        if param is None:
            param = self.freq_mask_param
        
        if param == 0:
            return spec
        
        n_mels, time_frames = spec.shape
        f0 = np.random.randint(0, n_mels - param + 1)
        spec[f0:f0+param, :] = 0
        return spec
    
    def time_shift(self, spec):
        """Apply time shifting."""
        if self.time_shift_max == 0:
            return spec
        
        n_mels, time_frames = spec.shape
        shift = int(time_frames * self.time_shift_max * (2 * np.random.rand() - 1))
        
        if shift > 0:
            spec = np.concatenate([np.zeros((n_mels, shift)), spec[:, :-shift]], axis=1)
        elif shift < 0:
            spec = np.concatenate([spec[:, -shift:], np.zeros((n_mels, -shift))], axis=1)
        
        return spec
    
    def mixup(self, spec1, spec2, alpha=0.4):
        """Apply Mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        return mixed_spec
    
    def cutmix(self, spec1, spec2, alpha=0.4):
        """Apply CutMix augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        n_mels, time_frames = spec1.shape
        
        # Generate random bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(n_mels * cut_ratio)
        cut_w = int(time_frames * cut_ratio)
        
        cy = np.random.randint(n_mels)
        cx = np.random.randint(time_frames)
        
        bby1 = np.clip(cy - cut_h // 2, 0, n_mels)
        bbx1 = np.clip(cx - cut_w // 2, 0, time_frames)
        bby2 = np.clip(cy + cut_h // 2, 0, n_mels)
        bbx2 = np.clip(cx + cut_w // 2, 0, time_frames)
        
        mixed_spec = spec1.copy()
        mixed_spec[bby1:bby2, bbx1:bbx2] = spec2[bby1:bby2, bbx1:bbx2]
        
        return mixed_spec
    
    def apply_global_view(self, spec):
        """Apply lighter augmentations for global view (teacher)."""
        spec = spec.copy()
        
        # Light augmentations
        if np.random.rand() < 0.5:
            spec = self.time_mask(spec, param=self.time_mask_param // 2)
        if np.random.rand() < 0.5:
            spec = self.freq_mask(spec, param=self.freq_mask_param // 2)
        if np.random.rand() < 0.3:
            spec = self.time_shift(spec)
        
        return spec
    
    def apply_local_view(self, spec):
        """Apply stronger augmentations for local view (student)."""
        spec = spec.copy()
        
        # Strong augmentations
        if np.random.rand() < 0.8:
            spec = self.time_mask(spec, param=self.time_mask_param)
        if np.random.rand() < 0.8:
            spec = self.freq_mask(spec, param=self.freq_mask_param)
        if np.random.rand() < 0.5:
            spec = self.time_shift(spec)
        
        return spec
    
    def apply_two_views(self, spec, spec2=None):
        """Apply augmentations to create two views for DINO."""
        global_view = self.apply_global_view(spec)
        local_view = self.apply_local_view(spec)
        
        # Optionally apply mixup/cutmix to local view
        if spec2 is not None:
            if self.use_mixup and np.random.rand() < 0.5:
                local_view = self.mixup(local_view, spec2, self.mixup_alpha)
            elif self.use_cutmix and np.random.rand() < 0.5:
                local_view = self.cutmix(local_view, spec2, self.mixup_alpha)
        
        return global_view, local_view
    
    def create_masked_view(self, spec, mask_ratio=0.6, patch_size_time=8, patch_size_freq=4):
        """
        Create a masked view for iBOT (DINOv2).
        
        Args:
            spec: (n_mels, time_frames) spectrogram
            mask_ratio: Ratio of patches to mask
            patch_size_time: Time dimension of each patch
            patch_size_freq: Frequency dimension of each patch
        
        Returns:
            masked_spec: (n_mels, time_frames) masked spectrogram
            mask: (num_patches_time, num_patches_freq) boolean mask (True = masked)
        """
        n_mels, time_frames = spec.shape
        
        # Calculate number of patches
        num_patches_time = (time_frames + patch_size_time - 1) // patch_size_time
        num_patches_freq = (n_mels + patch_size_freq - 1) // patch_size_freq
        
        # Create mask for patches
        num_patches = num_patches_time * num_patches_freq
        num_masked = int(num_patches * mask_ratio)
        
        # Randomly select patches to mask
        patch_indices = np.random.permutation(num_patches)[:num_masked]
        mask = np.zeros(num_patches, dtype=bool)
        mask[patch_indices] = True
        mask = mask.reshape(num_patches_freq, num_patches_time)
        
        # Apply mask to spectrogram
        masked_spec = spec.copy()
        for f_idx in range(num_patches_freq):
            for t_idx in range(num_patches_time):
                if mask[f_idx, t_idx]:
                    f_start = f_idx * patch_size_freq
                    f_end = min(f_start + patch_size_freq, n_mels)
                    t_start = t_idx * patch_size_time
                    t_end = min(t_start + patch_size_time, time_frames)
                    masked_spec[f_start:f_end, t_start:t_end] = 0
        
        return masked_spec, mask


# ========================
# 2. Modified MELSpectrogramDataset with DINO Support
# ========================
class MELSpectrogramDataset(Dataset):
    """Dataset class for MEL spectrogram extraction with DINO augmentation support."""
    
    def __init__(self, audio_dir, clip_ids, labels, indices=None, is_train=True, test_size=0.1, random_state=42,
                 target_length=160000, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512,
                 use_dino_aug=False, augmentation=None, return_label=True, use_dinov2=False, mask_ratio=0.6):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_train = is_train
        self.use_dino_aug = use_dino_aug
        self.augmentation = augmentation
        self.return_label = return_label
        self.use_dinov2 = use_dinov2
        self.mask_ratio = mask_ratio
        
        print(f"🔹 Looking for raw audio files in: {os.path.abspath(audio_dir)}")
        print(f"🔹 MEL parameters: n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}")
        if use_dino_aug:
            print(f"🔹 DINO augmentation enabled")
        if use_dinov2:
            print(f"🔹 DINOv2 mode enabled (with iBOT loss)")
        
        # Store the pre-filtered data
        self.clip_ids = np.array(clip_ids)
        self.labels = np.array(labels) if labels is not None else None
        
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
        
        # Additional safety checks
        if len(self.indices) == 0:
            raise ValueError(f"Empty {'training' if is_train else 'validation'} dataset. "
                           f"This might be due to an incorrect test_size parameter or insufficient data.")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        clip_idx = self.indices[idx]
        clip_id = self.clip_ids[clip_idx]
        label = self.labels[clip_idx] if self.labels is not None else None
        
        # Find audio file
        audio_path = None
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            temp_path = os.path.join(self.audio_dir, f"{clip_id}.{ext}")
            if os.path.exists(temp_path):
                audio_path = temp_path
                break
        
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for {clip_id}")
        
        # Load and preprocess audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or truncate to target length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.target_length]
            
            # Compute MEL spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=0,
                fmax=self.sample_rate // 2
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Apply DINO augmentations if enabled
            if self.use_dino_aug and self.augmentation is not None and self.is_train:
                # Get another random sample for mixup/cutmix if needed
                spec2 = None
                if (self.augmentation.use_mixup or self.augmentation.use_cutmix) and np.random.rand() < 0.5:
                    # Get another random sample
                    other_idx = np.random.randint(0, len(self.indices))
                    other_clip_idx = self.indices[other_idx]
                    other_clip_id = self.clip_ids[other_clip_idx]
                    
                    # Load other audio
                    other_audio_path = None
                    for ext in ['wav', 'mp3', 'flac', 'ogg']:
                        temp_path = os.path.join(self.audio_dir, f"{other_clip_id}.{ext}")
                        if os.path.exists(temp_path):
                            other_audio_path = temp_path
                            break
                    
                    if other_audio_path is not None:
                        try:
                            other_audio, _ = librosa.load(other_audio_path, sr=self.sample_rate)
                            if len(other_audio) < self.target_length:
                                other_audio = np.pad(other_audio, (0, self.target_length - len(other_audio)), mode='constant')
                            else:
                                other_audio = other_audio[:self.target_length]
                            
                            other_mel_spec = librosa.feature.melspectrogram(
                                y=other_audio,
                                sr=self.sample_rate,
                                n_mels=self.n_mels,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                fmin=0,
                                fmax=self.sample_rate // 2
                            )
                            spec2 = librosa.power_to_db(other_mel_spec, ref=np.max)
                        except:
                            pass
                
                # Apply two-view augmentation
                global_view, local_view = self.augmentation.apply_two_views(mel_spec_db, spec2)
                
                # Convert to tensors
                global_tensor = torch.tensor(global_view, dtype=torch.float32)
                local_tensor = torch.tensor(local_view, dtype=torch.float32)
                
                # For DINOv2: create masked view and mask
                if self.use_dinov2 and self.augmentation is not None:
                    masked_view, mask = self.augmentation.create_masked_view(
                        local_view, mask_ratio=self.mask_ratio
                    )
                    masked_tensor = torch.tensor(masked_view, dtype=torch.float32)
                    mask_tensor = torch.tensor(mask, dtype=torch.bool)
                    
                    if self.return_label and label is not None:
                        return global_tensor, local_tensor, masked_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)
                    else:
                        return global_tensor, local_tensor, masked_tensor, mask_tensor
                else:
                    if self.return_label and label is not None:
                        return global_tensor, local_tensor, torch.tensor(label, dtype=torch.float32)
                    else:
                        return global_tensor, local_tensor
            else:
                # No augmentation: return single view
                mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32)
                
                if self.return_label and label is not None:
                    return mel_tensor, torch.tensor(label, dtype=torch.float32)
                else:
                    return mel_tensor
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            # Return zero tensor as fallback
            time_frames = (self.target_length // self.hop_length) + 1
            fallback_tensor = torch.zeros(self.n_mels, time_frames, dtype=torch.float32)
            
            if self.use_dino_aug and self.is_train:
                if self.return_label and label is not None:
                    return fallback_tensor, fallback_tensor, torch.tensor(label, dtype=torch.float32)
                else:
                    return fallback_tensor, fallback_tensor
            else:
                if self.return_label and label is not None:
                    return fallback_tensor, torch.tensor(label, dtype=torch.float32)
                else:
                    return fallback_tensor


# ========================
# 3. CRNN Backbone (without classifier)
# ========================
class CRNNBackbone(nn.Module):
    """CRNN backbone that extracts features before classification."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, 
                 conv_channels=[64, 128, 256], conv_kernel_size=3, conv_stride=1,
                 bptt_length=60, use_bidirectional=False, init_weights="xavier"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bptt_length = bptt_length
        self.use_bidirectional = use_bidirectional
        
        # Build 1D convolutional layers for temporal feature extraction
        # For MEL: Input is (batch_size, n_mels, time_frames) -> Conv1d along temporal dimension
        conv_layers = []
        in_channels = input_dim  # n_mels for MEL spectrograms
        
        for i, out_channels in enumerate(conv_channels):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=conv_kernel_size // 2  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)  # Reduced dropout for conv layers
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        conv_output_dim = conv_channels[-1]  # Output dimension from conv layers
        
        # GRU after convolutions
        gru_dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(
            conv_output_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=gru_dropout,
            bidirectional=use_bidirectional
        )
        # Adjust hidden dimension for bidirectional
        self.output_dim = hidden_dim * 2 if use_bidirectional else hidden_dim
        
        # Initialize weights
        self._init_weights(init_weights)
    
    def _init_weights(self, init_method="xavier"):
        """Initialize weights for the backbone."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                if init_method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif init_method == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif init_method == "random":
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        if init_method == "xavier":
                            nn.init.xavier_uniform_(param)
                        elif init_method == "normal":
                            nn.init.normal_(param, mean=0.0, std=0.02)
                        elif init_method == "random":
                            nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
    
    def forward(self, x, return_patch_features=False):
        """
        Forward pass through CRNN backbone.
        
        Args:
            x: Input tensor of shape (batch_size, n_mels, time_frames)
            return_patch_features: If True, return patch-level features for iBOT
        
        Returns:
            features: (batch_size, hidden_dim) tensor if return_patch_features=False
            (pooled_features, patch_features): tuple if return_patch_features=True
                - pooled_features: (batch_size, hidden_dim) global features
                - patch_features: (batch_size, seq_len, hidden_dim) patch-level features
        """
        # MEL mode: x is already (batch_size, n_mels, time_frames)
        # This is the correct shape for Conv1d (batch, channels, time)
        # Apply convolutional layers
        x = self.conv_layers(x)  # (batch_size, conv_channels[-1], time_frames)
        
        # Transpose for GRU: (batch_size, time_frames, conv_channels[-1])
        x = x.transpose(1, 2)
        
        # Process with GRU (with BPTT if needed)
        if self.bptt_length > 0 and x.size(1) > self.bptt_length and not self.use_bidirectional:
            # Truncated BPTT: process in chunks
            batch_size, seq_len, features = x.shape
            outputs = []
            hidden = None
            
            for i in range(0, seq_len, self.bptt_length):
                chunk = x[:, i:i+self.bptt_length, :]
                chunk_output, hidden = self.gru(chunk, hidden)
                
                if i + self.bptt_length < seq_len:
                    hidden = hidden.detach()
                
                outputs.append(chunk_output)
            
            gru_output = torch.cat(outputs, dim=1)
        else:
            gru_output, _ = self.gru(x)
        
        # Mean pooling over time dimension
        pooled_output = torch.mean(gru_output, dim=1)  # (batch_size, hidden_dim)
        
        if return_patch_features:
            return pooled_output, gru_output  # (batch_size, hidden_dim), (batch_size, seq_len, hidden_dim)
        else:
            return pooled_output


# ========================
# 4. Projection Head for DINO
# ========================
class ProjectionHead(nn.Module):
    """Projection head for DINO self-supervised learning."""
    
    def __init__(self, in_dim, out_dim=256, num_layers=3, hidden_dim=2048, use_bn=True):
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
        
        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
    
    def forward(self, x):
        x = self.layers(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ========================
# 5. DINO Loss Function
# ========================
def dino_loss(student_output, teacher_output, center, temperature=0.07):
    """
    Compute DINO loss between student and teacher outputs.
    
    Args:
        student_output: (batch_size, proj_dim) student network output
        teacher_output: (batch_size, proj_dim) teacher network output
        center: (proj_dim,) running center for teacher outputs
        temperature: Temperature for sharpening teacher softmax
    
    Returns:
        loss: Scalar loss value
    """
    # Apply centering to teacher output
    teacher_output = teacher_output - center
    
    # Sharpening: apply temperature to teacher
    teacher_out = teacher_output / temperature
    teacher_probs = F.softmax(teacher_out, dim=-1)
    
    # Student output (no temperature for student)
    student_log_probs = F.log_softmax(student_output, dim=-1)
    
    # Cross-entropy loss
    loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
    
    return loss


# ========================
# 5b. iBOT Loss Function (for DINOv2)
# ========================
def ibot_loss(student_patch_output, teacher_patch_output, mask, center, temperature=0.07):
    """
    Compute iBOT loss for masked spectrogram modeling (patch-level).
    
    Args:
        student_patch_output: (batch_size, num_patches, proj_dim) student patch outputs
        teacher_patch_output: (batch_size, num_patches, proj_dim) teacher patch outputs
        mask: (batch_size, num_patches) boolean mask (True = masked, False = visible)
        center: (proj_dim,) running center for teacher outputs
        temperature: Temperature for sharpening teacher softmax
    
    Returns:
        loss: Scalar loss value
    """
    # Only compute loss on masked patches
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_patch_output.device, requires_grad=True)
    
    # Flatten batch and sequence dimensions: (batch_size, num_patches, proj_dim) -> (batch_size * num_patches, proj_dim)
    batch_size, num_patches, proj_dim = student_patch_output.shape
    student_flat = student_patch_output.reshape(batch_size * num_patches, proj_dim)
    teacher_flat = teacher_patch_output.reshape(batch_size * num_patches, proj_dim)
    mask_flat = mask.reshape(batch_size * num_patches)
    
    # Select only masked patches
    student_masked = student_flat[mask_flat]  # (num_masked, proj_dim)
    teacher_masked = teacher_flat[mask_flat]  # (num_masked, proj_dim)
    
    if student_masked.size(0) == 0:
        return torch.tensor(0.0, device=student_patch_output.device, requires_grad=True)
    
    # Apply centering to teacher output
    teacher_masked = teacher_masked - center
    
    # Sharpening: apply temperature to teacher
    teacher_out = teacher_masked / temperature
    teacher_probs = F.softmax(teacher_out, dim=-1)
    
    # Student output (no temperature for student)
    student_log_probs = F.log_softmax(student_masked, dim=-1)
    
    # Cross-entropy loss
    loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
    
    return loss


# ========================
# 5c. KoLeo Regularizer (for DINOv2)
# ========================
def koleo_regularizer(features, eps=1e-8):
    """
    KoLeo regularizer: promotes uniform spreading of features.
    
    Args:
        features: (batch_size, feature_dim) feature tensor
        eps: Small epsilon for numerical stability
    
    Returns:
        loss: Scalar regularization value
    """
    # Normalize features
    features_norm = F.normalize(features, p=2, dim=-1)
    
    # Compute pairwise distances
    # features_norm: (batch_size, feature_dim)
    # Compute (batch_size, batch_size) distance matrix
    pairwise_dist = torch.cdist(features_norm, features_norm, p=2)  # (batch_size, batch_size)
    
    # Add small epsilon to avoid log(0)
    pairwise_dist = pairwise_dist + eps
    
    # Compute log of distances (excluding diagonal)
    mask = ~torch.eye(pairwise_dist.size(0), dtype=torch.bool, device=pairwise_dist.device)
    log_distances = torch.log(pairwise_dist[mask])
    
    # KoLeo loss: negative mean of log distances (encourages larger distances)
    # We want to maximize distances, so we minimize negative mean
    loss = -log_distances.mean()
    
    return loss


# ========================
# 6. DINO Model (Lightning Module)
# ========================
class DINOModel(pl.LightningModule):
    """DINO model for self-supervised pretraining."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, lr, weight_decay,
                 conv_channels=[64, 128, 256], conv_kernel_size=3, conv_stride=1,
                 bptt_length=60, use_bidirectional=False,
                 teacher_hidden_dim=None, teacher_num_layers=None,
                 dino_momentum=0.996, dino_temperature=0.07, dino_center_momentum=0.9,
                 dino_proj_dim=256, dino_proj_layers=3):
        super().__init__()
        self.save_hyperparameters()
        
        # Student network
        self.student_backbone = CRNNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            bptt_length=bptt_length,
            use_bidirectional=use_bidirectional
        )
        
        self.student_proj = ProjectionHead(
            in_dim=self.student_backbone.output_dim,
            out_dim=dino_proj_dim,
            num_layers=dino_proj_layers
        )
        
        # Teacher network (same architecture, will be updated via EMA)
        teacher_hidden = teacher_hidden_dim if teacher_hidden_dim is not None else hidden_dim
        teacher_layers = teacher_num_layers if teacher_num_layers is not None else num_layers
        
        self.teacher_backbone = CRNNBackbone(
            input_dim=input_dim,
            hidden_dim=teacher_hidden,
            num_layers=teacher_layers,
            dropout=dropout,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            bptt_length=bptt_length,
            use_bidirectional=use_bidirectional
        )
        
        self.teacher_proj = ProjectionHead(
            in_dim=self.teacher_backbone.output_dim,
            out_dim=dino_proj_dim,
            num_layers=dino_proj_layers
        )
        
        # Initialize teacher with student weights
        self._init_teacher()
        
        # Freeze teacher (only update via EMA)
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_proj.parameters():
            param.requires_grad = False
        
        # Centering buffer
        self.register_buffer('center', torch.zeros(dino_proj_dim))
        
        # DINO hyperparameters
        self.dino_momentum = dino_momentum
        self.dino_temperature = dino_temperature
        self.dino_center_momentum = dino_center_momentum
        
        self.training_step_outputs = []
    
    def _init_teacher(self):
        """Initialize teacher network with student weights."""
        with torch.no_grad():
            # Copy student backbone to teacher backbone
            for student_param, teacher_param in zip(
                self.student_backbone.parameters(), 
                self.teacher_backbone.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
            
            # Copy student projection to teacher projection
            for student_param, teacher_param in zip(
                self.student_proj.parameters(), 
                self.teacher_proj.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
    
    def forward(self, x, is_teacher=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, n_mels, time_frames)
            is_teacher: If True, use teacher network, else use student
        
        Returns:
            output: (batch_size, proj_dim) projection output
        """
        if is_teacher:
            with torch.no_grad():
                features = self.teacher_backbone(x)
                output = self.teacher_proj(features)
        else:
            features = self.student_backbone(x)
            output = self.student_proj(features)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step for DINO pretraining."""
        # Batch contains (global_view, local_view) for unlabeled data
        # DataLoader automatically batches, so each view is (batch_size, n_mels, time_frames)
        global_view, local_view = batch
        
        # Forward pass
        student_local = self.forward(local_view, is_teacher=False)
        teacher_global = self.forward(global_view, is_teacher=True)
        
        # Compute DINO loss
        loss = dino_loss(
            student_local, 
            teacher_global, 
            self.center, 
            temperature=self.dino_temperature
        )
        
        # Update center
        with torch.no_grad():
            self.center = self.dino_center_momentum * self.center + (1 - self.dino_center_momentum) * teacher_global.mean(dim=0)
        
        self.training_step_outputs.append(loss.item())
        
        # Log loss
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('dino_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher network via EMA after each batch."""
        with torch.no_grad():
            # Update teacher backbone
            for student_param, teacher_param in zip(
                self.student_backbone.parameters(), 
                self.teacher_backbone.parameters()
            ):
                teacher_param.data = self.dino_momentum * teacher_param.data + (1 - self.dino_momentum) * student_param.data
            
            # Update teacher projection
            for student_param, teacher_param in zip(
                self.student_proj.parameters(), 
                self.teacher_proj.parameters()
            ):
                teacher_param.data = self.dino_momentum * teacher_param.data + (1 - self.dino_momentum) * student_param.data
    
    def on_train_epoch_end(self):
        """Log average training loss for the epoch."""
        if len(self.training_step_outputs) > 0:
            avg_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
            self.log('dino_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.training_step_outputs.clear()
        else:
            # No training steps executed (e.g., when resuming from checkpoint)
            self.training_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer for DINO training."""
        optimizer = optim.AdamW(
            self.student_backbone.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Also optimize projection head
        optimizer.add_param_group({
            'params': self.student_proj.parameters(),
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay
        })
        
        return optimizer


# ========================
# 6b. DINOv2 Model (Lightning Module) - Combines DINO and iBOT
# ========================
class DINOv2Model(pl.LightningModule):
    """DINOv2 model combining DINO (global) and iBOT (patch-level) self-supervised learning."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, lr, weight_decay,
                 conv_channels=[64, 128, 256], conv_kernel_size=3, conv_stride=1,
                 bptt_length=60, use_bidirectional=False,
                 teacher_hidden_dim=None, teacher_num_layers=None,
                 dino_momentum=0.996, dino_temperature=0.07, dino_center_momentum=0.9,
                 dino_proj_dim=256, dino_proj_layers=3,
                 ibot_weight=1.0, koleo_weight=0.04, mask_ratio=0.6):
        super().__init__()
        self.save_hyperparameters()
        
        # Student network
        self.student_backbone = CRNNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            bptt_length=bptt_length,
            use_bidirectional=use_bidirectional
        )
        
        # Global projection head (for DINO loss)
        self.student_proj = ProjectionHead(
            in_dim=self.student_backbone.output_dim,
            out_dim=dino_proj_dim,
            num_layers=dino_proj_layers
        )
        
        # Patch-level projection head (for iBOT loss)
        self.student_patch_proj = ProjectionHead(
            in_dim=self.student_backbone.output_dim,
            out_dim=dino_proj_dim,
            num_layers=dino_proj_layers
        )
        
        # Teacher network (same architecture, will be updated via EMA)
        teacher_hidden = teacher_hidden_dim if teacher_hidden_dim is not None else hidden_dim
        teacher_layers = teacher_num_layers if teacher_num_layers is not None else num_layers
        
        self.teacher_backbone = CRNNBackbone(
            input_dim=input_dim,
            hidden_dim=teacher_hidden,
            num_layers=teacher_layers,
            dropout=dropout,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            bptt_length=bptt_length,
            use_bidirectional=use_bidirectional
        )
        
        # Global projection head for teacher
        self.teacher_proj = ProjectionHead(
            in_dim=self.teacher_backbone.output_dim,
            out_dim=dino_proj_dim,
            num_layers=dino_proj_layers
        )
        
        # Patch-level projection head for teacher
        self.teacher_patch_proj = ProjectionHead(
            in_dim=self.teacher_backbone.output_dim,
            out_dim=dino_proj_dim,
            num_layers=dino_proj_layers
        )
        
        # Initialize teacher with student weights
        self._init_teacher()
        
        # Freeze teacher (only update via EMA)
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_proj.parameters():
            param.requires_grad = False
        for param in self.teacher_patch_proj.parameters():
            param.requires_grad = False
        
        # Centering buffer
        self.register_buffer('center', torch.zeros(dino_proj_dim))
        self.register_buffer('patch_center', torch.zeros(dino_proj_dim))
        
        # DINOv2 hyperparameters
        self.dino_momentum = dino_momentum
        self.dino_temperature = dino_temperature
        self.dino_center_momentum = dino_center_momentum
        self.ibot_weight = ibot_weight
        self.koleo_weight = koleo_weight
        self.mask_ratio = mask_ratio
        
        self.training_step_outputs = []
    
    def _init_teacher(self):
        """Initialize teacher network with student weights."""
        with torch.no_grad():
            # Copy student backbone to teacher backbone
            for student_param, teacher_param in zip(
                self.student_backbone.parameters(), 
                self.teacher_backbone.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
            
            # Copy student projections to teacher projections
            for student_param, teacher_param in zip(
                self.student_proj.parameters(), 
                self.teacher_proj.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
            
            for student_param, teacher_param in zip(
                self.student_patch_proj.parameters(), 
                self.teacher_patch_proj.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
    
    def forward(self, x, is_teacher=False, return_patch_features=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, n_mels, time_frames)
            is_teacher: If True, use teacher network, else use student
            return_patch_features: If True, return patch-level features for iBOT
        
        Returns:
            output: (batch_size, proj_dim) projection output if return_patch_features=False
            (global_output, patch_output): tuple if return_patch_features=True
        """
        if is_teacher:
            with torch.no_grad():
                if return_patch_features:
                    pooled_features, patch_features = self.teacher_backbone(x, return_patch_features=True)
                    global_output = self.teacher_proj(pooled_features)
                    # Reshape patch features for projection head: (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
                    batch_size, seq_len, hidden_dim = patch_features.shape
                    patch_features_flat = patch_features.view(batch_size * seq_len, hidden_dim)
                    patch_output_flat = self.teacher_patch_proj(patch_features_flat)
                    # Reshape back: (batch_size * seq_len, proj_dim) -> (batch_size, seq_len, proj_dim)
                    patch_output = patch_output_flat.view(batch_size, seq_len, -1)
                    return global_output, patch_output
                else:
                    features = self.teacher_backbone(x)
                    output = self.teacher_proj(features)
                    return output
        else:
            if return_patch_features:
                pooled_features, patch_features = self.student_backbone(x, return_patch_features=True)
                global_output = self.student_proj(pooled_features)
                # Reshape patch features for projection head: (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
                batch_size, seq_len, hidden_dim = patch_features.shape
                patch_features_flat = patch_features.view(batch_size * seq_len, hidden_dim)
                patch_output_flat = self.student_patch_proj(patch_features_flat)
                # Reshape back: (batch_size * seq_len, proj_dim) -> (batch_size, seq_len, proj_dim)
                patch_output = patch_output_flat.view(batch_size, seq_len, -1)
                return global_output, patch_output
            else:
                features = self.student_backbone(x)
                output = self.student_proj(features)
                return output
    
    def training_step(self, batch, batch_idx):
        """Training step for DINOv2 pretraining."""
        # Batch contains (global_view, local_view, masked_view, mask) for unlabeled data
        if len(batch) == 4:
            global_view, local_view, masked_view, mask = batch
        else:
            # Fallback to DINO-only mode if mask not provided
            global_view, local_view = batch
            masked_view = None
            mask = None
        
        # Forward pass for global features (DINO loss)
        student_local_global = self.forward(local_view, is_teacher=False, return_patch_features=False)
        teacher_global_global = self.forward(global_view, is_teacher=True, return_patch_features=False)
        
        # Compute DINO loss (global)
        dino_loss_val = dino_loss(
            student_local_global, 
            teacher_global_global, 
            self.center, 
            temperature=self.dino_temperature
        )
        
        # Forward pass for patch features (iBOT loss)
        ibot_loss_val = torch.tensor(0.0, device=dino_loss_val.device)
        if masked_view is not None and mask is not None:
            student_local_global_patch, student_local_patch = self.forward(
                local_view, is_teacher=False, return_patch_features=True
            )
            teacher_masked_global_patch, teacher_masked_patch = self.forward(
                masked_view, is_teacher=True, return_patch_features=True
            )
            
            # Convert mask to tensor if needed and handle shape
            batch_size, seq_len, _ = student_local_patch.shape
            
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).bool()
            
            # Ensure mask is on correct device
            mask = mask.to(student_local_patch.device)
            
            # Handle mask shape: mask from dataset can be 2D (num_patches_freq, num_patches_time)
            # or 3D (batch_size, num_patches_freq, num_patches_time) after batching
            if mask.dim() == 3:
                # 3D mask: (batch_size, num_patches_freq, num_patches_time)
                # Flatten each sample's mask to 1D
                mask_flat = mask.view(batch_size, -1)  # (batch_size, num_patches_freq * num_patches_time)
            elif mask.dim() == 2:
                # 2D mask: (num_patches_freq, num_patches_time) - single sample, repeat for batch
                mask_flat = mask.flatten().unsqueeze(0).repeat(batch_size, 1)  # (batch_size, num_patches)
            elif mask.dim() == 1:
                # 1D mask: (num_patches,)
                mask_flat = mask.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, num_patches)
            else:
                # Already in correct shape (batch_size, seq_len)
                mask_flat = mask
            
            # Truncate or pad to match sequence length
            num_patches = mask_flat.size(1)
            if num_patches > seq_len:
                # Truncate to sequence length
                mask_flat = mask_flat[:, :seq_len]
            elif num_patches < seq_len:
                # Pad with False (unmasked) to match sequence length
                padding = torch.zeros(batch_size, seq_len - num_patches, 
                                     dtype=torch.bool, device=mask_flat.device)
                mask_flat = torch.cat([mask_flat, padding], dim=1)
            
            # Compute iBOT loss (patch-level)
            ibot_loss_val = ibot_loss(
                student_local_patch,
                teacher_masked_patch,
                mask_flat,
                self.patch_center,
                temperature=self.dino_temperature
            )
        
        # KoLeo regularizer
        koleo_loss_val = koleo_regularizer(student_local_global)
        
        # Combined loss
        total_loss = dino_loss_val + self.ibot_weight * ibot_loss_val + self.koleo_weight * koleo_loss_val
        
        # Update centers
        with torch.no_grad():
            self.center = self.dino_center_momentum * self.center + (1 - self.dino_center_momentum) * teacher_global_global.mean(dim=0)
            if masked_view is not None:
                self.patch_center = self.dino_center_momentum * self.patch_center + (1 - self.dino_center_momentum) * teacher_masked_patch.mean(dim=[0, 1])
        
        self.training_step_outputs.append(total_loss.item())
        
        # Log losses
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('dino_loss', dino_loss_val, on_step=True, on_epoch=False, prog_bar=True)
            self.log('ibot_loss', ibot_loss_val, on_step=True, on_epoch=False, prog_bar=True)
            self.log('koleo_loss', koleo_loss_val, on_step=True, on_epoch=False, prog_bar=True)
            self.log('total_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        
        return total_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher network via EMA after each batch."""
        with torch.no_grad():
            # Update teacher backbone
            for student_param, teacher_param in zip(
                self.student_backbone.parameters(), 
                self.teacher_backbone.parameters()
            ):
                teacher_param.data = self.dino_momentum * teacher_param.data + (1 - self.dino_momentum) * student_param.data
            
            # Update teacher projections
            for student_param, teacher_param in zip(
                self.student_proj.parameters(), 
                self.teacher_proj.parameters()
            ):
                teacher_param.data = self.dino_momentum * teacher_param.data + (1 - self.dino_momentum) * student_param.data
            
            for student_param, teacher_param in zip(
                self.student_patch_proj.parameters(), 
                self.teacher_patch_proj.parameters()
            ):
                teacher_param.data = self.dino_momentum * teacher_param.data + (1 - self.dino_momentum) * student_param.data
    
    def on_train_epoch_end(self):
        """Log average training loss for the epoch."""
        if len(self.training_step_outputs) > 0:
            avg_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
            self.log('total_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.training_step_outputs.clear()
        else:
            # No training steps executed (e.g., when resuming from checkpoint)
            self.training_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer for DINOv2 training."""
        optimizer = optim.AdamW(
            self.student_backbone.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Also optimize projection heads
        optimizer.add_param_group({
            'params': self.student_proj.parameters(),
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay
        })
        
        optimizer.add_param_group({
            'params': self.student_patch_proj.parameters(),
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay
        })
        
        return optimizer


# ========================
# 7. Supervised Model (Lightning Module) for Fine-tuning
# ========================
class SupervisedModel(pl.LightningModule):
    """Supervised model for fine-tuning with pretrained backbone."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout, lr, weight_decay,
                 conv_channels=[64, 128, 256], conv_kernel_size=3, conv_stride=1,
                 bptt_length=60, use_bidirectional=False, freeze_backbone=False,
                 pretrained_backbone=None, init_weights="xavier"):
        super().__init__()
        self.save_hyperparameters()
        self.freeze_backbone = freeze_backbone
        self.init_weights = init_weights
        
        # Load pretrained backbone or create new one
        if pretrained_backbone is not None:
            self.backbone = pretrained_backbone
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            self.backbone = CRNNBackbone(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                conv_channels=conv_channels,
                conv_kernel_size=conv_kernel_size,
                conv_stride=conv_stride,
                bptt_length=bptt_length,
                use_bidirectional=use_bidirectional,
                init_weights=init_weights
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights(init_weights)
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.map = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        
        self.training_step_outputs = []
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
    
    def _init_classifier_weights(self, init_method="xavier"):
        """Initialize weights for the classifier head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif init_method == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif init_method == "random":
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step for supervised fine-tuning."""
        x, y = batch
        
        # Validate input data
        if batch_idx == 0:
            print(f"🔍 Training batch {batch_idx}: x shape={x.shape}, y shape={y.shape}")
        
        preds = self(x)
        loss = self.loss_fn(preds, y)
        
        self.training_step_outputs.append(loss.item())
        
        # Store predictions and targets for metrics computation
        self.train_preds.append(preds.detach())
        self.train_targets.append(y.detach())
        
        # Log loss
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute and log training metrics."""
        if len(self.training_step_outputs) > 0:
            avg_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.training_step_outputs.clear()
        else:
            # No training steps executed (e.g., when resuming from checkpoint)
            self.training_step_outputs.clear()
        
        # Compute training metrics
        if len(self.train_preds) > 0:
            try:
                # Concatenate all predictions and targets
                all_preds = torch.cat(self.train_preds, dim=0)
                all_targets = torch.cat(self.train_targets, dim=0)
                
                # Apply sigmoid to raw logits for metrics computation
                all_preds_probs = torch.sigmoid(all_preds)
                all_preds_probs = torch.clamp(all_preds_probs, min=1e-7, max=1.0-1e-7)
                all_targets = all_targets.int()
                
                # Compute metrics
                train_f1 = self.f1(all_preds_probs, all_targets)
                train_map = self.map(all_preds_probs, all_targets)
                train_auc = self.auc(all_preds_probs, all_targets)
                
                # Log metrics
                self.log('train_f1_eval', train_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('train_map_eval', train_map, on_step=False, on_epoch=True, prog_bar=True)
                self.log('train_auc_eval', train_auc, on_step=False, on_epoch=True, prog_bar=True)
                
                print(f"✅ Epoch {self.current_epoch}: train_f1={train_f1:.4f}, train_map={train_map:.4f}, train_auc={train_auc:.4f}")
                
            except Exception as e:
                print(f"⚠️ Warning: Error computing training metrics: {str(e)}")
        
        # Clear stored predictions and targets
        self.train_preds.clear()
        self.train_targets.clear()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        
        preds = self(x)
        loss = self.loss_fn(preds, y)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions and targets
        self.val_preds.append(preds.detach())
        self.val_targets.append(y.detach())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        if len(self.val_preds) > 0:
            try:
                # Concatenate all predictions and targets
                all_preds = torch.cat(self.val_preds, dim=0)
                all_targets = torch.cat(self.val_targets, dim=0)
                
                # Apply sigmoid to raw logits for metrics computation
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
        self.val_preds.clear()
        self.val_targets.clear()
    
    def configure_optimizers(self):
        """Configure optimizer for supervised fine-tuning."""
        # Only optimize parameters that require gradients
        params_to_optimize = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            params_to_optimize,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer


# ========================
# 8. Argument Parser
# ========================
parser = argparse.ArgumentParser(description="DINO-based Semi-Supervised Audio Classification with MEL Spectrograms and CRNN")
parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the model and metrics")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--audio_dir", type=str, default="../tmp/fsd50k/FSD50K.dev_audio", help="Directory containing raw audio files")
parser.add_argument("--dataset_type", type=str, default="complete", choices=["max10sec", "complete"], 
                   help="Dataset type: 'max10sec' for filtered dataset (max 10 seconds) or 'complete' for full dataset")

# Data parameters
parser.add_argument("--test_size", type=float, default=0.1, help="Test/validation size")
parser.add_argument("--labeled_ratio", type=float, default=0.1, help="Ratio of labeled training data (default: 0.1 for 10%%)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")

# MEL spectrogram parameters
parser.add_argument("--n_mels", type=int, default=128, help="Number of MEL filter banks")
parser.add_argument("--n_fft", type=int, default=2048, help="FFT window size for MEL spectrogram")
parser.add_argument("--hop_length", type=int, default=512, help="Hop length for MEL spectrogram")

# Model architecture parameters
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for GRU")
parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--conv_channels", type=int, nargs="+", default=[64, 128, 256], 
                   help="Number of channels for each convolutional layer in CRNN")
parser.add_argument("--conv_kernel_size", type=int, default=3, help="Kernel size for convolutional layers")
parser.add_argument("--conv_stride", type=int, default=1, help="Stride for convolutional layers")
parser.add_argument("--bptt_length", type=int, default=60, help="Length of sequences for truncated BPTT (0 = no truncation)")
parser.add_argument("--use_bidirectional", action="store_true", help="Use bidirectional GRU")

# DINO parameters
parser.add_argument("--dino_pretrain_epochs", type=int, default=100, help="Epochs for DINO pretraining")
parser.add_argument("--dino_momentum", type=float, default=0.996, help="EMA momentum for teacher updates")
parser.add_argument("--dino_temperature", type=float, default=0.07, help="Temperature for softmax")
parser.add_argument("--dino_center_momentum", type=float, default=0.9, help="Momentum for centering buffer")
parser.add_argument("--dino_proj_dim", type=int, default=256, help="Projection head output dimension")
parser.add_argument("--dino_proj_layers", type=int, default=3, help="Number of MLP layers in projection head")
parser.add_argument("--teacher_hidden_dim", type=int, default=None, help="Teacher hidden dim (if different from student)")
parser.add_argument("--teacher_num_layers", type=int, default=None, help="Teacher num layers (if different from student)")

# DINOv2 parameters
parser.add_argument("--use_dinov2", action="store_true", help="Use DINOv2 (combines DINO + iBOT) instead of DINO")
parser.add_argument("--ibot_weight", type=float, default=1.0, help="Weight for iBOT loss (patch-level)")
parser.add_argument("--koleo_weight", type=float, default=0.04, help="Weight for KoLeo regularizer")
parser.add_argument("--mask_ratio", type=float, default=0.6, help="Mask ratio for iBOT (fraction of patches to mask)")

# Fine-tuning parameters
parser.add_argument("--finetune_epochs", type=int, default=200, help="Epochs for supervised fine-tuning")
parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone during fine-tuning")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
parser.add_argument("--gradient_clip_val", type=float, default=10.0, help="Gradient clipping value")

# Augmentation parameters
parser.add_argument("--time_mask_param", type=int, default=27, help="Time masking parameter")
parser.add_argument("--freq_mask_param", type=int, default=12, help="Frequency masking parameter")
parser.add_argument("--use_mixup", action="store_true", help="Enable Mixup augmentation")
parser.add_argument("--use_cutmix", action="store_true", help="Enable CutMix augmentation")
parser.add_argument("--mixup_alpha", type=float, default=0.4, help="Mixup alpha parameter")

# Training parameters
parser.add_argument("--eval_interval", type=int, default=10, help="Interval for evaluating the model")
parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging metrics")
parser.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing scheduler")
parser.add_argument("--use_early_stopping", action="store_true", help="Use early stopping callback")
parser.add_argument("--early_stopping_patience", type=int, default=100, help="Patience for early stopping")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")

# Resume training
parser.add_argument("--resume_from_dino", type=str, default=None, help="Path to DINO pretrained checkpoint to resume from")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to supervised training checkpoint to resume from (for continuing fine-tuning)")
parser.add_argument("--skip_dino_pretrain", action="store_true", help="Skip DINO pretraining and go directly to fine-tuning")
parser.add_argument("--supervised_only", action="store_true", help="Train only with labeled data (supervised learning, no DINO pretraining)")
parser.add_argument("--pretrained_backbone_path", type=str, default=None, help="Path to pretrained backbone checkpoint for supervised training")
parser.add_argument("--init_weights", type=str, default="xavier", choices=["xavier", "normal", "random"], 
                   help="Weight initialization method for new models: 'xavier', 'normal', or 'random'")

args = parser.parse_args()


# ========================
# 9. Main Script
# ========================
if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    start_time = time.time()
    print(f"🔹 Starting DINO training script at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================
    # Device Setup
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    
    if args.use_gpu and torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0], device=device)
            print(f"\n🔹 Using device: {device}")
            print(f"🔹 GPU: {torch.cuda.get_device_name()}")
            torch.set_float32_matmul_precision('high')
            print("🔹 Enabled Tensor Cores for better performance")
        except Exception as e:
            print(f"⚠️ Warning: GPU compatibility issue detected: {str(e)}")
            print("🔹 Falling back to CPU")
            device = torch.device("cpu")
            args.use_gpu = False
    else:
        print(f"\n🔹 Using device: {device}")
    
    # ========================
    # Create Save Directory
    # ========================
    if os.path.exists(args.save_dir):
        if not os.path.isdir(args.save_dir):
            print(f"⚠️ {args.save_dir} exists as a file. Removing it to create a directory.")
            os.remove(args.save_dir)
            os.makedirs(args.save_dir, exist_ok=True)
    else:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Save args to JSON for reproducibility
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # ========================
    # Load Dataset
    # ========================
    # Select CSV file based on dataset type
    if args.dataset_type == "max10sec":
        csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    elif args.dataset_type == "complete":
        csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration.csv"
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")
    
    print(f"🔹 Loading CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    clip_ids = df["clip_id"].values
    labels = df.iloc[:, 2:-1].values
    AUDIO_DIR = args.audio_dir
    print(f"🔹 Audio directory: {AUDIO_DIR}")
    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(f"Audio directory not found at: {AUDIO_DIR}")
    
    print(f"🔹 Number of clips in CSV: {len(clip_ids)}")
    
    # Filter for available audio files
    print(f"🔹 Filtering for available audio files...")
    valid_indices = []
    valid_clip_ids = []
    valid_labels = []
    
    for idx, (clip_id, label) in enumerate(zip(clip_ids, labels)):
        audio_found = False
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            audio_path = os.path.join(AUDIO_DIR, f"{clip_id}.{ext}")
            if os.path.exists(audio_path):
                valid_indices.append(idx)
                valid_clip_ids.append(clip_id)
                valid_labels.append(label)
                audio_found = True
                break
    
    print(f"🔹 Found {len(valid_clip_ids)} audio files out of {len(clip_ids)} total clips")
    
    if len(valid_clip_ids) == 0:
        raise ValueError(f"No audio files found in {AUDIO_DIR}")
    
    # ========================
    # Split into Train/Val and Labeled/Unlabeled
    # ========================
    # First, create train/test split
    indices = np.arange(len(valid_clip_ids))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - args.test_size))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"🔹 Train split size: {len(train_indices)}")
    print(f"🔹 Validation split size: {len(val_indices)}")
    
    # Then, split training data into labeled/unlabeled
    labeled_size = int(len(train_indices) * args.labeled_ratio)
    labeled_train_indices = train_indices[:labeled_size]
    unlabeled_train_indices = train_indices[labeled_size:]
    
    print(f"🔹 Labeled training samples: {len(labeled_train_indices)} ({args.labeled_ratio*100:.1f}%)")
    print(f"🔹 Unlabeled training samples: {len(unlabeled_train_indices)} ({(1-args.labeled_ratio)*100:.1f}%)")
    
    # ========================
    # Create DINO Augmentation
    # ========================
    augmentation = DINOAugmentation(
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        mixup_alpha=args.mixup_alpha
    )
    
    # ========================
    # Create Datasets
    # ========================
    # Unlabeled dataset (for DINO pretraining) - with augmentation, no labels
    unlabeled_dataset = MELSpectrogramDataset(
        AUDIO_DIR, valid_clip_ids, valid_labels, indices=unlabeled_train_indices, 
        is_train=True, test_size=0.0, n_mels=args.n_mels, n_fft=args.n_fft, 
        hop_length=args.hop_length, use_dino_aug=True, augmentation=augmentation, 
        return_label=False, use_dinov2=args.use_dinov2, mask_ratio=args.mask_ratio
    )
    
    # Labeled dataset (for fine-tuning) - no augmentation, with labels
    labeled_dataset = MELSpectrogramDataset(
        AUDIO_DIR, valid_clip_ids, valid_labels, indices=labeled_train_indices, 
        is_train=True, test_size=0.0, n_mels=args.n_mels, n_fft=args.n_fft, 
        hop_length=args.hop_length, use_dino_aug=False, augmentation=None, 
        return_label=True
    )
    
    # Validation dataset - no augmentation, with labels
    val_dataset = MELSpectrogramDataset(
        AUDIO_DIR, valid_clip_ids, valid_labels, indices=val_indices, 
        is_train=False, test_size=0.0, n_mels=args.n_mels, n_fft=args.n_fft, 
        hop_length=args.hop_length, use_dino_aug=False, augmentation=None, 
        return_label=True
    )
    
    print(f"🔹 Unlabeled dataset size: {len(unlabeled_dataset)}")
    print(f"🔹 Labeled dataset size: {len(labeled_dataset)}")
    print(f"🔹 Validation dataset size: {len(val_dataset)}")
    
    # ========================
    # Create DataLoaders
    # ========================
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    labeled_loader = DataLoader(
        labeled_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # ========================
    # Determine Input Dimension
    # ========================
    sample_data = labeled_dataset[0][0]
    input_dim = sample_data.shape[0]  # n_mels
    num_classes = labeled_dataset[0][1].shape[0]
    
    print(f"🔹 Input dimension (n_mels): {input_dim}")
    print(f"🔹 Number of classes: {num_classes}")
    
    # ========================
    # Supervised-Only Mode Check
    # ========================
    if args.supervised_only:
        print(f"\n{'='*60}")
        print(f"SUPERVISED-ONLY MODE")
        print(f"Training only with labeled data ({len(labeled_dataset)} samples)")
        print(f"Skipping DINO pretraining")
        print(f"{'='*60}")
        args.skip_dino_pretrain = True
    
    # ========================
    # Phase 1: DINO Pretraining
    # ========================
    pretrained_backbone = None
    
    if not args.skip_dino_pretrain:
        print(f"\n{'='*60}")
        if args.use_dinov2:
            print(f"Phase 1: DINOv2 Pretraining (DINO + iBOT)")
        else:
            print(f"Phase 1: DINO Pretraining")
        print(f"{'='*60}")
        
        # Create DINO or DINOv2 model
        if args.use_dinov2:
            dino_model = DINOv2Model(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                conv_channels=args.conv_channels,
                conv_kernel_size=args.conv_kernel_size,
                conv_stride=args.conv_stride,
                bptt_length=args.bptt_length,
                use_bidirectional=args.use_bidirectional,
                teacher_hidden_dim=args.teacher_hidden_dim,
                teacher_num_layers=args.teacher_num_layers,
                dino_momentum=args.dino_momentum,
                dino_temperature=args.dino_temperature,
                dino_center_momentum=args.dino_center_momentum,
                dino_proj_dim=args.dino_proj_dim,
                dino_proj_layers=args.dino_proj_layers,
                ibot_weight=args.ibot_weight,
                koleo_weight=args.koleo_weight,
                mask_ratio=args.mask_ratio
            )
        else:
            dino_model = DINOModel(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            conv_channels=args.conv_channels,
            conv_kernel_size=args.conv_kernel_size,
            conv_stride=args.conv_stride,
            bptt_length=args.bptt_length,
            use_bidirectional=args.use_bidirectional,
            teacher_hidden_dim=args.teacher_hidden_dim,
            teacher_num_layers=args.teacher_num_layers,
            dino_momentum=args.dino_momentum,
            dino_temperature=args.dino_temperature,
            dino_center_momentum=args.dino_center_momentum,
            dino_proj_dim=args.dino_proj_dim,
            dino_proj_layers=args.dino_proj_layers
        )
        
        # Resume from checkpoint if specified
        if args.resume_from_dino is not None:
            print(f"🔹 Loading DINO checkpoint from: {args.resume_from_dino}")
            dino_model = DINOModel.load_from_checkpoint(args.resume_from_dino)
        
        # Create checkpoint callback
        dino_checkpoint_callback = ModelCheckpoint(
            monitor='dino_loss_epoch',
            dirpath=args.save_dir,
            filename='dino_pretrained',
            save_top_k=1,
            mode='min'
        )
        
        # Create CSV logger for DINO
        dino_csv_logger = CSVLogger(
            save_dir=args.save_dir,
            name="dino_metrics",
            version=None
        )
        
        # Configure trainer for DINO
        if args.use_gpu and torch.cuda.is_available():
            trainer_config = {'accelerator': 'gpu', 'devices': 1}
        else:
            trainer_config = {'accelerator': 'cpu', 'devices': 1}
        
        dino_trainer = pl.Trainer(
            max_epochs=args.dino_pretrain_epochs,
            callbacks=[dino_checkpoint_callback],
            default_root_dir=args.save_dir,
            logger=dino_csv_logger,
            log_every_n_steps=args.log_interval,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=args.gradient_accumulation_steps,
            **trainer_config
        )
        
        # Train DINO/DINOv2 model
        model_name = "DINOv2" if args.use_dinov2 else "DINO"
        print(f"🔹 Starting {model_name} pretraining for {args.dino_pretrain_epochs} epochs...")
        dino_trainer.fit(dino_model, unlabeled_loader)
        print(f"✅ {model_name} pretraining complete!")
        
        # Extract pretrained backbone from student network
        pretrained_backbone = dino_model.student_backbone
        print(f"🔹 Extracted pretrained backbone from student network")
    else:
        print(f"\n{'='*60}")
        print(f"Skipping DINO Pretraining (--skip_dino_pretrain flag set)")
        print(f"{'='*60}")
    
    # Load pretrained backbone if specified
    if args.pretrained_backbone_path is not None:
        print(f"\n🔹 Loading pretrained backbone from: {args.pretrained_backbone_path}")
        if not os.path.exists(args.pretrained_backbone_path):
            raise FileNotFoundError(f"Pretrained backbone not found at: {args.pretrained_backbone_path}")
        
        try:
            # Try to load as a full checkpoint
            checkpoint = torch.load(args.pretrained_backbone_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # PyTorch Lightning checkpoint
                state_dict = checkpoint['state_dict']
                # Remove 'backbone.' prefix if present
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
            else:
                # Assume it's a state dict or full model
                state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
            
            # Create new backbone and load weights
            pretrained_backbone = CRNNBackbone(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                conv_channels=args.conv_channels,
                conv_kernel_size=args.conv_kernel_size,
                conv_stride=args.conv_stride,
                bptt_length=args.bptt_length,
                use_bidirectional=args.use_bidirectional,
                init_weights=args.init_weights
            )
            pretrained_backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ Successfully loaded pretrained backbone")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load pretrained backbone: {str(e)}")
            print(f"🔹 Creating new backbone with {args.init_weights} initialization instead")
            pretrained_backbone = None
    
    # ========================
    # Phase 2: Supervised Fine-tuning
    # ========================
    print(f"\n{'='*60}")
    print(f"Phase 2: Supervised Fine-tuning")
    print(f"{'='*60}")
    
    # Create supervised model with pretrained backbone
    supervised_model = SupervisedModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        conv_stride=args.conv_stride,
        bptt_length=args.bptt_length,
        use_bidirectional=args.use_bidirectional,
        freeze_backbone=args.freeze_backbone,
        pretrained_backbone=pretrained_backbone,
        init_weights=args.init_weights
    )
    
    # Create checkpoint callback
    supervised_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    
    # Add early stopping if requested
    callbacks = [supervised_checkpoint_callback]
    if args.use_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # Create CSV logger for supervised training
    supervised_csv_logger = CSVLogger(
        save_dir=args.save_dir,
        name="supervised_metrics",
        version=None
    )
    
    # Configure trainer for supervised training
    if args.use_gpu and torch.cuda.is_available():
        trainer_config = {'accelerator': 'gpu', 'devices': 1}
    else:
        trainer_config = {'accelerator': 'cpu', 'devices': 1}
    
    supervised_trainer = pl.Trainer(
        max_epochs=args.finetune_epochs,
        callbacks=callbacks,
        default_root_dir=args.save_dir,
        logger=supervised_csv_logger,
        check_val_every_n_epoch=args.eval_interval,
        log_every_n_steps=args.log_interval,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.gradient_accumulation_steps,
        **trainer_config
    )
    
    # Train supervised model
    ckpt_path = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    if ckpt_path:
        print(f"🔹 Resuming supervised fine-tuning from checkpoint: {ckpt_path}")
        print(f"🔹 Continuing for {args.finetune_epochs} more epochs...")
    else:
        print(f"🔹 Starting supervised fine-tuning for {args.finetune_epochs} epochs...")
    supervised_trainer.fit(supervised_model, labeled_loader, val_loader, ckpt_path=ckpt_path)
    print("✅ Supervised fine-tuning complete!")
    
    # ========================
    # Final Summary
    # ========================
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"🔹 Total training time: {time.time() - start_time:.2f} seconds")
    print(f"🔹 Results saved to: {args.save_dir}")
    print(f"🔹 DINO pretrained model: {os.path.join(args.save_dir, 'dino_pretrained.ckpt')}")
    print(f"🔹 Best supervised model: {os.path.join(args.save_dir, 'best-checkpoint.ckpt')}")

# ASiT-based Semi-Supervised Audio Classification with MEL Spectrograms and Vision Transformer
# Based on ASiT: Audio Spectrogram vIsion Transformer for General Audio Representation
# arXiv:2211.13189 (Atito et al., 2022)
#
# Key differences from run_experiment_dino_lightning.py:
# - Backbone: CRNN (Conv1D+GRU) → AudioViT (Vision Transformer on 2D spectrograms)
# - Augmentation: DINO two-view → GMML patch corruption (student=corrupted, teacher=clean)
# - Loss: DINO cross-entropy → CLS_PATCH_Loss (contrastive on CLS + patch tokens)
# - Corruption types: gaussian noise, zeros, time masking, freq masking, inpainting
#
# Training stages:
# 1. ASIT pretraining on unlabeled data (via --asit_pretrain_epochs)
#    Student processes corrupted spectrograms, teacher processes clean ones.
#    CLS + patch contrastive loss aligns student representations with teacher.
# 2. Supervised fine-tuning for classification (via --finetune_epochs)
#    Uses CLS token from pretrained ViT backbone.


# python run_experiment_asit_lightning.py \
#   --save_dir "complete_asit_001" --asit_pretrain_epochs 100 --finetune_epochs 200 \
#   --vit_model small --labeled_ratio 0.1 --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_001" \
#   --supervised_only \
#   --finetune_epochs 500 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_002" \
#   --supervised_only \
#   --finetune_epochs 700 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 3e-5 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --use_early_stopping \
#   --early_stopping_patience 60 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_003" \
#   --supervised_only \
#   --finetune_epochs 700 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-3 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --use_early_stopping \
#   --early_stopping_patience 60 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_004" \
#   --supervised_only \
#   --finetune_epochs 700 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --use_early_stopping \
#   --early_stopping_patience 60 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_005" \
#   --supervised_only \
#   --finetune_epochs 700 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-6 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --use_early_stopping \
#   --early_stopping_patience 60 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_006" \
#   --supervised_only \
#   --finetune_epochs 1500 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --use_early_stopping \
#   --early_stopping_patience 60 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_007" \
#   --supervised_only \
#   --finetune_epochs 2000 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --use_early_stopping \
#   --early_stopping_patience 60 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_008" \
#   --supervised_only \
#   --finetune_epochs 700 \
#   --vit_model base \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_008" \
#   --supervised_only \
#   --finetune_epochs 700 \
#   --vit_model base \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_009" \
#   --supervised_only \
#   --finetune_epochs 700 \
#   --vit_model base \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "asit_supervised_010" \
#   --asit_pretrain_epochs 100 \
#   --finetune_epochs 500 \
#   --vit_model base \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --dropout 0.1 \
#   --eval_interval 1 \
#   --gradient_accumulation_steps 2 \
#   --use_gpu


# I haven't ran the ones below yet, but these are the commands I plan to run for the experiments in the paper. You can run them too if you want to reproduce the results or test the code.
# python run_experiment_asit_lightning.py \
#   --save_dir "complete_asit_001" \
#   --asit_pretrain_epochs 100 \
#   --finetune_epochs 200 \
#   --vit_model small \
#   --patch_size_freq 16 \
#   --patch_size_time 16 \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "complete_asit_002" \
#   --asit_pretrain_epochs 200 \
#   --finetune_epochs 500 \
#   --vit_model small \
#   --patch_size_freq 16 \
#   --patch_size_time 16 \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --use_gpu

# python run_experiment_asit_lightning.py \
#   --save_dir "complete_asit_001" \
#   --skip_asit_pretrain \
#   --resume_from_checkpoint "complete_asit_001/best-checkpoint.ckpt" \
#   --finetune_epochs 500 \
#   --vit_model small \
#   --labeled_ratio 0.1 \
#   --dataset_type "complete" \
#   --batch_size 32 \
#   --num_workers 4 \
#   --use_gpu

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import math
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC
import librosa
from pytorch_lightning.loggers import CSVLogger
import json
import multiprocessing
import time
import warnings
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

TARGET_LENGTH = 10 * 16000
SAMPLE_RATE = 16000


# ========================
# 1. GMML Augmentation
# ========================
class GMMLAugmentation:
    """
    GMML (Group Masked Model Learning) augmentation for ASiT.

    Applies random rectangular region corruptions to a spectrogram.
    Student receives corrupted view; teacher receives the clean original.
    Corruption types (from ASiT paper):
      - gaussian: inject Gaussian noise into the region
      - zeros:    zero out the region
      - time_mask: zero entire time frames that fall within the region
      - freq_mask: zero entire frequency bands that fall within the region
      - inpainting: replace region with content from another random location
    """

    def __init__(self, num_regions=4, region_min_size=0.1, region_max_size=0.4,
                 methods=('gaussian', 'zeros', 'time_mask', 'freq_mask', 'inpainting')):
        self.num_regions = num_regions
        self.region_min_size = region_min_size
        self.region_max_size = region_max_size
        self.methods = list(methods)

    def _get_random_region(self, n_mels, time_frames):
        """Sample a random rectangular region as (f0, t0, f1, t1)."""
        h_frac = np.random.uniform(self.region_min_size, self.region_max_size)
        w_frac = np.random.uniform(self.region_min_size, self.region_max_size)
        h = max(1, int(n_mels * h_frac))
        w = max(1, int(time_frames * w_frac))
        f0 = np.random.randint(0, max(1, n_mels - h + 1))
        t0 = np.random.randint(0, max(1, time_frames - w + 1))
        return f0, t0, f0 + h, t0 + w

    def _corrupt_region(self, spec, f0, f1, t0, t1, method):
        """Apply a single corruption type to the given rectangular region."""
        result = spec.copy()
        n_mels, time_frames = spec.shape

        if method == 'gaussian':
            region = spec[f0:f1, t0:t1]
            noise_std = np.std(region) if np.std(region) > 0 else 1.0
            result[f0:f1, t0:t1] = np.random.randn(*region.shape) * noise_std
        elif method == 'zeros':
            result[f0:f1, t0:t1] = 0.0
        elif method == 'time_mask':
            result[:, t0:t1] = 0.0
        elif method == 'freq_mask':
            result[f0:f1, :] = 0.0
        elif method == 'inpainting':
            h, w = f1 - f0, t1 - t0
            src_f = np.random.randint(0, max(1, n_mels - h + 1))
            src_t = np.random.randint(0, max(1, time_frames - w + 1))
            result[f0:f1, t0:t1] = spec[src_f:src_f + h, src_t:src_t + w]

        return result

    def apply(self, spec):
        """
        Apply GMML corruption.

        Returns:
            corrupted: (n_mels, time_frames) corrupted spectrogram for student
            clean:     (n_mels, time_frames) clean spectrogram for teacher
        """
        n_mels, time_frames = spec.shape
        clean = spec.copy()
        corrupted = spec.copy()

        for _ in range(self.num_regions):
            f0, t0, f1, t1 = self._get_random_region(n_mels, time_frames)
            method = np.random.choice(self.methods)
            corrupted = self._corrupt_region(corrupted, f0, f1, t0, t1, method)

        return corrupted, clean


# ========================
# 2. MEL Spectrogram Dataset
# ========================
class MELSpectrogramDataset(Dataset):
    """Dataset for MEL spectrograms with ASIT/GMML augmentation support."""

    def __init__(self, audio_dir, clip_ids, labels, indices=None, is_train=True,
                 target_length=TARGET_LENGTH, sample_rate=SAMPLE_RATE,
                 n_mels=128, n_fft=2048, hop_length=512,
                 use_asit_aug=False, augmentation=None, return_label=True):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_train = is_train
        self.use_asit_aug = use_asit_aug
        self.augmentation = augmentation
        self.return_label = return_label

        print(f"🔹 Looking for raw audio files in: {os.path.abspath(audio_dir)}")
        print(f"🔹 MEL parameters: n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}")
        if use_asit_aug:
            print(f"🔹 ASIT/GMML augmentation enabled")

        self.clip_ids = np.array(clip_ids)
        self.labels = np.array(labels) if labels is not None else None
        self.indices = indices if indices is not None else np.arange(len(self.clip_ids))

        print(f"🔹 {'Training' if is_train else 'Validation'} dataset size: {len(self.indices)}")

        if len(self.indices) == 0:
            raise ValueError(f"Empty {'training' if is_train else 'validation'} dataset.")

    def __len__(self):
        return len(self.indices)

    def _load_mel(self, clip_id):
        """Load audio and compute MEL spectrogram in dB scale."""
        audio_path = None
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            temp_path = os.path.join(self.audio_dir, f"{clip_id}.{ext}")
            if os.path.exists(temp_path):
                audio_path = temp_path
                break

        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for clip_id={clip_id}")

        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.target_length]

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length,
            fmin=0, fmax=self.sample_rate // 2
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    def __getitem__(self, idx):
        clip_idx = self.indices[idx]
        clip_id = self.clip_ids[clip_idx]
        label = self.labels[clip_idx] if self.labels is not None else None

        try:
            mel = self._load_mel(clip_id)

            if self.use_asit_aug and self.augmentation is not None and self.is_train:
                corrupted, clean = self.augmentation.apply(mel)
                corrupted_t = torch.tensor(corrupted, dtype=torch.float32)
                clean_t = torch.tensor(clean, dtype=torch.float32)
                if self.return_label and label is not None:
                    return corrupted_t, clean_t, torch.tensor(label, dtype=torch.float32)
                return corrupted_t, clean_t
            else:
                mel_t = torch.tensor(mel, dtype=torch.float32)
                if self.return_label and label is not None:
                    return mel_t, torch.tensor(label, dtype=torch.float32)
                return mel_t

        except Exception as e:
            print(f"Error loading {clip_id}: {e}")
            time_frames = (self.target_length // self.hop_length) + 1
            fallback = torch.zeros(self.n_mels, time_frames, dtype=torch.float32)
            if self.use_asit_aug and self.is_train:
                if self.return_label and label is not None:
                    return fallback, fallback, torch.tensor(label if label is not None else np.zeros(1), dtype=torch.float32)
                return fallback, fallback
            else:
                if self.return_label and label is not None:
                    return fallback, torch.tensor(label if label is not None else np.zeros(1), dtype=torch.float32)
                return fallback


# ========================
# 3. AudioViT (Vision Transformer backbone for audio)
# ========================
class PatchEmbed(nn.Module):
    """Embed a 2D spectrogram into a sequence of patch tokens via Conv2d."""

    def __init__(self, patch_size_freq=16, patch_size_time=16, embed_dim=384):
        super().__init__()
        self.patch_size_freq = patch_size_freq
        self.patch_size_time = patch_size_time
        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=(patch_size_freq, patch_size_time),
            stride=(patch_size_freq, patch_size_time)
        )

    def forward(self, x):
        # x: (B, n_mels, T)
        x = x.unsqueeze(1)         # (B, 1, n_mels, T)
        x = self.proj(x)           # (B, embed_dim, n_patches_freq, n_patches_time)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


VIT_CONFIGS = {
    'tiny':  {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
    'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base':  {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
}


class AudioViT(nn.Module):
    """
    Vision Transformer adapted for audio spectrograms.

    Treats the spectrogram as a 2D image divided into non-overlapping patches.
    Output:
      - CLS token (global representation) for classification / CLS loss
      - All patch tokens for patch-level ASIT loss
    """

    def __init__(self, n_mels=128, time_frames=313, patch_size_freq=16, patch_size_time=16,
                 vit_model='small', mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        cfg = VIT_CONFIGS[vit_model]
        embed_dim = cfg['embed_dim']
        depth = cfg['depth']
        num_heads = cfg['num_heads']

        self.patch_embed = PatchEmbed(patch_size_freq, patch_size_time, embed_dim)
        self.n_patches_freq = n_mels // patch_size_freq
        self.n_patches_time = time_frames // patch_size_time
        self.n_patches = self.n_patches_freq * self.n_patches_time

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.output_dim = embed_dim

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _interpolate_pos_embed(self, n_patches_actual):
        """Interpolate positional embedding if actual patch count differs from expected."""
        cls_pos = self.pos_embed[:, :1, :]          # (1, 1, D)
        patch_pos = self.pos_embed[:, 1:, :]         # (1, n_patches, D)
        D = patch_pos.shape[-1]

        patch_pos = patch_pos.reshape(1, self.n_patches_freq, self.n_patches_time, D)
        patch_pos = patch_pos.permute(0, 3, 1, 2)   # (1, D, freq, time)

        # Estimate actual time patches
        n_time_actual = n_patches_actual // self.n_patches_freq
        patch_pos = F.interpolate(
            patch_pos,
            size=(self.n_patches_freq, n_time_actual),
            mode='bilinear', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, D)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x, return_patch_tokens=False):
        """
        Args:
            x: (B, n_mels, T)
            return_patch_tokens: if True, also return patch token features

        Returns:
            cls_out: (B, embed_dim)
            patch_out: (B, n_patches, embed_dim)  [only if return_patch_tokens=True]
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)           # (B, n_patches_actual, D)
        n_patches_actual = tokens.shape[1]

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+n_patches_actual, D)

        if n_patches_actual == self.n_patches:
            tokens = tokens + self.pos_embed
        else:
            tokens = tokens + self._interpolate_pos_embed(n_patches_actual).to(tokens.device)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]      # (B, D)
        patch_out = tokens[:, 1:]   # (B, n_patches_actual, D)

        if return_patch_tokens:
            return cls_out, patch_out
        return cls_out


# ========================
# 4. Projection Heads
# ========================
class CLSProjectionHead(nn.Module):
    """MLP projection head for CLS token (with BatchNorm, as in DINO)."""

    def __init__(self, in_dim, out_dim=256, num_layers=3, hidden_dim=2048, use_bn=True):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(in_ch, out_ch))
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        # x: (B, in_dim)
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class PatchProjectionHead(nn.Module):
    """
    Shared MLP projection head applied independently to each patch token.
    Uses LayerNorm (instead of BatchNorm) so it applies cleanly per position.
    """

    def __init__(self, in_dim, out_dim=256, num_layers=3, hidden_dim=2048):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(in_ch, out_ch))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(hidden_dim, eps=1e-6))
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        # x: (B, N, in_dim)
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x.reshape(B, N, -1)


# ========================
# 5. CLS_PATCH_Loss
# ========================
def cls_patch_loss(student_cls, student_patches,
                   teacher_cls, teacher_patches,
                   center_cls, center_patch,
                   temperature=0.07, patch_loss_weight=1.0):
    """
    ASiT CLS_PATCH contrastive loss.

    CLS loss: student CLS token matches teacher CLS token (after centering + softmax).
    Patch loss: student patch tokens match teacher patch tokens at all positions.

    Args:
        student_cls:     (B, proj_dim)
        student_patches: (B, N, proj_dim)
        teacher_cls:     (B, proj_dim)  -- no grad expected
        teacher_patches: (B, N, proj_dim) -- no grad expected
        center_cls:      (proj_dim,)
        center_patch:    (proj_dim,)
        temperature:     scalar
        patch_loss_weight: scalar weight for patch loss term

    Returns:
        total_loss, cls_loss_val, patch_loss_val
    """
    # --- CLS loss (same formulation as DINO) ---
    teacher_cls_centered = (teacher_cls - center_cls) / temperature
    teacher_cls_probs = F.softmax(teacher_cls_centered, dim=-1)
    student_cls_log = F.log_softmax(student_cls, dim=-1)
    cls_loss_val = -torch.sum(teacher_cls_probs * student_cls_log, dim=-1).mean()

    # --- Patch loss (averaged over all N patch positions) ---
    # Broadcast center: (B, N, proj_dim)
    teacher_patch_centered = (teacher_patches - center_patch.unsqueeze(0).unsqueeze(0)) / temperature
    teacher_patch_probs = F.softmax(teacher_patch_centered, dim=-1)
    student_patch_log = F.log_softmax(student_patches, dim=-1)
    patch_loss_val = -torch.sum(teacher_patch_probs * student_patch_log, dim=-1).mean()

    total_loss = cls_loss_val + patch_loss_weight * patch_loss_val
    return total_loss, cls_loss_val, patch_loss_val


# ========================
# 6. ASITModel (Lightning Module for pretraining)
# ========================
class ASITModel(pl.LightningModule):
    """
    ASiT self-supervised pretraining model.

    Dual student-teacher ViT with EMA updates.
    Student receives GMML-corrupted spectrograms; teacher receives clean ones.
    Loss = CLS_PATCH contrastive loss.
    """

    def __init__(self, n_mels, time_frames, patch_size_freq, patch_size_time,
                 vit_model, dropout, lr, weight_decay,
                 asit_pretrain_epochs=100, warmup_epochs=10,
                 asit_momentum=0.996, asit_temperature=0.07, asit_center_momentum=0.9,
                 asit_proj_dim=256, asit_proj_layers=3, patch_loss_weight=1.0):
        super().__init__()
        self.save_hyperparameters()

        # Student ViT
        self.student = AudioViT(
            n_mels=n_mels, time_frames=time_frames,
            patch_size_freq=patch_size_freq, patch_size_time=patch_size_time,
            vit_model=vit_model, dropout=dropout
        )
        embed_dim = self.student.output_dim

        # Teacher ViT (EMA, no grad)
        self.teacher = AudioViT(
            n_mels=n_mels, time_frames=time_frames,
            patch_size_freq=patch_size_freq, patch_size_time=patch_size_time,
            vit_model=vit_model, dropout=dropout
        )
        self._copy_student_to_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student projection heads
        hidden_proj = max(embed_dim * 2, 2048)
        self.student_cls_head = CLSProjectionHead(embed_dim, asit_proj_dim, asit_proj_layers, hidden_proj)
        self.student_patch_head = PatchProjectionHead(embed_dim, asit_proj_dim, asit_proj_layers, hidden_proj)

        # Teacher projection heads (EMA, no grad)
        self.teacher_cls_head = CLSProjectionHead(embed_dim, asit_proj_dim, asit_proj_layers, hidden_proj)
        self.teacher_patch_head = PatchProjectionHead(embed_dim, asit_proj_dim, asit_proj_layers, hidden_proj)
        self._copy_student_heads_to_teacher()
        for p in list(self.teacher_cls_head.parameters()) + list(self.teacher_patch_head.parameters()):
            p.requires_grad = False

        # Center buffers for numerical stability (EMA-updated)
        self.register_buffer('center_cls', torch.zeros(asit_proj_dim))
        self.register_buffer('center_patch', torch.zeros(asit_proj_dim))

        self.training_step_outputs = []

    # -------- helpers --------
    def _copy_student_to_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.data.copy_(s.data)

    def _copy_student_heads_to_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.student_cls_head.parameters(), self.teacher_cls_head.parameters()):
                t.data.copy_(s.data)
            for s, t in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
                t.data.copy_(s.data)

    def _ema_update(self):
        m = self.hparams.asit_momentum
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.data = m * t.data + (1 - m) * s.data
            for s, t in zip(self.student_cls_head.parameters(), self.teacher_cls_head.parameters()):
                t.data = m * t.data + (1 - m) * s.data
            for s, t in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
                t.data = m * t.data + (1 - m) * s.data

    # -------- forward --------
    def forward(self, x):
        return self.student(x)

    # -------- training --------
    def training_step(self, batch, batch_idx):
        corrupted, clean = batch

        # Student forward (receives corrupted spectrogram)
        s_cls, s_patches = self.student(corrupted, return_patch_tokens=True)
        s_cls_proj = self.student_cls_head(s_cls)
        s_patch_proj = self.student_patch_head(s_patches)

        # Teacher forward on clean spectrogram (no gradient)
        with torch.no_grad():
            t_cls, t_patches = self.teacher(clean, return_patch_tokens=True)
            t_cls_proj = self.teacher_cls_head(t_cls)
            t_patch_proj = self.teacher_patch_head(t_patches)

        # Compute CLS_PATCH loss
        loss, cls_loss, patch_loss = cls_patch_loss(
            s_cls_proj, s_patch_proj,
            t_cls_proj, t_patch_proj,
            self.center_cls, self.center_patch,
            self.hparams.asit_temperature,
            self.hparams.patch_loss_weight
        )

        # Update center buffers via EMA
        cm = self.hparams.asit_center_momentum
        with torch.no_grad():
            self.center_cls = cm * self.center_cls + (1 - cm) * t_cls_proj.mean(0).detach()
            self.center_patch = cm * self.center_patch + (1 - cm) * t_patch_proj.mean([0, 1]).detach()

        self.training_step_outputs.append({
            'loss': loss.item(),
            'cls_loss': cls_loss.item(),
            'patch_loss': patch_loss.item()
        })

        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('asit_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log('cls_loss', cls_loss, on_step=True, on_epoch=False, prog_bar=False)
            self.log('patch_loss', patch_loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._ema_update()

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            avg = {k: sum(d[k] for d in self.training_step_outputs) / len(self.training_step_outputs)
                   for k in ('loss', 'cls_loss', 'patch_loss')}
            self.log('asit_loss_epoch', avg['loss'], on_step=False, on_epoch=True, prog_bar=True)
            self.log('cls_loss_epoch', avg['cls_loss'], on_step=False, on_epoch=True, prog_bar=False)
            self.log('patch_loss_epoch', avg['patch_loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        params = (
            list(self.student.parameters())
            + list(self.student_cls_head.parameters())
            + list(self.student_patch_head.parameters())
        )
        optimizer = optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999), eps=1e-8
        )

        total_epochs = self.hparams.asit_pretrain_epochs
        warmup = self.hparams.warmup_epochs

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]


# ========================
# 7. SupervisedModel (Lightning Module for fine-tuning)
# ========================
class SupervisedModel(pl.LightningModule):
    """
    Supervised fine-tuning model.

    Uses the CLS token from the pretrained AudioViT backbone for multi-label classification.
    """

    def __init__(self, n_mels, time_frames, patch_size_freq, patch_size_time,
                 vit_model, num_classes, dropout, lr, weight_decay,
                 freeze_backbone=False, pretrained_backbone=None):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_backbone'])
        self.freeze_backbone = freeze_backbone

        if pretrained_backbone is not None:
            self.backbone = pretrained_backbone
        else:
            self.backbone = AudioViT(
                n_mels=n_mels, time_frames=time_frames,
                patch_size_freq=patch_size_freq, patch_size_time=patch_size_time,
                vit_model=vit_model, dropout=dropout
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        embed_dim = self.backbone.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self._init_classifier_weights()

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.map = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.auc = MultilabelAUROC(num_labels=num_classes, average="macro")

        self.training_step_outputs = []
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []

    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        cls = self.backbone(x)   # (B, embed_dim) — CLS token only
        return self.classifier(cls)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if batch_idx == 0:
            print(f"🔍 Training batch {batch_idx}: x shape={x.shape}, y shape={y.shape}")
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.training_step_outputs.append(loss.item())
        self.train_preds.append(preds.detach())
        self.train_targets.append(y.detach())
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            avg_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()

        if self.train_preds:
            try:
                all_preds = torch.cat(self.train_preds, dim=0)
                all_targets = torch.cat(self.train_targets, dim=0)
                probs = torch.clamp(torch.sigmoid(all_preds), 1e-7, 1 - 1e-7)
                targets_int = all_targets.int()
                train_f1 = self.f1(probs, targets_int)
                train_map = self.map(probs, targets_int)
                train_auc = self.auc(probs, targets_int)
                self.log('train_f1_eval', train_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('train_map_eval', train_map, on_step=False, on_epoch=True, prog_bar=True)
                self.log('train_auc_eval', train_auc, on_step=False, on_epoch=True, prog_bar=True)
                print(f"✅ Epoch {self.current_epoch}: train_f1={train_f1:.4f}, "
                      f"train_map={train_map:.4f}, train_auc={train_auc:.4f}")
            except Exception as e:
                print(f"⚠️ Warning: Error computing training metrics: {e}")
        self.train_preds.clear()
        self.train_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_preds.append(preds.detach())
        self.val_targets.append(y.detach())
        return loss

    def on_validation_epoch_end(self):
        if self.val_preds:
            try:
                all_preds = torch.cat(self.val_preds, dim=0)
                all_targets = torch.cat(self.val_targets, dim=0)
                probs = torch.clamp(torch.sigmoid(all_preds), 1e-7, 1 - 1e-7)
                targets_int = all_targets.int()
                val_f1 = self.f1(probs, targets_int)
                val_map = self.map(probs, targets_int)
                val_auc = self.auc(probs, targets_int)
                self.log('val_f1', val_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_map', val_map, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_auc', val_auc, on_step=False, on_epoch=True, prog_bar=True)
                print(f"✅ Epoch {self.current_epoch}: val_f1={val_f1:.4f}, "
                      f"val_map={val_map:.4f}, val_auc={val_auc:.4f}")
            except Exception as e:
                print(f"⚠️ Warning: Error computing validation metrics: {e}")
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999), eps=1e-8
        )


# ========================
# 8. Argument Parser
# ========================
parser = argparse.ArgumentParser(
    description="ASiT-based Semi-Supervised Audio Classification with MEL Spectrograms and Vision Transformer"
)

# General
parser.add_argument("--save_dir", type=str, default="results",
                    help="Directory to save the model and metrics")
parser.add_argument("--use_gpu", action="store_true",
                    help="Use GPU if available")
parser.add_argument("--audio_dir", type=str,
                    default="../tmp/fsd50k/FSD50K.dev_audio",
                    help="Directory containing raw audio files")
parser.add_argument("--dataset_type", type=str, default="complete",
                    choices=["max10sec", "complete"],
                    help="Dataset type: 'max10sec' or 'complete'")

# Data
parser.add_argument("--test_size", type=float, default=0.1,
                    help="Validation split fraction")
parser.add_argument("--labeled_ratio", type=float, default=0.1,
                    help="Fraction of training data treated as labeled (default: 0.1 = 10%%)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--num_workers", type=int, default=1,
                    help="DataLoader worker processes")

# MEL spectrogram
parser.add_argument("--n_mels", type=int, default=128,
                    help="Number of MEL filter banks")
parser.add_argument("--n_fft", type=int, default=2048,
                    help="FFT window size")
parser.add_argument("--hop_length", type=int, default=512,
                    help="Hop length for MEL spectrogram")

# ViT architecture
parser.add_argument("--vit_model", type=str, default="small",
                    choices=["tiny", "small", "base"],
                    help="ViT variant: tiny (192d), small (384d), base (768d)")
parser.add_argument("--patch_size_freq", type=int, default=16,
                    help="Patch height in frequency dimension")
parser.add_argument("--patch_size_time", type=int, default=16,
                    help="Patch width in time dimension")
parser.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate inside ViT blocks")

# ASIT pretraining
parser.add_argument("--asit_pretrain_epochs", type=int, default=100,
                    help="Epochs for ASIT pretraining")
parser.add_argument("--warmup_epochs", type=int, default=10,
                    help="LR warmup epochs during ASIT pretraining")
parser.add_argument("--asit_momentum", type=float, default=0.996,
                    help="EMA momentum for teacher updates")
parser.add_argument("--asit_temperature", type=float, default=0.07,
                    help="Temperature for softmax in CLS_PATCH loss")
parser.add_argument("--asit_center_momentum", type=float, default=0.9,
                    help="Momentum for center buffer EMA updates")
parser.add_argument("--asit_proj_dim", type=int, default=256,
                    help="Projection head output dimension")
parser.add_argument("--asit_proj_layers", type=int, default=3,
                    help="Number of MLP layers in projection heads")
parser.add_argument("--patch_loss_weight", type=float, default=1.0,
                    help="Weight for patch-level loss term (total = cls_loss + w * patch_loss)")

# GMML augmentation
parser.add_argument("--num_gmml_regions", type=int, default=4,
                    help="Number of GMML corruption regions per spectrogram")
parser.add_argument("--gmml_region_min_size", type=float, default=0.1,
                    help="Minimum region size as fraction of spectrogram")
parser.add_argument("--gmml_region_max_size", type=float, default=0.4,
                    help="Maximum region size as fraction of spectrogram")
parser.add_argument("--gmml_methods", type=str, nargs="+",
                    default=["gaussian", "zeros", "time_mask", "freq_mask", "inpainting"],
                    help="GMML corruption methods to sample from")

# Supervised fine-tuning
parser.add_argument("--finetune_epochs", type=int, default=200,
                    help="Epochs for supervised fine-tuning")
parser.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone during fine-tuning (only train classifier head)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5,
                    help="Weight decay")
parser.add_argument("--gradient_clip_val", type=float, default=10.0,
                    help="Gradient clipping value")

# Training loop
parser.add_argument("--eval_interval", type=int, default=10,
                    help="Validate every N epochs during fine-tuning")
parser.add_argument("--log_interval", type=int, default=10,
                    help="Log every N steps")
parser.add_argument("--use_early_stopping", action="store_true",
                    help="Use early stopping on val_loss")
parser.add_argument("--early_stopping_patience", type=int, default=100,
                    help="Patience for early stopping")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Gradient accumulation steps")

# Checkpointing / resume
parser.add_argument("--resume_from_asit", type=str, default=None,
                    help="Path to ASIT checkpoint to resume pretraining from")
parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to supervised checkpoint to resume fine-tuning from")
parser.add_argument("--skip_asit_pretrain", action="store_true",
                    help="Skip ASIT pretraining and go directly to supervised fine-tuning")
parser.add_argument("--supervised_only", action="store_true",
                    help="Supervised training only (no ASIT pretraining)")
parser.add_argument("--pretrained_asit_path", type=str, default=None,
                    help="Path to a pretrained ASIT checkpoint for transfer learning "
                         "(loads weights into student, then continues ASIT training)")

args = parser.parse_args()


# ========================
# 9. Main Script
# ========================
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    print(f"🔹 Starting ASiT training script at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Device setup ----
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if args.use_gpu and torch.cuda.is_available():
        try:
            torch.tensor([1.0], device=device)
            print(f"\n🔹 Using device: {device}")
            print(f"🔹 GPU: {torch.cuda.get_device_name()}")
            torch.set_float32_matmul_precision('high')
            print("🔹 Enabled Tensor Cores for better performance")
        except Exception as e:
            print(f"⚠️ GPU issue: {e}. Falling back to CPU.")
            device = torch.device("cpu")
            args.use_gpu = False
    else:
        print(f"\n🔹 Using device: {device}")

    # ---- Save directory ----
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Load dataset CSV ----
    if args.dataset_type == "max10sec":
        csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
    else:
        csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration.csv"

    print(f"🔹 Loading CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    clip_ids = df["clip_id"].values
    labels = df.iloc[:, 2:-1].values
    AUDIO_DIR = args.audio_dir

    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    print(f"🔹 Total clips in CSV: {len(clip_ids)}")

    # ---- Filter for available audio files ----
    print("🔹 Filtering for available audio files...")
    valid_clip_ids, valid_labels = [], []
    for clip_id, label in zip(clip_ids, labels):
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            if os.path.exists(os.path.join(AUDIO_DIR, f"{clip_id}.{ext}")):
                valid_clip_ids.append(clip_id)
                valid_labels.append(label)
                break

    print(f"🔹 Found {len(valid_clip_ids)} audio files out of {len(clip_ids)} clips")
    if not valid_clip_ids:
        raise ValueError(f"No audio files found in {AUDIO_DIR}")

    # ---- Train / val / labeled / unlabeled split ----
    indices = np.arange(len(valid_clip_ids))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - args.test_size))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    labeled_size = int(len(train_indices) * args.labeled_ratio)
    labeled_train_indices = train_indices[:labeled_size]

    print(f"🔹 Train split: {len(train_indices)} | Val split: {len(val_indices)}")
    print(f"🔹 Labeled: {len(labeled_train_indices)} ({args.labeled_ratio*100:.1f}%) | "
          f"Unlabeled: {len(train_indices) - labeled_size} ({(1-args.labeled_ratio)*100:.1f}%)")

    # ---- Compute expected time_frames ----
    time_frames = (TARGET_LENGTH // args.hop_length) + 1
    print(f"🔹 Spectrogram shape: ({args.n_mels}, {time_frames})")
    print(f"🔹 ViT patch size: ({args.patch_size_freq}, {args.patch_size_time})")
    n_patches_freq = args.n_mels // args.patch_size_freq
    n_patches_time = time_frames // args.patch_size_time
    print(f"🔹 Patch grid: {n_patches_freq}×{n_patches_time} = {n_patches_freq * n_patches_time} patches")

    # ---- GMML Augmentation ----
    augmentation = GMMLAugmentation(
        num_regions=args.num_gmml_regions,
        region_min_size=args.gmml_region_min_size,
        region_max_size=args.gmml_region_max_size,
        methods=args.gmml_methods
    )

    # ---- Datasets ----
    # ASIT pretraining: all training data, GMML augmentation, no labels
    asit_pretrain_dataset = MELSpectrogramDataset(
        AUDIO_DIR, valid_clip_ids, valid_labels, indices=train_indices,
        is_train=True, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
        use_asit_aug=True, augmentation=augmentation, return_label=False
    )

    # Labeled dataset: labeled subset, no augmentation, with labels
    labeled_dataset = MELSpectrogramDataset(
        AUDIO_DIR, valid_clip_ids, valid_labels, indices=labeled_train_indices,
        is_train=True, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
        use_asit_aug=False, augmentation=None, return_label=True
    )

    # Validation dataset: no augmentation, with labels
    val_dataset = MELSpectrogramDataset(
        AUDIO_DIR, valid_clip_ids, valid_labels, indices=val_indices,
        is_train=False, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
        use_asit_aug=False, augmentation=None, return_label=True
    )

    print(f"🔹 ASIT pretrain dataset: {len(asit_pretrain_dataset)} samples")
    print(f"🔹 Labeled dataset:       {len(labeled_dataset)} samples")
    print(f"🔹 Validation dataset:    {len(val_dataset)} samples")

    # ---- DataLoaders ----
    persistent = args.num_workers > 0
    asit_pretrain_loader = DataLoader(
        asit_pretrain_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent
    )
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent
    )

    # Infer num_classes from labeled data
    num_classes = labeled_dataset[0][1].shape[0]
    print(f"🔹 Number of classes: {num_classes}")

    # ---- Supervised-only mode ----
    if args.supervised_only:
        print(f"\n{'='*60}")
        print(f"SUPERVISED-ONLY MODE — skipping ASIT pretraining")
        print(f"{'='*60}")
        args.skip_asit_pretrain = True

    # ========================
    # Phase 1: ASIT Pretraining
    # ========================
    pretrained_backbone = None

    if not args.skip_asit_pretrain:
        print(f"\n{'='*60}")
        print(f"Phase 1: ASIT Self-Supervised Pretraining")
        print(f"  ViT: {args.vit_model} | patches: {args.patch_size_freq}×{args.patch_size_time}")
        print(f"  Epochs: {args.asit_pretrain_epochs} | Warmup: {args.warmup_epochs}")
        print(f"  GMML regions: {args.num_gmml_regions} | methods: {args.gmml_methods}")
        print(f"{'='*60}")

        asit_model = ASITModel(
            n_mels=args.n_mels,
            time_frames=time_frames,
            patch_size_freq=args.patch_size_freq,
            patch_size_time=args.patch_size_time,
            vit_model=args.vit_model,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            asit_pretrain_epochs=args.asit_pretrain_epochs,
            warmup_epochs=args.warmup_epochs,
            asit_momentum=args.asit_momentum,
            asit_temperature=args.asit_temperature,
            asit_center_momentum=args.asit_center_momentum,
            asit_proj_dim=args.asit_proj_dim,
            asit_proj_layers=args.asit_proj_layers,
            patch_loss_weight=args.patch_loss_weight
        )

        # Load pretrained ASIT weights for transfer learning
        if args.pretrained_asit_path is not None:
            print(f"\n🔹 Loading pretrained ASIT model from: {args.pretrained_asit_path}")
            if not os.path.exists(args.pretrained_asit_path):
                raise FileNotFoundError(f"Pretrained ASIT checkpoint not found: {args.pretrained_asit_path}")
            try:
                checkpoint = torch.load(args.pretrained_asit_path, map_location=device)
                pretrained_sd = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
                model_sd = asit_model.state_dict()
                matched, skipped = [], []
                for k, v in pretrained_sd.items():
                    if k in model_sd and model_sd[k].shape == v.shape:
                        model_sd[k] = v
                        matched.append(k)
                    else:
                        skipped.append(k)
                asit_model.load_state_dict(model_sd, strict=False)
                print(f"✅ Loaded pretrained ASIT: {len(matched)} matched, {len(skipped)} skipped")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained ASIT: {e}. Starting from scratch.")

        # Resume full checkpoint if specified
        if args.resume_from_asit is not None:
            print(f"🔹 Resuming ASIT from checkpoint: {args.resume_from_asit}")
            asit_model = ASITModel.load_from_checkpoint(args.resume_from_asit)

        asit_checkpoint_cb = ModelCheckpoint(
            monitor='asit_loss_epoch',
            dirpath=args.save_dir,
            filename='asit_pretrained',
            save_top_k=1,
            mode='min'
        )

        asit_csv_logger = CSVLogger(
            save_dir=args.save_dir, name="asit_metrics", version=None
        )

        trainer_cfg = {'accelerator': 'gpu', 'devices': 1} if (args.use_gpu and torch.cuda.is_available()) else {'accelerator': 'cpu', 'devices': 1}

        asit_trainer = pl.Trainer(
            max_epochs=args.asit_pretrain_epochs,
            callbacks=[asit_checkpoint_cb],
            default_root_dir=args.save_dir,
            logger=asit_csv_logger,
            log_every_n_steps=args.log_interval,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=args.gradient_accumulation_steps,
            **trainer_cfg
        )

        print(f"🔹 Starting ASIT pretraining for {args.asit_pretrain_epochs} epochs...")
        asit_trainer.fit(asit_model, asit_pretrain_loader)
        print("✅ ASIT pretraining complete!")

        pretrained_backbone = asit_model.student
        print("🔹 Extracted pretrained backbone (student ViT) for fine-tuning")

    else:
        print(f"\n{'='*60}")
        print("Skipping ASIT pretraining (--skip_asit_pretrain or --supervised_only)")
        print(f"{'='*60}")

    # ========================
    # Phase 2: Supervised Fine-tuning
    # ========================
    print(f"\n{'='*60}")
    print("Phase 2: Supervised Fine-tuning")
    print(f"  ViT: {args.vit_model} | freeze_backbone={args.freeze_backbone}")
    print(f"  Epochs: {args.finetune_epochs} | labeled samples: {len(labeled_dataset)}")
    print(f"{'='*60}")

    supervised_model = SupervisedModel(
        n_mels=args.n_mels,
        time_frames=time_frames,
        patch_size_freq=args.patch_size_freq,
        patch_size_time=args.patch_size_time,
        vit_model=args.vit_model,
        num_classes=num_classes,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        pretrained_backbone=pretrained_backbone
    )

    supervised_checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    callbacks = [supervised_checkpoint_cb]
    if args.use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            verbose=True,
            mode='min'
        ))

    supervised_csv_logger = CSVLogger(
        save_dir=args.save_dir, name="supervised_metrics", version=None
    )

    trainer_cfg = {'accelerator': 'gpu', 'devices': 1} if (args.use_gpu and torch.cuda.is_available()) else {'accelerator': 'cpu', 'devices': 1}

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
        **trainer_cfg
    )

    ckpt_path = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    if ckpt_path:
        print(f"🔹 Resuming supervised fine-tuning from: {ckpt_path}")
    else:
        print(f"🔹 Starting supervised fine-tuning for {args.finetune_epochs} epochs...")

    supervised_trainer.fit(supervised_model, labeled_loader, val_loader, ckpt_path=ckpt_path)
    print("✅ Supervised fine-tuning complete!")

    # ========================
    # Summary
    # ========================
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"🔹 Total time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"🔹 Results saved to: {args.save_dir}")
    if not args.skip_asit_pretrain:
        print(f"🔹 ASIT pretrained model: {os.path.join(args.save_dir, 'asit_pretrained.ckpt')}")
    print(f"🔹 Best supervised model: {os.path.join(args.save_dir, 'best-checkpoint.ckpt')}")

"""
Extract the necessary classes from run_experiment_gru_lightning.py to avoid import conflicts.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC
from utils import asymmetric_loss, MeanContrastiveRankingLoss, wu_auc_loss, combined_wu_bce_loss, combined_wu_asymmetric_loss, combined_asymmetric_bce_loss

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, clip_ids, labels, indices=None, is_train=True, test_size=0.1, random_state=42):
        self.embedding_dir = embedding_dir
        self.is_train = is_train
        
        print(f"🔹 Looking for embedding files in: {os.path.abspath(embedding_dir)}")
        
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
            raise ValueError(f"No embedding files found in {os.path.abspath(embedding_dir)}. "
                           f"Please check if the embedding files exist and are named correctly "
                           f"(should be named like 'clip_id.npy').")
        
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
        
        # Additional safety checks
        if len(self.indices) == 0:
            raise ValueError(f"Empty {'training' if is_train else 'validation'} dataset. "
                           f"This might be due to an incorrect test_size parameter or insufficient data.")
    
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

class RawAudioDataset(Dataset):
    """Dataset class for raw audio files with learnable feature extraction."""
    
    def __init__(self, audio_dir, clip_ids, labels, indices=None, is_train=True, test_size=0.1, random_state=42, 
                 target_length=160000, sample_rate=16000, window_size=1024, hop_size=512):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.window_size = window_size  # Size of each time window
        self.hop_size = hop_size        # Stride between windows
        self.is_train = is_train
        
        print(f"🔹 Looking for raw audio files in: {os.path.abspath(audio_dir)}")
        print(f"🔹 Window size: {window_size}, Hop size: {hop_size}")
        
        # Store the pre-filtered data (no need to filter again)
        self.clip_ids = np.array(clip_ids)
        self.labels = np.array(labels)
        
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
        label = self.labels[clip_idx]
        
        # Find audio file (we know it exists since we pre-filtered)
        audio_path = None
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            temp_path = os.path.join(self.audio_dir, f"{clip_id}.{ext}")
            if os.path.exists(temp_path):
                audio_path = temp_path
                break
        
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for {clip_id} - this should not happen with pre-filtered data")
        
        # Load and preprocess audio
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or truncate to target length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.target_length]
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            return audio_tensor, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            # Return zero tensor as fallback
            return torch.zeros(self.target_length, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class LitRNNClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, lr, weight_decay, dropout, 
                 loss_fn="bce", loss_margin=0.1, gamma_pos=0.0, gamma_neg=4.0, wu_weight=0.5, bce_weight=0.5,
                 feature_mode="wav2vec", window_size=1024, hop_size=512, use_scheduler=False,
                 bptt_length=50, use_attention=False, use_bidirectional=False, attention_heads=8):
        super().__init__()
        self.save_hyperparameters()
        self.feature_mode = feature_mode
        self.window_size = window_size
        self.hop_size = hop_size
        self.bptt_length = bptt_length
        self.use_attention = use_attention
        self.use_bidirectional = use_bidirectional
        
        # Feature extraction layers (for both modes to maintain consistency)
        if feature_mode == "raw":
            # Learnable feature extraction: raw audio window -> 768 features
            # Use reduced dropout in feature extractor to prevent vanishing gradients
            feature_dropout = max(0.05, dropout * 0.3)  # Further reduce dropout for better gradient flow
            
            # Create simplified feature extractor to prevent gradient issues
            self.feature_extractor = nn.ModuleList([
                nn.Linear(window_size, 768),  # Direct mapping to avoid intermediate layers
            ])
            
            # Normalization and activation layers
            self.feature_norms = nn.ModuleList([
                nn.LayerNorm(768),
            ])
            
            self.feature_activations = nn.ModuleList([
                nn.LeakyReLU(0.1),
            ])
            
            self.feature_dropouts = nn.ModuleList([
                nn.Dropout(feature_dropout),
            ])
            # Update input dimension for GRU
            gru_input_dim = 768
        else:
            # Wav2Vec mode: add a lightweight feature projection layer for consistency
            # This ensures both architectures have similar structure and training dynamics
            feature_dropout = max(0.05, dropout * 0.3)  # Same dropout strategy
            
            # Lightweight projection: 768 -> 768 (identity-like transformation)
            # This maintains the same architecture pattern without changing the data
            self.feature_extractor = nn.ModuleList([
                nn.Linear(input_dim, input_dim)  # Identity-like projection
            ])
            
            # Normalization and activation layers (same as raw mode)
            self.feature_norms = nn.ModuleList([
                nn.LayerNorm(input_dim)
            ])
            
            self.feature_activations = nn.ModuleList([
                nn.LeakyReLU(0.1)
            ])
            
            self.feature_dropouts = nn.ModuleList([
                nn.Dropout(feature_dropout)
            ])
            gru_input_dim = input_dim
        
        # Only apply dropout if num_layers > 1
        gru_dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(
            gru_input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=gru_dropout,
            bidirectional=use_bidirectional
        )
        
        # Adjust hidden dimension for bidirectional
        gru_output_dim = hidden_dim * 2 if use_bidirectional else hidden_dim
        
        # Add temporal attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=gru_output_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(gru_output_dim)
        
        # Enhanced classifier with residual connections
        self.fc = nn.Sequential(
            nn.Linear(gru_output_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
            # Removed Sigmoid - we'll work with raw logits for better gradient flow
        )
        
        # Initialize weights properly to avoid zero predictions
        self._init_weights()
        
        # Initialize loss function based on the argument
        if loss_fn == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for raw logits
        elif loss_fn == "asymmetric":
            # Use a simpler asymmetric loss configuration
            self.loss_fn = lambda preds, targets: asymmetric_loss(
                preds, targets, gamma_pos=0.0, gamma_neg=2.0, margin=0.05
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
            
        self.f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.map = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.training_step_outputs = []

    def forward(self, x):
        # Apply feature extraction for both modes to maintain consistency
        if self.feature_mode == "raw":
            # x shape: (batch_size, audio_length)
            batch_size, audio_length = x.shape
            
            # NORMALIZE INPUT: This is crucial for preventing vanishing gradients
            # Normalize to zero mean and unit variance per batch
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
            
            # Use unfold to create sliding windows more efficiently
            # This maintains better gradient flow than the loop approach
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, audio_length)
            
            # Create sliding windows using unfold
            # This is equivalent to the loop but much more efficient and gradient-friendly
            x = x.unfold(dimension=2, size=self.window_size, step=self.hop_size)
            # Result: (batch_size, 1, num_windows, window_size)
            
            # Reshape for batch processing: (batch_size * num_windows, window_size)
            num_windows = x.size(2)
            x = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_windows, 1, window_size)
            x = x.view(batch_size * num_windows, self.window_size)
            
            # Extract features for all windows at once (simplified)
            x = self.feature_extractor[0](x)  # (batch_size * num_windows, 768)
            x = self.feature_norms[0](x)
            x = self.feature_activations[0](x)
            x = self.feature_dropouts[0](x)
            
            # Reshape back to sequence: (batch_size, num_windows, 768)
            x = x.view(batch_size, num_windows, -1)
            
        else:
            # Wav2Vec mode: x is already (batch_size, seq_len, 768)
            # Apply the same feature processing pattern for consistency
            batch_size, seq_len, features = x.shape
            
            # Reshape for batch processing: (batch_size * seq_len, features)
            x = x.view(batch_size * seq_len, features)
            
            # Apply the same feature processing pattern (lightweight transformation)
            x = self.feature_extractor[0](x)  # (batch_size * seq_len, features)
            x = self.feature_norms[0](x)
            x = self.feature_activations[0](x)
            x = self.feature_dropouts[0](x)
            
            # Reshape back to sequence: (batch_size, seq_len, features)
            x = x.view(batch_size, seq_len, -1)
        
        # Process sequences with BPTT if they're too long (but not for bidirectional GRU)
        if self.bptt_length > 0 and x.size(1) > self.bptt_length and not self.use_bidirectional:
            # Truncated BPTT: process in chunks
            batch_size, seq_len, features = x.shape
            outputs = []
            hidden = None
            
            for i in range(0, seq_len, self.bptt_length):
                chunk = x[:, i:i+self.bptt_length, :]
                chunk_output, hidden = self.gru(chunk, hidden)
                
                # Detach hidden state to prevent gradient flow across chunks
                # This is crucial for truncated BPTT
                if i + self.bptt_length < seq_len:
                    # For both unidirectional and bidirectional GRU, hidden is a tuple of tensors
                    # Each tensor represents the hidden state for each layer
                    hidden = tuple(h.detach() for h in hidden)
                
                outputs.append(chunk_output)
            
            # Concatenate outputs
            gru_output = torch.cat(outputs, dim=1)
        else:
            # Process entire sequence at once
            gru_output, _ = self.gru(x)
        
        # Apply attention mechanism
        if self.use_attention:
            attn_output, _ = self.attention(gru_output, gru_output, gru_output)
            gru_output = self.attention_norm(gru_output + attn_output)
        
        # Use attention-weighted average instead of just last hidden state
        if self.use_attention:
            # Global average pooling with attention weights
            attention_weights = torch.softmax(
                torch.mean(gru_output, dim=-1), dim=1
            ).unsqueeze(-1)
            pooled_output = torch.sum(gru_output * attention_weights, dim=1)
        else:
            # Use mean pooling instead of just last state
            pooled_output = torch.mean(gru_output, dim=1)
        
        output = self.fc(pooled_output)
        
        return output

    def _init_weights(self):
        """Initialize weights to prevent vanishing gradients"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for LeakyReLU activations
                if 'feature_extractor' in name:
                    # Special initialization for feature extractor layers (both modes)
                    if self.feature_mode == "raw":
                        # Raw mode: standard Kaiming initialization
                        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                    else:
                        # Wav2Vec mode: identity-like initialization for the projection layer
                        nn.init.eye_(module.weight)  # Identity matrix initialization
                        # Don't override with zeros - keep the identity matrix
                    
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.01)  # Small positive bias to avoid dead neurons
                else:
                    # Standard initialization for other linear layers
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
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
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.training_step_outputs.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        # Log average training loss for the epoch
        avg_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use lower learning rate for raw mode to prevent gradient issues
        if self.feature_mode == "raw":
            lr = self.hparams.lr * 0.1  # Reduce learning rate for raw audio
        else:
            lr = self.hparams.lr
        
        # Create optimizer with better parameters
        import torch.optim as optim
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=lr, 
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
                eta_min=lr * 0.001  # Minimum LR
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

# python run_experiment_gru_lightning.py --save_dir "gru_002" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu
# python run_experiment_gru_lightning.py --save_dir "gru_003" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu
# python run_experiment_gru_lightning.py --save_dir "gru_004" --epochs 1000 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_005" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_006" --epochs 1000 --eval_interval 100 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --pretrained_model "gru_005"
# python run_experiment_gru_lightning.py --save_dir "gru_007" --epochs 1000 --eval_interval 10 --lr 1e-4 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_008" --epochs 10000 --eval_interval 100 --lr 1e-4 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --pretrained_model "gru_007"
# python run_experiment_gru_lightning.py --save_dir "gru_009" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_010" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce"
# python run_experiment_gru_lightning.py --save_dir "gru_011" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0
# python run_experiment_gru_lightning.py --save_dir "gru_012" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_013" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce"
# python run_experiment_gru_lightning.py --save_dir "gru_014" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0
# python run_experiment_gru_lightning.py --save_dir "gru_015" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_016" --epochs 1000 --pretrained_model "gru_013" --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce"
# python run_experiment_gru_lightning.py --save_dir "gru_017" --epochs 1000 --pretrained_model "gru_014" --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0
# python run_experiment_gru_lightning.py --save_dir "gru_018" --epochs 1000 --pretrained_model "gru_015" --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm
from pytorch_lightning.loggers import CSVLogger
import json
from utils import preprocess_audio, extract_wav2vec_embeddings, SAMPLE_RATE, TARGET_LENGTH, asymmetric_loss, MeanContrastiveRankingLoss
import multiprocessing

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, clip_ids, labels, is_train=True, test_size=0.1, random_state=42):
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
        if len(self.indices) == 0:
            raise ValueError(f"Empty {'training' if is_train else 'validation'} dataset. "
                           f"This might be due to an incorrect test_size parameter.")
    
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
    def __init__(self, train_loader):
        super().__init__()
        self.train_loader = train_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only compute train metrics on validation epochs
        pl_module.eval()
        device = pl_module.device
        loss_fn = pl_module.loss_fn
        f1 = pl_module.f1.to(device)
        map_metric = pl_module.map.to(device)
        auc = pl_module.auc.to(device)

        all_preds = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                preds = pl_module(x)
                loss = loss_fn(preds, y)
                total_loss += loss.item() * x.size(0)
                all_preds.append(preds)
                all_targets.append(y)
                total_samples += x.size(0)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / total_samples

        train_f1 = f1(all_preds, all_targets.int()).item()
        train_map = map_metric(all_preds, all_targets.int()).item()
        train_auc = auc(all_preds, all_targets.int()).item()

        # Log metrics with current epoch
        current_epoch = trainer.current_epoch
        trainer.logger.log_metrics({
            "epoch": current_epoch,
            "train_loss_eval": avg_loss,
            "train_f1_eval": train_f1,
            "train_map_eval": train_map,
            "train_auc_eval": train_auc
        }, step=current_epoch)

        pl_module.train()  # Switch back to training mode

class WeightNormCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Calculate L2 norm for each parameter group
        weight_norms = {}
        grad_norms = {}
        
        # Add epoch to metrics
        weight_norms["epoch"] = trainer.current_epoch
        grad_norms["epoch"] = trainer.current_epoch
        
        for name, param in pl_module.named_parameters():
            if param.requires_grad:  # Only compute for trainable parameters
                # Weight norms
                weight_norms[f"weight_norm/{name}"] = torch.norm(param.data, p=2).item()
                
                # Gradient norms (if gradients exist)
                if param.grad is not None:
                    grad_norms[f"grad_norm/{name}"] = torch.norm(param.grad.data, p=2).item()
        
        # Log all weight and gradient norms
        trainer.logger.log_metrics(weight_norms, step=trainer.current_epoch)
        trainer.logger.log_metrics(grad_norms, step=trainer.current_epoch)

# ========================
# 1. Parse Input Arguments
# ========================
parser = argparse.ArgumentParser(description="Train an audio classification model with Wav2Vec2 embeddings and RNN (Lightning).")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--eval_interval", type=int, default=100, help="Interval for evaluating the model")
parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging metrics")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
parser.add_argument("--test_size", type=float, default=0.1, help="Test size")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the model and metrics")
parser.add_argument("--pretrained_model", type=str, default=None, help="Path to a pretrained model checkpoint")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--embedding_dir", type=str, default=".", help="Directory to load/save embeddings")
parser.add_argument("--loss_fn", type=str, default="bce", choices=["bce", "asymmetric", "contrastive"], 
                   help="Loss function to use: bce, asymmetric, or contrastive")
parser.add_argument("--loss_margin", type=float, default=0.1, help="Margin for contrastive loss")
parser.add_argument("--gamma_pos", type=float, default=0.0, help="Gamma positive for asymmetric loss")
parser.add_argument("--gamma_neg", type=float, default=4.0, help="Gamma negative for asymmetric loss")
args = parser.parse_args()

# ========================
# 2. Device
# ========================
device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
print(f"\n🔹 Using device: {device}\n")

# ========================
# 3. Load Wav2Vec 2.0 Model
# ========================
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
wav2vec_model.eval()
wav2vec_model.to(device)

TARGET_LENGTH = 10 * 16000
SAMPLE_RATE = 16000

os.makedirs(args.save_dir, exist_ok=True)
# Save args to JSON for reproducibility
with open(os.path.join(args.save_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

# ========================
# 5. Load Dataset & Extract Features
# ========================
csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
print(f"🔹 Loading CSV from: {csv_path}")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)
clip_ids = df["clip_id"].values
labels = df.iloc[:, 2:-1].values
AUDIO_DIR = "../tmp/fsd50k/FSD50K.dev_audio"
print(f"🔹 Audio directory: {AUDIO_DIR}")
if not os.path.exists(AUDIO_DIR):
    raise FileNotFoundError(f"Audio directory not found at: {AUDIO_DIR}")

print(f"🔹 Number of clips in CSV: {len(clip_ids)}")

embedding_dir = args.embedding_dir
# Create a subdirectory for embeddings at repository root level
embeddings_subdir = os.path.join(embedding_dir, "embeddings")
os.makedirs(embeddings_subdir, exist_ok=True)

print(f"🔹 Checking for precomputed embeddings in: {embeddings_subdir}")
if os.path.exists(os.path.join(embedding_dir, "metadata.json")):
    print("🔹 Loading precomputed embeddings metadata...")
    with open(os.path.join(embedding_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    print(f"🔹 Found {metadata['total_samples']} precomputed embeddings")
else:
    print("🔹 No precomputed embeddings found. Starting extraction...")
    processed_count = 0
    error_count = 0
    missing_files = []
    
    # Create embedding directory
    os.makedirs(embeddings_subdir, exist_ok=True)
    
    for clip_id, label in tqdm(zip(clip_ids, labels), total=len(clip_ids)):
        audio_path = os.path.join(AUDIO_DIR, f"{clip_id}.wav")
        if os.path.exists(audio_path):
            try:
                emb = extract_wav2vec_embeddings(audio_path, processor, wav2vec_model, device)
                # Save individual embedding in the subdirectory
                embedding_path = os.path.join(embeddings_subdir, f"{clip_id}.npy")
                np.save(embedding_path, emb)
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"🔹 Processed {processed_count} files")
            except Exception as e:
                print(f"Warning: Error processing {clip_id}: {str(e)}")
                error_count += 1
        else:
            missing_files.append(clip_id)
            error_count += 1
    
    print(f"🔹 Processed {processed_count} files successfully")
    print(f"🔹 Encountered {error_count} errors")
    print(f"🔹 Missing files: {len(missing_files)}")
    
    if processed_count == 0:
        raise ValueError("No audio files were successfully processed. Please check the audio directory path and file permissions.")
    
    # Save metadata
    metadata = {
        "total_samples": processed_count,
        "embedding_shape": emb.shape,
        "label_shape": labels.shape[1:]
    }
    with open(os.path.join(embedding_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("🔹 Saved embeddings and metadata for future runs.")

# Create datasets
try:
    train_dataset = EmbeddingDataset(embeddings_subdir, clip_ids, labels, is_train=True, test_size=args.test_size)
    val_dataset = EmbeddingDataset(embeddings_subdir, clip_ids, labels, is_train=False, test_size=args.test_size)
except Exception as e:
    print(f"❌ Error creating datasets: {str(e)}")
    print("\nPossible solutions:")
    print("1. Check if the embedding files exist in the correct directory")
    print("2. Make sure the embedding files are named correctly (clip_id.npy)")
    print("3. Verify that the test_size parameter is appropriate")
    raise

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
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
# 7. Lightning Model
# ========================
class LitRNNClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, lr, weight_decay, dropout, 
                 loss_fn="bce", loss_margin=0.1, gamma_pos=0.0, gamma_neg=4.0):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
        # Initialize loss function based on the argument
        if loss_fn == "bce":
            self.loss_fn = nn.BCELoss()
        elif loss_fn == "asymmetric":
            self.loss_fn = lambda preds, targets: asymmetric_loss(
                preds, targets, gamma_pos=gamma_pos, gamma_neg=gamma_neg
            )
        elif loss_fn == "contrastive":
            self.loss_fn = MeanContrastiveRankingLoss(margin=loss_margin)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
            
        self.f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.map = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.training_step_outputs = []

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n[-1]
        return self.fc(h_n)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.training_step_outputs.append(loss.item())
        
        # Log based on the log_interval parameter
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
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
        
        # Only compute metrics on the first batch to save memory
        if batch_idx == 0:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_f1', self.f1(preds, y.int()), on_step=False, on_epoch=True)
            self.log('val_map', self.map(preds, y.int()), on_step=False, on_epoch=True)
            self.log('val_auc', self.auc(preds, y.int()), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

# ========================
# 8. Training
# ========================
model = LitRNNClassifier(
    input_dim=train_dataset[0][0].shape[1],  # Use the feature dimension (768) from the embeddings
    hidden_dim=256,
    num_layers=1,
    num_classes=train_dataset[0][1].shape[0],
    lr=args.lr,
    weight_decay=args.weight_decay,
    dropout=args.dropout,
    loss_fn=args.loss_fn,
    loss_margin=args.loss_margin,
    gamma_pos=args.gamma_pos,
    gamma_neg=args.gamma_neg
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args.save_dir,
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=True,
    mode='min'
)

# Configure CSV logger with reduced logging frequency
csv_logger = CSVLogger(
    save_dir=args.save_dir,
    name="metrics",
    version=None,  # Don't create new version directories
    flush_logs_every_n_steps=args.log_interval  # Use log_interval parameter
)
train_eval_callback = TrainEvalMetricsCallback(train_loader)
weight_norm_callback = WeightNormCallback()  # Add weight norm callback
trainer = pl.Trainer(
    max_epochs=args.epochs,
    callbacks=[checkpoint_callback, early_stop_callback, train_eval_callback, weight_norm_callback],  # Add weight_norm_callback
    accelerator='gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu',
    default_root_dir=args.save_dir,
    logger=csv_logger,
    check_val_every_n_epoch=args.eval_interval,
    log_every_n_steps=args.log_interval  # Use log_interval parameter
)

if __name__ == '__main__':
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load pretrained model if specified
    if args.pretrained_model is not None:
        print(f"🔹 Loading pretrained model from {args.pretrained_model}")
        # Find the best checkpoint in the pretrained model directory
        checkpoint_dir = os.path.join(args.pretrained_model, "best-checkpoint.ckpt")
        if os.path.exists(checkpoint_dir):
            model = LitRNNClassifier.load_from_checkpoint(checkpoint_dir)
            print("✅ Successfully loaded pretrained model")
        else:
            print(f"⚠️ Warning: No checkpoint found at {checkpoint_dir}")

    trainer.fit(model, train_loader, val_loader)
    print("✅ Training complete!") 
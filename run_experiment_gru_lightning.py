# python run_experiment_gru_lightning.py --save_dir "gru_002" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu
# python run_experiment_gru_lightning.py --save_dir "gru_003" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu
# python run_experiment_gru_lightning.py --save_dir "gru_004" --epochs 1000 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_005" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm
from pytorch_lightning.loggers import CSVLogger
import json

class TrainEvalMetricsCallback(Callback):
    def __init__(self, train_loader):
        super().__init__()
        self.train_loader = train_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        device = pl_module.device
        loss_fn = pl_module.loss_fn
        f1 = pl_module.f1
        map_metric = pl_module.map
        auc = pl_module.auc

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
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                total_samples += x.size(0)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / total_samples

        train_f1 = f1(all_preds, all_targets.int()).item()
        train_map = map_metric(all_preds, all_targets.int()).item()
        train_auc = auc(all_preds, all_targets.int()).item()

        # Log metrics
        trainer.logger.log_metrics({
            "train_loss_eval": avg_loss,
            "train_f1_eval": train_f1,
            "train_map_eval": train_map,
            "train_auc_eval": train_auc
        }, step=trainer.global_step)

        pl_module.train()  # Switch back to training mode

# ========================
# 1. Parse Input Arguments
# ========================
parser = argparse.ArgumentParser(description="Train an audio classification model with Wav2Vec2 embeddings and RNN (Lightning).")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--eval_interval", type=int, default=100, help="Interval for evaluating the model")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
parser.add_argument("--test_size", type=float, default=0.1, help="Test size")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the model and metrics")
parser.add_argument("--pretrained_model", type=str, default=None, help="Path to a pretrained model checkpoint")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--embedding_dir", type=str, default=".", help="Directory to load/save embeddings")
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
# 4. Audio Preprocessing
# ========================
def preprocess_audio(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(waveform).float()
    if waveform.shape[-1] > TARGET_LENGTH:
        waveform = waveform[:TARGET_LENGTH]
    elif waveform.shape[-1] < TARGET_LENGTH:
        padding = torch.zeros(TARGET_LENGTH - waveform.shape[-1])
        waveform = torch.cat((waveform, padding))
    return waveform

def extract_wav2vec_embeddings(audio_path):
    waveform = preprocess_audio(audio_path)
    input_values = processor(waveform.numpy(), return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values.to(device)
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
        embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (time_steps, 768)
    
    # We should not use the mean of the embeddings, but the whole sequence.
    # chunks = [embeddings[i * (embeddings.shape[0] // 10):(i + 1) * (embeddings.shape[0] // 10)].mean(dim=0) for i in range(10)]
    # return torch.stack(chunks).cpu().numpy()  # Shape: (10, 768)
    return embeddings.cpu().numpy()  # Shape: (time_steps, 768)

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
embedding_path = os.path.join(embedding_dir, "embeddings.npy")
label_path = os.path.join(embedding_dir, "labels.npy")

print(f"🔹 Checking for precomputed embeddings in: {embedding_dir}")
if os.path.exists(embedding_path) and os.path.exists(label_path):
    print("🔹 Loading precomputed embeddings...")
    embeddings = np.load(embedding_path)
    labels = np.load(label_path)
    print(f"🔹 Loaded embeddings shape: {embeddings.shape}")
    print(f"🔹 Loaded labels shape: {labels.shape}")
else:
    print("🔹 No precomputed embeddings found. Starting extraction...")
    embeddings = []
    valid_labels = []
    processed_count = 0
    error_count = 0
    missing_files = []
    for clip_id, label in tqdm(zip(clip_ids, labels), total=len(clip_ids)):
        audio_path = os.path.join(AUDIO_DIR, f"{clip_id}.wav")
        if os.path.exists(audio_path):
            try:
                emb = extract_wav2vec_embeddings(audio_path)
                embeddings.append(emb)
                valid_labels.append(label)
                processed_count += 1
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
    embeddings = np.array(embeddings)
    labels = np.array(valid_labels)
    print(f"🔹 Extracted embeddings shape: {embeddings.shape}")
    print(f"🔹 Extracted labels shape: {labels.shape}")
    os.makedirs(embedding_dir, exist_ok=True)
    np.save(embedding_path, embeddings)
    np.save(label_path, labels)
    print("🔹 Saved embeddings and labels for future runs.")

# ========================
# 6. Split Dataset
# ========================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=args.test_size, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# ========================
# 7. Lightning Model
# ========================
class LitRNNClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, lr, weight_decay, dropout):
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
        self.loss_fn = nn.BCELoss()
        self.f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.map = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.auc = MultilabelAUROC(num_labels=num_classes, average="macro")

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n[-1]
        return self.fc(h_n)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
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
    input_dim=X_train.shape[2],
    hidden_dim=256,
    num_layers=1,
    num_classes=y_train.shape[1],
    lr=args.lr,
    weight_decay=args.weight_decay,
    dropout=args.dropout
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
    patience=10,
    verbose=True,
    mode='min'
)

csv_logger = CSVLogger(save_dir=args.save_dir, name="metrics")
train_eval_callback = TrainEvalMetricsCallback(train_loader)
trainer = pl.Trainer(
    max_epochs=args.epochs,
    callbacks=[checkpoint_callback, early_stop_callback, train_eval_callback],
    accelerator='gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu',
    default_root_dir=args.save_dir,
    logger=csv_logger,
    check_val_every_n_epoch=args.eval_interval
)

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
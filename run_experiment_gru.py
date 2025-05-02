import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import pandas as pd
import os
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ========================
# 1. Parse Input Arguments
# ========================
parser = argparse.ArgumentParser(description="Train an audio classification model with Wav2Vec2 embeddings and RNN.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--eval_interval", type=int, default=100, help="Interval for evaluating the model")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the model and metrics")
parser.add_argument("--pretrained_model", type=str, default=None, help="Path to a pretrained model checkpoint")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--embedding_dir", type=str, default=".", help="Directory to load/save embeddings")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
print(f"\n🔹 Using device: {device}\n")

# ========================
# 2. Load Wav2Vec 2.0 Model
# ========================
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()
model.to(device)

TARGET_LENGTH = 10 * 16000
SAMPLE_RATE = 16000

os.makedirs(args.save_dir, exist_ok=True)

# ========================
# 3. Audio Preprocessing
# ========================
def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    waveform = waveform.squeeze()
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
        outputs = model(input_values)
        embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (time_steps, 768)
    chunks = [embeddings[i * (embeddings.shape[0] // 10):(i + 1) * (embeddings.shape[0] // 10)].mean(dim=0) for i in range(10)]
    return torch.stack(chunks).cpu().numpy()  # Shape: (10, 768)

# ========================
# 4. Load Dataset & Extract Features
# ========================
csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
df = pd.read_csv(csv_path)
clip_ids = df["clip_id"].values
labels = df.iloc[:, 2:-1].values
AUDIO_DIR = "../tmp/fsd50k/FSD50K.dev_audio"
print(f"🔹 Audio directory: {AUDIO_DIR}")
print(f"🔹 Number of clips in CSV: {len(clip_ids)}")

# ========================
# 5. Define RNN Classifier
# ========================
class RNNClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=1, num_classes=200):
        super(RNNClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n[-1]  # Last hidden state
        return self.fc(h_n)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# ============================
# 6. Evaluation Helper Function
# ============================
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    valid_classes = np.where(y_test_np.sum(axis=0) > 0)[0]
    if len(valid_classes) == 0:
        return {"mAP": None, "F1": None, "ROC-AUC": None}
    return {
        "mAP": average_precision_score(y_test_np[:, valid_classes], preds[:, valid_classes], average="macro"),
        "F1": f1_score(y_test_np, preds > 0.5, average="macro", zero_division=1),
        "ROC-AUC": roc_auc_score(y_test_np[:, valid_classes], preds[:, valid_classes], average="macro")
    }

# ==========================
# 7. Training Function
# ==========================
def train_classifier(X_train, y_train, X_test, y_test, epochs, lr, batch_size, eval_interval, save_dir, pretrained_model=None):
    model = RNNClassifier(input_dim=X_train.shape[2], num_classes=y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    if pretrained_model and os.path.exists(pretrained_model):
        print(f"🔹 Loading pretrained model from {pretrained_model}")
        checkpoint = torch.load(pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    loss_fn = nn.BCELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    eval_results = []
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(loader)
        if (epoch + 1) % eval_interval == 0:
            with torch.no_grad():
                preds_test = model(X_test.to(device))
                test_loss = loss_fn(preds_test, y_test.to(device)).item()
            results = evaluate_model(model, X_test, y_test)
            eval_results.append({
                "Epoch": epoch + 1,
                "Training Loss": avg_train_loss,
                "Test Loss": test_loss,
                "mAP": results["mAP"],
                "F1": results["F1"],
                "ROC-AUC": results["ROC-AUC"]
            })
            print(f"🔹 Evaluation Results (Epoch {epoch+1}): {eval_results[-1]}")
            df_metrics = pd.DataFrame(eval_results)
            df_metrics.to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_dir, "last_model.pth"))
    return model

# ==========================
# 8. Feature Extraction and Training
# ==========================
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
            print(f"Warning: Audio file not found: {audio_path}")
            error_count += 1

    print(f"🔹 Processed {processed_count} files successfully")
    print(f"🔹 Encountered {error_count} errors")
    
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

print("🔹 Splitting dataset into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.5, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("🔹 Starting training...")
supervised_model = train_classifier(
    X_train, y_train, X_test, y_test,
    epochs=args.epochs,
    lr=args.lr,
    batch_size=args.batch_size,
    eval_interval=args.eval_interval,
    save_dir=args.save_dir,
    pretrained_model=args.pretrained_model
)

print("✅ Training complete!")
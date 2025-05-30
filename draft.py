from transformers import Wav2Vec2Processor, Wav2Vec2Model
from utils import preprocess_audio, SAMPLE_RATE
import torch
import os
import pandas as pd

csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
print(f"🔹 Loading CSV from: {csv_path}")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)
clip_ids = df["clip_id"].values

AUDIO_DIR = "../tmp/fsd50k/FSD50K.dev_audio"
clip_id = clip_ids[0]
audio_path = os.path.join(AUDIO_DIR, f"{clip_id}.wav")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔹 Using device: {device}\n")

MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
wav2vec_model.eval()
wav2vec_model.to(device)

waveform = preprocess_audio(audio_path)
input_values = processor(waveform.numpy(), return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values.to(device)
with torch.no_grad():
    outputs = wav2vec_model(input_values)
    embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (time_steps, 768)


print(f"Embeddings shape: {embeddings.shape}")
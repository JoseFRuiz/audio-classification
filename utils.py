import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

TARGET_LENGTH = 10 * 16000
SAMPLE_RATE = 16000

def preprocess_audio(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(waveform).float()
    if waveform.shape[-1] > TARGET_LENGTH:
        waveform = waveform[:TARGET_LENGTH]
    elif waveform.shape[-1] < TARGET_LENGTH:
        padding = torch.zeros(TARGET_LENGTH - waveform.shape[-1])
        waveform = torch.cat((waveform, padding))
    return waveform

def extract_wav2vec_embeddings(audio_path, processor, wav2vec_model, device):
    waveform = preprocess_audio(audio_path)
    input_values = processor(waveform.numpy(), return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values.to(device)
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
        embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (time_steps, 768)
    return embeddings.cpu().numpy()  # Shape: (time_steps, 768) 
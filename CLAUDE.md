# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Two environments are in use — pick based on context:

**Local dev (uv):**
```bash
uv sync                          # install deps
uv run python <script>.py        # run any script
```

**HiPerGator cluster (conda):**
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio-classification
python <script>.py
```

Submit cluster jobs with:
```bash
sbatch gpu_job.sh    # edit the script first to set the right experiment command
sbatch cpu_job.sh
```
Each `gpu_job*.sh` targets one experiment. Edit the `python run_experiment_*.py ...` call inside before submitting.

## Dataset

- **Audio**: `../tmp/fsd50k/FSD50K.dev_audio/` — WAV/MP3/FLAC/OGG files named `{clip_id}.{ext}`
- **Labels CSV** (two variants):
  - `../tmp/fsd50k_spc/fsd50k_clips_labels_duration.csv` — full dataset
  - `../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv` — clips ≤10s only
- **CSV layout**: column 0 = `clip_id`, columns 1 = ignored, columns 2:-1 = multi-hot binary labels, last column = duration. Labels are extracted as `df.iloc[:, 2:-1].values`.
- **Audio target length**: 10s at 16 kHz (160,000 samples); shorter clips are zero-padded, longer are truncated.

## Experiment Scripts

All experiment scripts are self-contained (data loading, model, training loop in one file) and follow PyTorch Lightning. Each accepts `--save_dir` and writes `args.json`, `best-checkpoint.ckpt`, and `supervised_metrics/version_0/metrics.csv` into that directory.

| Script | Backbone | Pretraining | Notes |
|--------|----------|-------------|-------|
| `run_experiment_gru_lightning.py` | CRNN (Conv1D + GRU) | None | Most configurable; supports BCE/ASL/WuAUC losses, wav2vec features, raw audio |
| `run_experiment_dino_lightning.py` | CRNN | DINO / DINOv2 | Semi-supervised; two-view MEL augmentation; iBOT patch loss variant |
| `run_experiment_asit_lightning.py` | AudioViT (ViT-tiny/small/base) | ASiT (GMML) | Self-supervised with patch corruption; CLS+PATCH contrastive loss |
| `run_birdclef_experiment.py` | CRNN | None | BirdCLEF dataset variant |

**Common flags shared across all scripts:**
```
--save_dir        output directory
--use_gpu         enable CUDA
--dataset_type    complete | max10sec
--labeled_ratio   fraction of training used as labeled data (default 0.1)
--batch_size, --num_workers, --lr, --weight_decay
--eval_interval   validate every N epochs
--use_early_stopping --early_stopping_patience
```

## Architecture Patterns

### Data pipeline (all scripts)
1. Load CSV → filter to clips with audio files on disk → 90/10 train/val split (seed=42) → labeled/unlabeled split by `--labeled_ratio`
2. MEL spectrogram: 128 bins, FFT=2048, hop=512 → log-dB scale → `(n_mels, time_frames)` tensor
3. Expected `time_frames` = `(160000 // hop_length) + 1` ≈ 313

### CRNN backbone (`run_experiment_dino_lightning.py`, `run_experiment_gru_lightning.py`)
- `Conv1d` stack along time → `GRU` → mean pooling → `(batch, hidden_dim)` feature vector
- Output dim = `hidden_dim` (× 2 if bidirectional)

### AudioViT backbone (`run_experiment_asit_lightning.py`)
- `PatchEmbed` (Conv2d, no padding) → CLS token + patch tokens → N Transformer blocks → LayerNorm
- Patch count: `(n_mels // patch_size_freq) × (time_frames // patch_size_time)` — pixels that don't fit a full patch are dropped
- Output: CLS token `(batch, embed_dim)` for classification; all patch tokens for ASIT pretraining loss
- Config: tiny=192d/3h, small=384d/6h, base=768d/12h (all 12 layers)

### Two-stage training (DINO / ASiT scripts)
1. **Pretraining** (`--dino_pretrain_epochs` / `--asit_pretrain_epochs`): self-supervised on all training data; saves `dino_pretrained.ckpt` / `asit_pretrained.ckpt`
2. **Fine-tuning** (`--finetune_epochs`): supervised on labeled subset; saves `best-checkpoint.ckpt`
- `--supervised_only` skips stage 1; `--skip_asit_pretrain` / `--skip_dino_pretrain` jumps directly to fine-tuning using an existing checkpoint

### Metrics (all supervised models)
Macro-averaged over all classes, computed per epoch: F1 (`train_f1_eval`, `val_f1`), mAP (`train_map_eval`, `val_map`), AUROC (`train_auc_eval`, `val_auc`). Logged to CSV via `CSVLogger` and printed each epoch.

## Shared Utilities

- **`utils.py`**: loss functions only — `wu_auc_loss`, `asymmetric_loss`, `combined_*_loss`, `MeanContrastiveRankingLoss`, `preprocess_audio`, `extract_wav2vec_embeddings`
- **`model_classes.py`**: extracts `EmbeddingDataset` and supervised model classes from `run_experiment_gru_lightning.py` to avoid circular imports in downstream scripts (e.g., `generate_predictions.py`)

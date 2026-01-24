# Generate Predictions Script

This script loads a trained model checkpoint and generates predictions for the validation dataset, saving the results to CSV files.

## Files Generated

The script generates two CSV files in the experiment directory:

1. **`validation_predictions.csv`**: Contains predicted scores (logits) for each class
2. **`validation_targets.csv`**: Contains actual binary labels for each class

Both files include a `clip_id` column to identify each sample.

## Usage

### Basic Usage

```bash
python generate_predictions.py <experiment_dir>
```

Example:
```bash
python generate_predictions.py wav2vec_032
```

**Note**: `experiment_dir` is a positional argument, not a flag. Do not use `--experiment_dir`.

### Advanced Usage

```bash
python generate_predictions.py <experiment_dir> [OPTIONS]
```

#### Options

- `--csv_path PATH`: Path to CSV file with labels (default: `../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv`)
- `--embedding_dir PATH`: Directory containing embeddings for wav2vec mode (default: `embeddings`)
- `--audio_dir PATH`: Directory containing audio files for raw mode (default: `../tmp/fsd50k/FSD50K.dev_audio`)
- `--batch_size SIZE`: Batch size for prediction (default: 32)
- `--num_workers N`: Number of workers for data loading (default: 1)
- `--class_names NAMES`: List of class names (if not provided, uses generic names like `class_001`)

#### Examples

1. **Wav2Vec experiment with custom batch size:**
   ```bash
   python generate_predictions.py wav2vec_032 --batch_size 64
   ```

2. **Raw audio experiment:**
   ```bash
   python generate_predictions.py raw_001 --audio_dir /path/to/audio/files
   ```

3. **With custom class names:**
   ```bash
   python generate_predictions.py wav2vec_032 --class_names music speech noise
   ```

4. **With custom CSV path:**
   ```bash
   python generate_predictions.py wav2vec_032 --csv_path /path/to/labels.csv
   ```

**Common mistake**: Don't use `--experiment_dir` as a flag. The experiment directory name goes directly after the script name:
- ✅ Correct: `python generate_predictions.py wav2vec_032`
- ❌ Wrong: `python generate_predictions.py --experiment_dir wav2vec_032`

## Requirements

The script requires:

1. **Experiment directory** containing:
   - `args.json`: Configuration file from the original training
   - `best-checkpoint.ckpt`: Trained model checkpoint

2. **Data files** (depending on feature mode):
   - For `wav2vec` mode: Embedding files (`.npy`) in the embedding directory
   - For `raw` mode: Audio files (`.wav`, `.mp3`, `.flac`, `.ogg`) in the audio directory

3. **CSV file** with labels containing:
   - `clip_id` column
   - Label columns (binary 0/1 values)

## How It Works

1. **Load Configuration**: Reads `args.json` from the experiment directory to get the original training parameters
2. **Create Validation Dataset**: Recreates the validation dataset using the same train/test split as the original experiment
3. **Load Model**: Loads the trained model from `best-checkpoint.ckpt`
4. **Generate Predictions**: Runs inference on the validation dataset
5. **Save Results**: Saves predictions and targets to CSV files

## Output Format

### validation_predictions.csv
```csv
clip_id,class_001,class_002,class_003,...
clip_001,0.1234,-0.5678,0.9012,...
clip_002,-0.2345,0.6789,-0.1234,...
...
```

### validation_targets.csv
```csv
clip_id,class_001,class_002,class_003,...
clip_001,1,0,1,...
clip_002,0,1,0,...
...
```

## Notes

- The script uses the same random seed (42) as the original training to ensure the validation split is identical
- Predictions are raw logits (before sigmoid activation)
- Targets are binary values (0 or 1)
- The script automatically detects whether to use wav2vec or raw audio mode based on the experiment configuration
- GPU is used automatically if available

## Troubleshooting

1. **Import errors**: Make sure `model_classes.py` is in the same directory as `generate_predictions.py`
2. **Missing files**: Check that the experiment directory contains `args.json` and `best-checkpoint.ckpt`
3. **Data not found**: Verify that embedding files or audio files exist in the specified directories
4. **Memory issues**: Reduce `--batch_size` if you encounter out-of-memory errors

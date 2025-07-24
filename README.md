# Audio Classification

Audio classification project using PyTorch Lightning and GRU networks with B200 GPU support.

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Quick Setup

1. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

## Running Experiments

### Local Development

To train the model:
```bash
uv run python run_experiment_gru_lightning.py --save_dir "results" --epochs 1000 --eval_interval 10 --lr 1e-3 --batch_size 32 --use_gpu
```

### Cluster Environment

For GPU cluster environments, use the provided batch scripts:
```bash
sbatch gpu_job.sh    # GPU training
sbatch cpu_job.sh    # CPU training
```

## Project Structure

- `run_experiment_gru_lightning.py` - Main training script
- `utils.py` - Utility functions for audio processing
- `test_gpu.py` - GPU compatibility testing
- `pyproject.toml` - Project dependencies and configuration
- `gru_*/` - Experiment results and checkpoints
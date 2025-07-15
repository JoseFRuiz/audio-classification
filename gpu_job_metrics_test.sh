#!/bin/bash
#SBATCH --job-name=audio-classification-metrics
#SBATCH --output=metrics_test.out
#SBATCH --error=metrics_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem=180gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --qos=azare
#SBATCH --account=azare
#SBATCH --time=24:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Set up error handling
set -e  # Exit on any error

# Check if uv is available for other dependencies
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not available. Please install uv first."
    exit 1
fi

echo "🔹 Current Python version:"
uv python -c "import sys; print(f'Python {sys.version}')"

echo "🔹 Checking PyTorch installation (WITHOUT module load)..."
uv run check_pytorch_versions.py

echo "🔹 Testing GPU compatibility (WITHOUT module load)..."
uv run check_gpu_compatibility.py

# Install other dependencies with uv (excluding PyTorch)
echo "🔹 Installing other dependencies with uv..."
uv pip install pytorch-lightning torchmetrics transformers librosa tqdm pandas numpy scikit-learn accelerate

echo "🚀 Starting training with improved metrics computation..."
uv run run_experiment_gru_lightning.py --save_dir "gru_025" --epochs 50 --eval_interval 5 --log_interval 5 --lr 1e-3 --batch_size 32 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
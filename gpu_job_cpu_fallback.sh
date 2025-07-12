#!/bin/bash
#SBATCH --job-name=audio-classification-cpu
#SBATCH --output=cpu_training.out
#SBATCH --error=cpu_training.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem=64gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=hpg-b200
#SBATCH --qos=azare
#SBATCH --account=azare
#SBATCH --time=24:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Set up error handling
set -e  # Exit on any error

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not available. Please install uv first."
    exit 1
fi

echo "🔹 Current Python version:"
uv run python --version

echo "🔹 Checking PyTorch installation..."
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name()}')
    print('⚠️ Note: B200 GPU detected but not compatible with current PyTorch')
    print('🔹 Falling back to CPU training')
else:
    print('✅ Using CPU for training')
"

echo "🚀 Starting CPU training (B200 GPU not compatible with current PyTorch)..."
uv run python run_experiment_gru_lightning.py --save_dir "gru_025" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 32 --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4

echo "✅ Training completed successfully!"
echo "🔹 Results saved in gru_025/" 
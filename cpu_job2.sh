#!/bin/bash
#SBATCH --job-name=audio-classification-cpu2
#SBATCH --output=cpu_training2.out
#SBATCH --error=cpu_training2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem=180gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=hpg-default
#SBATCH --qos=azare
#SBATCH --account=azare
#SBATCH --time=96:00:00

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
print('✅ PyTorch installation verified!')
"

echo "🚀 Starting CPU training..."
uv run python run_experiment_gru_lightning.py --save_dir "gru_024" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 50 --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 1

echo "✅ Training completed successfully!"
echo "🔹 Results saved in gru_024/" 
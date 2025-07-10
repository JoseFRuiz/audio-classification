#!/bin/bash
#SBATCH --job-name=audio-classification-cpu
#SBATCH --output=cpu_training.out
#SBATCH --error=cpu_training.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem=180gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=hpg-cpu
#SBATCH --qos=azare
#SBATCH --account=azare
#SBATCH --time=96:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Set up error handling
set -e  # Exit on any error

# Activate existing conda environment
echo "🔹 Activating existing audio-classification conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio-classification

# Check current Python version and PyTorch installation
echo "🔹 Current Python version:"
python --version

echo "🔹 Checking PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✅ PyTorch installation verified!')
"

echo "🚀 Starting CPU training..."
python run_experiment_gru_lightning.py --save_dir "gru_023" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 50 --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1

echo "✅ Training completed successfully!"
echo "🔹 Results saved in gru_023/" 
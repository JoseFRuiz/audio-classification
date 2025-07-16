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

# Load the cluster's PyTorch module (which is compatible with B200)
echo "🔹 Loading cluster PyTorch module..."
module load pytorch

# Check if uv is available for other dependencies
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not available. Please install uv first."
    exit 1
fi

echo "🔹 Current Python version:"
python --version

echo "🔹 Checking PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name()}')
    print(f'GPU capability: {torch.cuda.get_device_capability()}')
    # Test GPU functionality
    try:
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.mm(x, y)
        print('✅ GPU test successful!')
    except Exception as e:
        print(f'❌ GPU test failed: {e}')
else:
    print('❌ CUDA not available')
"

# Install other dependencies with uv (excluding PyTorch)
echo "🔹 Installing other dependencies with uv..."
uv pip install pytorch-lightning torchmetrics transformers librosa tqdm pandas numpy scikit-learn accelerate

echo "🚀 Starting training with improved metrics computation..."
python run_experiment_gru_lightning.py --save_dir "gru_025" --epochs 50 --eval_interval 5 --log_interval 5 --lr 1e-3 --batch_size 32 --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
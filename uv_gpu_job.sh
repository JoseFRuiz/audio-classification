#!/bin/bash
#SBATCH --job-name=audio-classification-uv
#SBATCH --output=uv_gpu_training.out
#SBATCH --error=uv_gpu_training.err
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
#SBATCH --qos=azare
#SBATCH --account=azare
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Set up error handling
set -e  # Exit on any error

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please run setup_uv.sh first."
    exit 1
else
    echo "✅ uv is installed"
fi

# Check if uv project exists
if [ ! -d "audio-classification-uv" ]; then
    echo "❌ uv project 'audio-classification-uv' does not exist."
    echo "Please run setup_uv.sh first to create the environment."
    exit 1
else
    echo "✅ uv project exists"
fi

# Test GPU compatibility
echo "🔹 Testing GPU compatibility with uv..."
cd audio-classification-uv

# Check if PyTorch is installed and CUDA is available
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('❌ CUDA is not available. Cannot run GPU training.')
    exit(1)

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
    print('Cannot run GPU training.')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU test failed. Cannot proceed with training."
    exit 1
fi

# Run GPU training
echo "🚀 Starting GPU training with uv..."
uv run python run_experiment_gru_lightning.py --save_dir "gru_024" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 50 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 2.0 --num_workers 1

cd ..

echo "✅ Training completed successfully!"
echo "🔹 Results saved in gru_024/" 
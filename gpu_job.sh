#!/bin/bash
#SBATCH --job-name=audio-classification
#SBATCH --output=multi_gpu.out
#SBATCH --error=mult_gpu.err
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

module load cuda/12.9.1

# Create and activate virtual environment
echo "Creating virtual environment..."
if [ -d "pytorch_env" ]; then
    echo "Virtual environment already exists. Removing it..."
    rm -rf pytorch_env
fi

python -m venv pytorch_env
source pytorch_env/bin/activate

# Upgrade pip in the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install compatible PyTorch version for B200 GPU
echo "Installing compatible PyTorch version for B200 GPU..."
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other required packages
echo "Installing other required packages..."
pip install pytorch-lightning==1.9.5 torchmetrics==0.11.4 transformers==4.21.3 librosa==0.9.2 tqdm==4.64.1 pandas==1.3.5 numpy==1.21.6 scikit-learn==1.0.2

# Verify PyTorch installation and GPU compatibility
echo "Testing GPU compatibility..."
python test_gpu.py

if [ $? -ne 0 ]; then
    echo "❌ GPU test failed. Exiting..."
    exit 1
fi

echo "🚀 Starting training..."
python run_experiment_gru_lightning.py --save_dir "gru_023" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 50 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1

echo "✅ Training completed successfully!"
echo "🔹 Results saved in gru_023/"

# Clean up virtual environment (optional - comment out if you want to keep it)
# echo "Cleaning up virtual environment..."
# deactivate
# rm -rf pytorch_env

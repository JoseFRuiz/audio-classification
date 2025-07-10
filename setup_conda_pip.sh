#!/bin/bash
# Conda + pip setup script for audio classification

echo "🔹 Setting up conda environment with pip for audio classification..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No NVIDIA GPU detected. Will install CPU-only PyTorch."
fi

# Activate existing conda environment
echo "🔹 Activating existing audio-classification conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio-classification

# Check current Python version
echo "🔹 Current Python version:"
python --version

# Upgrade pip
echo "🔹 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with pip (often more reliable than conda)
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 Installing PyTorch with CUDA support via pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "🔹 Installing PyTorch CPU-only version via pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other packages with pip
echo "🔹 Installing other packages via pip..."
pip install pytorch-lightning torchmetrics transformers librosa tqdm pandas numpy scikit-learn

# Test installation
echo "🔹 Testing installation..."
python test_gpu.py

echo "✅ Setup complete!"
echo "🔹 To activate the environment: conda activate audio-classification"
echo "🔹 To deactivate: conda deactivate" 
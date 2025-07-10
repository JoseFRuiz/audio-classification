#!/bin/bash
# Simple conda setup script for audio classification

echo "🔹 Setting up conda environment for audio classification..."

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

# Install PyTorch step by step to avoid memory issues
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 Installing PyTorch with CUDA support..."
    echo "Step 1: Installing PyTorch core..."
    conda install pytorch -c pytorch -y
    
    echo "Step 2: Installing torchvision..."
    conda install torchvision -c pytorch -y
    
    echo "Step 3: Installing torchaudio..."
    conda install torchaudio -c pytorch -y
    
    echo "Step 4: Installing CUDA support..."
    conda install pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "🔹 Installing PyTorch CPU-only version..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install other packages one by one
echo "🔹 Installing other packages..."
echo "Installing pytorch-lightning..."
conda install pytorch-lightning -c conda-forge -y

echo "Installing torchmetrics..."
conda install torchmetrics -c conda-forge -y

echo "Installing transformers..."
conda install transformers -c conda-forge -y

echo "Installing librosa..."
conda install librosa -c conda-forge -y

echo "Installing other utilities..."
conda install tqdm pandas numpy scikit-learn -c conda-forge -y

# Test installation
echo "🔹 Testing installation..."
python test_gpu.py

echo "✅ Setup complete!"
echo "🔹 To activate the environment: conda activate audio-classification"
echo "🔹 To deactivate: conda deactivate" 
#!/bin/bash
# Setup script for local development with GPU support

echo "🔹 Setting up virtual environment for audio classification..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No NVIDIA GPU detected. Will install CPU-only PyTorch."
fi

# Create virtual environment
if [ -d "pytorch_env" ]; then
    echo "🔹 Virtual environment already exists. Removing it..."
    rm -rf pytorch_env
fi

echo "🔹 Creating virtual environment..."
python -m venv pytorch_env
source pytorch_env/bin/activate

# Upgrade pip
echo "🔹 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support if available
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "🔹 Installing PyTorch CPU-only version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "🔹 Installing other requirements..."
pip install -r requirements.txt

# Test installation
echo "🔹 Testing installation..."
python test_gpu.py

echo "✅ Setup complete!"
echo "🔹 To activate the environment: source pytorch_env/bin/activate"
echo "🔹 To deactivate: deactivate" 
#!/bin/bash
# Setup script for local development with GPU support

echo "🔹 Setting up virtual environment for audio classification..."

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🔹 Python version: $PYTHON_VERSION"

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
    # Use older PyTorch version compatible with Python 3.6
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
else
    echo "🔹 Installing PyTorch CPU-only version..."
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements with compatible versions
echo "🔹 Installing other requirements..."
pip install pytorch-lightning==1.9.5 torchmetrics==0.11.4 transformers==4.21.3 librosa==0.9.2 tqdm==4.64.1 pandas==1.3.5 numpy==1.21.6 scikit-learn==1.0.2

# Test installation
echo "🔹 Testing installation..."
python test_gpu.py

echo "✅ Setup complete!"
echo "🔹 To activate the environment: source pytorch_env/bin/activate"
echo "🔹 To deactivate: deactivate" 
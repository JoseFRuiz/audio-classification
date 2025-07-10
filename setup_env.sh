#!/bin/bash
# Setup script for local development with GPU support

echo "🔹 Setting up virtual environment for audio classification..."

# Check available Python versions
echo "🔹 Checking available Python versions..."
if command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
    echo "🔹 Found Python 3.8"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo "🔹 Found Python 3.9"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "🔹 Found Python 3.10"
elif command -v python3.7 &> /dev/null; then
    PYTHON_CMD="python3.7"
    echo "🔹 Found Python 3.7"
else
    PYTHON_CMD="python"
    echo "⚠️ Using system Python (may be old)"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
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

echo "🔹 Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv pytorch_env
source pytorch_env/bin/activate

# Upgrade pip
echo "🔹 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support if available
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 Installing PyTorch with CUDA support..."
    # Try newer versions first, fall back to older ones
    if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
        echo "✅ Successfully installed latest PyTorch with CUDA 12.1"
    elif pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116; then
        echo "✅ Successfully installed PyTorch 1.13.1 with CUDA 11.6"
    elif pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113; then
        echo "✅ Successfully installed PyTorch 1.10.2 with CUDA 11.3"
    else
        echo "❌ Failed to install PyTorch with CUDA support"
        exit 1
    fi
else
    echo "🔹 Installing PyTorch CPU-only version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements with compatible versions
echo "🔹 Installing other requirements..."
pip install pytorch-lightning torchmetrics transformers librosa tqdm pandas numpy scikit-learn

# Test installation
echo "🔹 Testing installation..."
python test_gpu.py

echo "✅ Setup complete!"
echo "🔹 To activate the environment: source pytorch_env/bin/activate"
echo "🔹 To deactivate: deactivate" 
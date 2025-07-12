#!/bin/bash
# Setup script for audio classification project using uv

echo "🔹 Setting up audio classification project..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No NVIDIA GPU detected. Will use CPU."
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "🔹 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo "🔹 Added uv to PATH"
else
    echo "✅ uv already installed"
fi

# Ensure uv is in PATH
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not in PATH. Trying to add it..."
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to find uv. Please restart your shell or run: source $HOME/.local/bin/env"
        exit 1
    fi
fi

echo "✅ uv is available: $(which uv)"

# Install dependencies with uv
echo "🔹 Installing dependencies with uv..."
uv pip install torch torchvision torchaudio pytorch-lightning torchmetrics transformers librosa tqdm pandas numpy scikit-learn accelerate

# Test PyTorch installation
echo "🔹 Testing PyTorch installation..."
uv run python -c "
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
    print('⚠️ CUDA not available - will use CPU')
"

echo "✅ Setup complete!"
echo "🔹 To run your script: uv run python run_experiment_gru_lightning.py [args]"
echo "🔹 To activate the environment: uv shell" 
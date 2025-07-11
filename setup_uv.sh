#!/bin/bash
# Setup script using uv for B200 GPU support

echo "🔹 Setting up uv for B200 GPU support..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🔹 NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No NVIDIA GPU detected."
    exit 1
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "🔹 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
else
    echo "✅ uv already installed"
fi

# Create a new uv project
echo "🔹 Creating uv project..."
if [ -d "audio-classification-uv" ]; then
    echo "🔹 Removing existing uv project..."
    rm -rf audio-classification-uv
fi

mkdir audio-classification-uv
cd audio-classification-uv

# Initialize uv project
echo "🔹 Initializing uv project..."
uv init --python 3.11

# Create pyproject.toml with latest PyTorch
echo "🔹 Creating pyproject.toml with latest PyTorch..."
cat > pyproject.toml << 'EOF'
[project]
name = "audio-classification"
version = "0.1.0"
description = "Audio classification with B200 GPU support"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "torchaudio>=2.2.0",
    "pytorch-lightning>=2.2.0",
    "torchmetrics>=1.3.0",
    "transformers>=4.37.0",
    "librosa>=0.10.0",
    "tqdm>=4.66.0",
    "pandas>=2.1.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = []
EOF

# Install dependencies with uv
echo "🔹 Installing dependencies with uv..."
uv sync

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
    print('❌ CUDA not available')
"

# Copy your scripts to the uv project
echo "🔹 Copying scripts to uv project..."
cp ../run_experiment_gru_lightning.py .
cp ../utils.py .
cp ../test_gpu.py .

echo "✅ uv setup complete!"
echo "🔹 To run your script: cd audio-classification-uv && uv run python run_experiment_gru_lightning.py [args]"
echo "🔹 To activate the environment: cd audio-classification-uv && uv shell" 
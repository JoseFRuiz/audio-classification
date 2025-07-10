#!/bin/bash
# CPU fallback setup script for audio classification

echo "🔹 Setting up conda environment for CPU training (B200 GPU not supported yet)..."

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

# Install latest PyTorch CPU version
echo "🔹 Installing latest PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other packages with pip
echo "🔹 Installing other packages via pip..."
pip install pytorch-lightning torchmetrics transformers librosa tqdm pandas numpy scikit-learn

# Test installation
echo "🔹 Testing installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✅ CPU PyTorch installation successful!')
"

echo "✅ Setup complete!"
echo "🔹 To activate the environment: conda activate audio-classification"
echo "🔹 To deactivate: conda deactivate"
echo ""
echo "⚠️ Note: B200 GPU is not yet supported by PyTorch."
echo "🔹 Training will use CPU, which is still very fast for this model type."
echo "🔹 To run training without GPU, remove the --use_gpu flag from your commands." 
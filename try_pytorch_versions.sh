#!/bin/bash
# Script to try different PyTorch versions for B200 compatibility

echo "🔹 Testing different PyTorch versions for B200 compatibility..."

# Test PyTorch 2.0.1 (older version, might have better compatibility)
echo "🔹 Testing PyTorch 2.0.1..."
uv pip uninstall torch torchvision torchaudio -y
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

uv run python -c "
import torch
print(f'PyTorch 2.0.1 - CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    try:
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.mm(x, y)
        print('✅ PyTorch 2.0.1 GPU test successful!')
    except Exception as e:
        print(f'❌ PyTorch 2.0.1 GPU test failed: {e}')
"

# Test PyTorch 1.13.1 (even older, might work)
echo "🔹 Testing PyTorch 1.13.1..."
uv pip uninstall torch torchvision torchaudio -y
uv pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116

uv run python -c "
import torch
print(f'PyTorch 1.13.1 - CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    try:
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.mm(x, y)
        print('✅ PyTorch 1.13.1 GPU test successful!')
    except Exception as e:
        print(f'❌ PyTorch 1.13.1 GPU test failed: {e}')
"

# Reinstall latest version
echo "🔹 Reinstalling latest PyTorch..."
uv pip uninstall torch torchvision torchaudio -y
uv pip install torch torchvision torchaudio

echo "✅ Testing complete!" 
#!/usr/bin/env python3
import torch
import sys
import os

def check_pytorch_versions():
    print("🔍 PyTorch Version Check")
    print("=" * 50)
    
    # Check Python executable
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[0]}")
    
    # Check PyTorch installation
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"PyTorch location: {torch.__file__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU capability: {torch.cuda.get_device_capability()}")
    
    # Check environment variables
    print(f"\nEnvironment variables:")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"\n✅ Running in virtual environment: {sys.prefix}")
    else:
        print(f"\n❌ Not running in virtual environment")

if __name__ == "__main__":
    check_pytorch_versions() 
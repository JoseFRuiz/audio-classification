#!/usr/bin/env python3
"""
Test script to verify GPU compatibility and functionality
"""

import torch
import sys

def test_gpu():
    print("🔹 Testing GPU compatibility...")
    print(f"🔹 PyTorch version: {torch.__version__}")
    print(f"🔹 CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    print(f"🔹 CUDA version: {torch.version.cuda}")
    print(f"🔹 Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"🔹 GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"🔹 GPU {i} capability: {torch.cuda.get_device_capability(i)}")
        
        # Test basic GPU operations
        try:
            device = torch.device(f'cuda:{i}')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print(f"✅ GPU {i} test successful!")
            
            # Test memory allocation
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"🔹 GPU {i} memory allocated: {memory_allocated:.2f} GB")
            print(f"🔹 GPU {i} memory reserved: {memory_reserved:.2f} GB")
            
        except Exception as e:
            print(f"❌ GPU {i} test failed: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    success = test_gpu()
    if success:
        print("✅ All GPU tests passed!")
        sys.exit(0)
    else:
        print("❌ GPU tests failed!")
        sys.exit(1) 
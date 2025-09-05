#!/usr/bin/env python3
"""
GPU Compatibility Test Script
Run this before the main training to verify GPU setup.
"""

import torch
import torch.nn as nn
import numpy as np
import time

def test_gpu_compatibility():
    """Test basic GPU operations to ensure compatibility."""
    print("🔍 Testing GPU compatibility...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    try:
        # Test 1: Basic tensor operations
        print("🔹 Test 1: Basic tensor operations...")
        device = torch.device("cuda")
        test_tensor = torch.randn(100, 100).to(device)
        result = torch.mm(test_tensor, test_tensor.T)
        print(f"   ✅ Tensor multiplication: {result.shape}")
        
        # Test 2: GRU operations (similar to our model)
        print("🔹 Test 2: GRU operations...")
        gru = nn.GRU(100, 50, 2, batch_first=True).to(device)
        input_tensor = torch.randn(32, 10, 100).to(device)
        output, hidden = gru(input_tensor)
        print(f"   ✅ GRU forward pass: {output.shape}")
        
        # Test 3: Memory allocation
        print("🔹 Test 3: Memory allocation...")
        large_tensor = torch.randn(1000, 1000).to(device)
        del large_tensor
        torch.cuda.empty_cache()
        print("   ✅ Memory allocation and cleanup")
        
        # Test 4: GPU info
        print("🔹 Test 4: GPU information...")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        
        # Test 5: Enable Tensor Cores
        print("🔹 Test 5: Tensor Cores...")
        torch.set_float32_matmul_precision('high')
        print("   ✅ Tensor Cores enabled")
        
        print("✅ All GPU compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ GPU compatibility test failed: {str(e)}")
        return False

def test_dataloader_workers():
    """Test DataLoader with multiple workers."""
    print("\n🔍 Testing DataLoader workers...")
    
    try:
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy dataset
        data = torch.randn(1000, 768)
        labels = torch.randint(0, 2, (1000, 264))
        dataset = TensorDataset(data, labels)
        
        # Test with different worker counts
        for num_workers in [0, 1, 2]:
            try:
                loader = DataLoader(
                    dataset, 
                    batch_size=32, 
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False
                )
                
                # Test iteration
                for i, (x, y) in enumerate(loader):
                    if i >= 2:  # Just test first few batches
                        break
                
                print(f"   ✅ {num_workers} workers: OK")
                
            except Exception as e:
                print(f"   ❌ {num_workers} workers: Failed - {str(e)}")
                
    except Exception as e:
        print(f"❌ DataLoader test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting GPU compatibility tests...")
    
    # Test GPU compatibility
    gpu_ok = test_gpu_compatibility()
    
    if gpu_ok:
        # Test DataLoader workers
        test_dataloader_workers()
        
        print("\n✅ GPU setup appears to be working correctly!")
        print("🔹 You can now run the main training script.")
    else:
        print("\n❌ GPU setup has issues.")
        print("🔹 Please check your CUDA installation and GPU drivers.")
        print("🔹 Consider running with --use_gpu=False for CPU training.")

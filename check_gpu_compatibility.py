#!/usr/bin/env python3
import torch
import subprocess
import sys

def check_gpu_compatibility():
    print("🔍 GPU Compatibility Check")
    print("=" * 50)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"\nGPU {i}: {gpu_name}")
            print(f"  CUDA Capability: {gpu_capability}")
            print(f"  Memory: {gpu_memory:.1f} GB")
        
        # Check PyTorch CUDA version
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        
        # Check CUDA architecture
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                compute_caps = result.stdout.strip().split('\n')
                print(f"GPU Compute Capabilities (from nvidia-smi): {compute_caps}")
        except FileNotFoundError:
            print("nvidia-smi not found")
        
        # Test basic CUDA operations
        print("\nTesting basic CUDA operations...")
        try:
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.mm(x, y)
            print("✅ Basic CUDA operations work")
        except Exception as e:
            print(f"❌ CUDA operations failed: {e}")
        
        # Test GRU on GPU
        print("\nTesting GRU on GPU...")
        try:
            gru = torch.nn.GRU(768, 256, 2, batch_first=True).cuda()
            x = torch.randn(8, 499, 768).cuda()
            output, hidden = gru(x)
            print("✅ GRU works on GPU")
        except Exception as e:
            print(f"❌ GRU failed on GPU: {e}")
            print("This is likely the cause of your training error")
    
    else:
        print("❌ CUDA not available")

if __name__ == "__main__":
    check_gpu_compatibility() 
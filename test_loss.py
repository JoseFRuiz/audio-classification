#!/usr/bin/env python3
"""
Test script to verify the loss function works correctly.
"""

import torch
import torch.nn as nn
from utils import asymmetric_loss

def test_loss_functions():
    """Test different loss functions."""
    print("🔹 Testing loss functions...")
    
    # Create sample data
    batch_size = 4
    num_classes = 10
    
    # Create logits and labels
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print(f"🔹 Logits shape: {logits.shape}")
    print(f"🔹 Labels shape: {labels.shape}")
    print(f"🔹 Labels sum: {labels.sum()}")
    
    # Test BCE loss
    try:
        bce_loss = nn.BCELoss()
        bce_result = bce_loss(torch.sigmoid(logits), labels)
        print(f"✅ BCE Loss: {bce_result.item():.6f}")
    except Exception as e:
        print(f"❌ BCE Loss failed: {str(e)}")
    
    # Test asymmetric loss with different configurations
    configs = [
        {"gamma_pos": 0.0, "gamma_neg": 2.0, "margin": 0.05},
        {"gamma_pos": 0.0, "gamma_neg": 4.0, "margin": 0.05},
        {"gamma_pos": 1.0, "gamma_neg": 4.0, "margin": 0.05},
    ]
    
    for i, config in enumerate(configs):
        try:
            asym_result = asymmetric_loss(logits, labels, **config)
            print(f"✅ Asymmetric Loss {i+1} ({config}): {asym_result.item():.6f}")
        except Exception as e:
            print(f"❌ Asymmetric Loss {i+1} failed: {str(e)}")
    
    # Test with extreme values
    print("\n🔹 Testing with extreme values...")
    extreme_logits = torch.randn(2, 5) * 10  # Large values
    extreme_labels = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]).float()
    
    try:
        extreme_result = asymmetric_loss(extreme_logits, extreme_labels, gamma_pos=0.0, gamma_neg=2.0)
        print(f"✅ Extreme values test: {extreme_result.item():.6f}")
    except Exception as e:
        print(f"❌ Extreme values test failed: {str(e)}")

if __name__ == "__main__":
    test_loss_functions()
    print("\n✅ Loss function tests completed!") 
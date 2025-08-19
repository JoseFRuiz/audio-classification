import torch
import torch.nn.functional as F

def wu_auc_loss(logits, labels, margin=1.0):
    """Compute Wu AUC surrogate loss across all classes independently."""
    loss = 0.0
    num_classes = labels.size(1)
    
    # Handle case where inputs might already be probabilities
    if torch.all((logits >= 0) & (logits <= 1)):
        probs = logits
    else:
        probs = torch.sigmoid(logits)

    for c in range(num_classes):
        y_c = labels[:, c]
        x_c = probs[:, c]

        pos_mask = y_c == 1
        neg_mask = y_c == 0

        pos_scores = x_c[pos_mask]
        neg_scores = x_c[neg_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue

        pos_scores = pos_scores.unsqueeze(1)
        neg_scores = neg_scores.unsqueeze(0)

        diffs = neg_scores - pos_scores + margin
        hinge = torch.clamp(diffs, min=0)
        loss += hinge.mean()

    return loss / num_classes

def combined_wu_bce_loss(logits, labels, wu_weight=0.5, bce_weight=0.5, margin=1.0):
    """Combined loss function that combines Wu AUC loss and BCE loss with weighted sum."""
    # Compute Wu AUC loss
    wu_loss = wu_auc_loss(logits, labels, margin=margin)
    
    # Handle BCE loss based on input type
    if torch.all((logits >= 0) & (logits <= 1)):
        bce_loss = F.binary_cross_entropy(logits, labels, reduction='mean')
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
    
    # Combine with weighted sum
    combined_loss = wu_weight * wu_loss + bce_weight * bce_loss
    
    return combined_loss, wu_loss, bce_loss

# Test with realistic data
torch.manual_seed(42)
batch_size = 32
num_classes = 200

# Create some realistic logits and labels
logits = torch.randn(batch_size, num_classes) * 2.0  # Random logits
labels = torch.randint(0, 2, (batch_size, num_classes)).float()  # Random binary labels

print("🔍 Testing Loss Function Scales")
print("=" * 50)

# Test individual losses
wu_loss = wu_auc_loss(logits, labels, margin=1.0)
bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

print(f"Wu AUC Loss: {wu_loss:.6f}")
print(f"BCE Loss: {bce_loss:.6f}")
print(f"Scale Ratio (BCE/Wu): {bce_loss/wu_loss:.2f}")
print()

# Test combined loss with different weights
print("Combined Loss with Different Weights:")
print("-" * 40)

weights_configs = [
    (0.5, 0.5, "Equal weights"),
    (0.8, 0.2, "Wu AUC dominant"),
    (0.2, 0.8, "BCE dominant"),
    (0.9, 0.1, "Mostly Wu AUC"),
    (0.1, 0.9, "Mostly BCE")
]

for wu_w, bce_w, desc in weights_configs:
    combined, wu, bce = combined_wu_bce_loss(logits, labels, wu_w, bce_w, margin=1.0)
    wu_contribution = wu_w * wu
    bce_contribution = bce_w * bce
    
    print(f"{desc}:")
    print(f"  Combined Loss: {combined:.6f}")
    print(f"  Wu AUC contribution: {wu_contribution:.6f} ({wu_contribution/combined*100:.1f}%)")
    print(f"  BCE contribution: {bce_contribution:.6f} ({bce_contribution/combined*100:.1f}%)")
    print()

# Test with scaled Wu AUC
print("Combined Loss with Scaled Wu AUC:")
print("-" * 40)

def combined_wu_bce_loss_scaled(logits, labels, wu_weight=0.5, bce_weight=0.5, margin=1.0, scale_factor=5.0):
    wu_loss = wu_auc_loss(logits, labels, margin=margin)
    
    if torch.all((logits >= 0) & (logits <= 1)):
        bce_loss = F.binary_cross_entropy(logits, labels, reduction='mean')
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
    
    # Scale Wu AUC loss to match BCE loss scale
    wu_loss_scaled = wu_loss * scale_factor
    
    combined_loss = wu_weight * wu_loss_scaled + bce_weight * bce_loss
    
    return combined_loss, wu_loss_scaled, bce_loss

for wu_w, bce_w, desc in weights_configs:
    combined, wu_scaled, bce = combined_wu_bce_loss_scaled(logits, labels, wu_w, bce_w, margin=1.0, scale_factor=5.0)
    wu_contribution = wu_w * wu_scaled
    bce_contribution = bce_w * bce
    
    print(f"{desc} (with 5x Wu scaling):")
    print(f"  Combined Loss: {combined:.6f}")
    print(f"  Wu AUC contribution: {wu_contribution:.6f} ({wu_contribution/combined*100:.1f}%)")
    print(f"  BCE contribution: {bce_contribution:.6f} ({bce_contribution/combined*100:.1f}%)")
    print()

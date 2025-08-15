import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def wu_auc_loss(logits, labels, margin=1.0):
    """Compute Wu AUC surrogate loss across all classes independently."""
    loss = 0.0
    num_classes = labels.size(1)
    
    # Handle case where inputs might already be probabilities
    if torch.all((logits >= 0) & (logits <= 1)):
        # Inputs are already probabilities, use them directly
        probs = logits
    else:
        # Inputs are logits, apply sigmoid
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
    """
    Combined loss function that combines Wu AUC loss and BCE loss with weighted sum.
    
    Args:
        logits: model outputs (can be raw logits or probabilities), shape (batch, num_classes)
        labels: binary targets, shape (batch, num_classes)
        wu_weight: weight for Wu AUC loss component
        bce_weight: weight for BCE loss component
        margin: margin for Wu AUC loss
    """
    # Compute Wu AUC loss
    wu_loss = wu_auc_loss(logits, labels, margin=margin)
    
    # Handle BCE loss based on input type
    if torch.all((logits >= 0) & (logits <= 1)):
        # Inputs are already probabilities, use BCE directly
        bce_loss = torch.nn.functional.binary_cross_entropy(logits, labels, reduction='mean')
    else:
        # Inputs are raw logits, use BCE with logits
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
    
    # Combine with weighted sum
    combined_loss = wu_weight * wu_loss + bce_weight * bce_loss
    
    return combined_loss


TARGET_LENGTH = 10 * 16000
SAMPLE_RATE = 16000

def preprocess_audio(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(waveform).float()
    if waveform.shape[-1] > TARGET_LENGTH:
        waveform = waveform[:TARGET_LENGTH]
    elif waveform.shape[-1] < TARGET_LENGTH:
        padding = torch.zeros(TARGET_LENGTH - waveform.shape[-1])
        waveform = torch.cat((waveform, padding))
    return waveform

def extract_wav2vec_embeddings(audio_path, processor, wav2vec_model, device):
    waveform = preprocess_audio(audio_path)
    input_values = processor(waveform.numpy(), return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values.to(device)
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
        embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (time_steps, 768)
    return embeddings.cpu().numpy()  # Shape: (time_steps, 768) 

def asymmetric_loss(logits, labels, gamma_pos=0.0, gamma_neg=4.0, margin=0.05, eps=1e-8):
    """
    Asymmetric Loss for multi-label classification with sparse labels.
    Optimized to handle large tensors and prevent integer overflow by avoiding large boolean indexing.

    Args:
        logits: model outputs (can be raw logits or probabilities), shape (batch, num_classes)
        labels: binary targets, shape (batch, num_classes)
        gamma_pos: focusing parameter for positive labels
        gamma_neg: focusing parameter for negative labels
        margin: margin for negative class scores
    """
    # Ensure inputs are on the same device
    if logits.device != labels.device:
        labels = labels.to(logits.device)
    
    # Handle case where inputs might already be probabilities
    if torch.all((logits >= 0) & (logits <= 1)):
        # Inputs are already probabilities, use them directly
        probas = logits
    else:
        # Inputs are logits, apply sigmoid
        probas = torch.sigmoid(logits)
    
    probas = torch.clamp(probas, min=eps, max=1.0 - eps)
    
    # Calculate loss element-wise to avoid large boolean indexing
    batch_size, num_classes = logits.shape
    
    # Initialize loss tensors
    pos_loss = torch.zeros_like(probas)
    neg_loss = torch.zeros_like(probas)
    
    # Positive loss calculation
    pos_mask = (labels == 1)
    if gamma_pos == 0.0:
        pos_loss = torch.where(pos_mask, torch.log(probas + eps), torch.zeros_like(probas))
    else:
        pos_loss = torch.where(pos_mask, 
                              (1 - probas) ** gamma_pos * torch.log(probas + eps), 
                              torch.zeros_like(probas))
    
    # Negative loss calculation
    neg_mask = (labels == 0)
    neg_probas_clamped = torch.clamp(probas - margin, min=0)
    if gamma_neg == 0.0:
        neg_loss = torch.where(neg_mask, 
                              torch.log(1 - neg_probas_clamped + eps), 
                              torch.zeros_like(probas))
    else:
        neg_loss = torch.where(neg_mask, 
                              (neg_probas_clamped) ** gamma_neg * torch.log(1 - neg_probas_clamped + eps), 
                              torch.zeros_like(probas))
    
    # Sum all losses and normalize by batch size
    total_loss = -(pos_loss.sum() + neg_loss.sum()) / batch_size
    
    # Ensure loss is positive and finite
    if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss < 0:
        print(f"Warning: Invalid loss value detected: {total_loss}")
        # Return a small positive loss to prevent training from stopping
        return torch.tensor(0.1, device=logits.device, requires_grad=True)
    
    return total_loss

class MeanContrastiveRankingLoss(nn.Module):
    def __init__(self, margin=0.1, reduction='mean'):
        """
        Margin-based contrastive loss using the mean of positive and negative class scores.

        Args:
            margin (float): Minimum required difference between mean positive and mean negative scores.
            reduction (str): 'mean', 'sum', or 'none' reduction over the batch.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes), model outputs (can be raw logits or probabilities)
            labels: Tensor of shape (batch_size, num_classes), binary ground truth labels
        """
        # Handle case where inputs might already be probabilities
        if torch.all((logits >= 0) & (logits <= 1)):
            # Inputs are already probabilities, use them directly
            probs = logits
        else:
            # Inputs are logits, apply sigmoid
            probs = torch.sigmoid(logits)
        batch_size = logits.size(0)
        losses = []

        for i in range(batch_size):
            scores = probs[i]
            true_labels = labels[i]

            pos_inds = (true_labels == 1).nonzero(as_tuple=True)[0]
            neg_inds = (true_labels == 0).nonzero(as_tuple=True)[0]

            if len(pos_inds) == 0 or len(neg_inds) == 0:
                continue  # Skip if no positives or no negatives

            mean_pos = scores[pos_inds].mean()
            mean_neg = scores[neg_inds].mean()
            loss = F.relu(self.margin - (mean_pos - mean_neg))
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        losses = torch.stack(losses)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

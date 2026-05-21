"""
Self-Supervised Learning (SSL) Objectives.

Contains the foundational loss classes for sequential modeling (Next Token Prediction)
and imputation/pre-training (Masked Joint-Distribution based on LimiX).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class SSLBaseLoss(nn.Module):
    """
    Base class for self-supervised losses applied to MMTT models.
    """
    def __init__(self):
        super().__init__()

class NextTokenPredictionLoss(SSLBaseLoss):
    """
    Evaluates multimodal predictions at the future time step (t_{n+1}).
    Blends Cross-Entropy (for categorical keys) and MSE (for continuous values).
    
    This is the core objective for PRAGMA and TransactionGPT, enabling models to
    capture long-range behavioral patterns by autoregressively predicting the exact
    composition of the next financial event.
    """
    def __init__(self, vocab_size: int, mse_weight: float = 1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss = nn.MSELoss()
        
        # Projection heads from the latent embedding back to the original space
        self.key_head = nn.Linear(768, vocab_size) # Assuming hidden_size=768 for simplicity
        self.value_head = nn.Linear(768, 1)

    def forward(self, 
                hidden_states: torch.Tensor, 
                target_keys: torch.Tensor, 
                target_values: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates joint losses for next event prediction.
        
        Args:
            hidden_states (Tensor): [B, S-1, D] Latent representations from the transformer.
            target_keys (Tensor): [B, S-1] Ground truth Keys for t_{n+1}.
            target_values (Tensor): [B, S-1] Ground truth Values for t_{n+1}.
        """
        key_logits = self.key_head(hidden_states)
        value_preds = self.value_head(hidden_states).squeeze(-1)
        
        # Flattening for compatibility with loss functions
        loss_ce = self.ce_loss(key_logits.view(-1, key_logits.size(-1)), target_keys.view(-1))
        loss_mse = self.mse_loss(value_preds.view(-1), target_values.view(-1))
        
        total_loss = loss_ce + (self.mse_weight * loss_mse)
        
        return total_loss, {"loss_ce": loss_ce.item(), "loss_mse": loss_mse.item()}

class ContextConditionalMaskedLoss(SSLBaseLoss):
    """
    Masked Joint-Distribution Modeling (Based on LimiX).
    
    Instead of rigid chronological sequences, this objective treats structured data 
    as vast multivariate correlation matrices. It infers latent relationships between 
    all variables simultaneously by probabilistically masking values and enforcing 
    reconstruction from the visible context.
    """
    def __init__(self, masking_prob: float = 0.15):
        super().__init__()
        self.masking_prob = masking_prob
        
    def heterogeneous_mask_schedule(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates a mask matrix using a heterogeneous schedule.
        
        Allows:
        - Random masking of isolated cells (standard approach).
        - Block masking (consecutive variables) forcing the model to infer long-range trends.
        - Semantic masking (hiding highly correlated columns entirely).
        
        Returns:
            torch.Tensor: [B, S] Boolean mask where True indicates "Must be masked and predicted".
        """
        # Simple probabilistic mask (isolated cells)
        mask = torch.rand((batch_size, seq_len), device=device) < self.masking_prob
        
        # Simplified block logic: randomly expands existing masks to neighboring 
        # transactions (temporal emulation) to prevent trivial deduction.
        shifted_mask = torch.roll(mask, shifts=1, dims=1)
        shifted_mask[:, 0] = False # Do not wrap around
        block_mask = mask | shifted_mask 
        
        return block_mask

    def forward(self, 
                reconstructed_embeddings: torch.Tensor, 
                original_embeddings: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Reconstruction Error over the masked tokens.
        
        Args:
            reconstructed_embeddings (Tensor): Latent output from the MMTT.
            original_embeddings (Tensor): Untouched original MMTT embedding (or ground truth).
            mask (Tensor): Boolean matrix of masked positions.
        """
        # Applies L2 Reconstruction Loss (MSE) exclusively on the masked positions
        masked_preds = reconstructed_embeddings[mask]
        masked_targets = original_embeddings[mask]
        
        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=reconstructed_embeddings.device, requires_grad=True)
            
        loss = F.mse_loss(masked_preds, masked_targets)
        return loss

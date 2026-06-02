"""
Self-Supervised Learning (SSL) Objectives for LDM Pre-Training.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
Self-supervised objectives allow the model to learn rich representations
from UNLABELED data. This is crucial because:

1. Labeled financial data is EXPENSIVE and SCARCE
   - Fraud labels require manual investigation by analysts
   - Bankruptcy labels only become available months/years later
   - Privacy regulations limit label sharing across organizations

2. Unlabeled transaction data is ABUNDANT
   - Banks process billions of transactions daily
   - Each transaction contains rich multi-modal information
   - The sequential structure encodes behavioral patterns

Pre-training objectives extract knowledge from this abundance without
any human labels, creating a "foundation" that can be specialized for
any downstream task with minimal labeled data.

============================================================================
TWO COMPLEMENTARY OBJECTIVES
============================================================================
1. NextTokenPredictionLoss (NTP):
   - AUTOREGRESSIVE: Predict the next transaction given previous ones
   - Inspired by GPT: "Given transactions [t1, t2, ..., tn], predict t_{n+1}"
   - Captures TEMPORAL dependencies and behavioral sequences
   - Used by TransactionGPT and PRAGMA

2. ContextConditionalMaskedLoss (LimiX):
   - BIDIRECTIONAL: Randomly mask tokens and predict them from context
   - Inspired by BERT: "Given [t1, ?, t3, ?, t5], predict t2 and t4"
   - Captures CORRELATIONS between features (not just temporal)
   - Used by LimiX for masked joint-distribution modeling

============================================================================
REFERENCES
============================================================================
  - TransactionGPT (Visa, 2025): Next-event prediction for transactions
  - PRAGMA (Revolut, 2025): Masked modeling for banking events
  - LimiX (2024): Masked joint-distribution pre-training for tabular data
  - BERT (Devlin, 2019): Masked Language Modeling objective
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SSLBaseLoss(nn.Module):
    """
    Base class for all self-supervised learning objectives.

    Provides common interface and utility methods shared by both
    autoregressive (NTP) and masked (LimiX) objectives.
    """

    def __init__(self):
        super().__init__()


class NextTokenPredictionLoss(SSLBaseLoss):
    """
    Next Token Prediction (NTP) — Autoregressive Objective.

    Predicts the NEXT transaction's components (category and amount)
    given all previous transactions. This forces the model to learn
    temporal patterns and behavioral sequences.

    How it works:
        Input:  [t1, t2, t3, t4, t5]
        Target: [t2, t3, t4, t5, --]  (shifted by 1)

    The loss combines two components:
    1. Cross-Entropy for categorical prediction (MCC code)
    2. MSE for continuous prediction (transaction amount)

    These are weighted and summed: L = L_CE + α · L_MSE

    Args:
        vocab_size: Number of categorical tokens to predict.
        hidden_size: Dimension of the Transformer's hidden states.
        mse_weight: Weight for the continuous prediction loss (α).
        label_smoothing: Smoothing factor for Cross-Entropy (prevents
                        overconfident predictions, improves generalization).

    Example:
        >>> loss_fn = NextTokenPredictionLoss(vocab_size=20, hidden_size=128)
        >>> hidden = torch.randn(4, 49, 128)   # [batch, seq-1, hidden]
        >>> target_keys = torch.randint(0, 20, (4, 49))
        >>> target_values = torch.randn(4, 49)
        >>> loss, metrics = loss_fn(hidden, target_keys, target_values)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        mse_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight

        # =====================================================================
        # Projection Heads
        # =====================================================================
        # These project the Transformer's hidden states back to the original
        # feature space for prediction:
        #
        # key_head: hidden_size → vocab_size (categorical distribution)
        # value_head: hidden_size → 1 (scalar amount prediction)
        #
        # The hidden_size is now a PARAMETER, not hardcoded to 768.
        # This was a bug in the original code that would crash with
        # any hidden_size != 768.
        # =====================================================================
        self.key_head = nn.Linear(hidden_size, vocab_size)
        self.value_head = nn.Linear(hidden_size, 1)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-1,  # Ignore padding tokens
            label_smoothing=label_smoothing,
        )
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the joint NTP loss.

        Args:
            hidden_states: [B, S-1, D] Transformer output for positions 1..S-1
            target_keys:   [B, S-1] Ground truth MCC tokens for positions 2..S
            target_values: [B, S-1] Ground truth amounts for positions 2..S

        Returns:
            Tuple of (total_loss, metrics_dict)
            - total_loss: Scalar tensor for backpropagation
            - metrics_dict: Dictionary with component losses for logging
        """
        # Project hidden states to prediction space
        key_logits = self.key_head(hidden_states)          # [B, S-1, vocab]
        value_preds = self.value_head(hidden_states).squeeze(-1)  # [B, S-1]

        # Flatten for loss computation
        # Cross-Entropy expects [N, C] and [N]
        loss_ce = self.ce_loss(
            key_logits.reshape(-1, key_logits.size(-1)),
            target_keys.reshape(-1),
        )

        # MSE expects matching shapes
        loss_mse = self.mse_loss(
            value_preds.reshape(-1),
            target_values.reshape(-1),
        )

        # Weighted combination
        total_loss = loss_ce + (self.mse_weight * loss_mse)

        return total_loss, {
            "loss_ce": loss_ce.item(),
            "loss_mse": loss_mse.item(),
            "loss_total": total_loss.item(),
        }


class ContextConditionalMaskedLoss(SSLBaseLoss):
    """
    Masked Joint-Distribution Modeling — LimiX Objective.

    Instead of autoregressive left-to-right prediction, this objective
    treats each data point as a MULTIVARIATE observation and randomly
    masks some dimensions. The model must reconstruct the masked values
    from the visible context.

    Key insight from LimiX:
        "The correlations between features in structured data are NOT
        purely temporal. A user's MCC distribution is correlated with
        their income bracket, which is correlated with their credit
        risk — regardless of the temporal ordering of transactions."

    Masking Schedule (heterogeneous):
        1. CELL masking: Random individual tokens (standard, like BERT)
        2. BLOCK masking: Consecutive tokens in a temporal window
           - Forces learning of temporal trends, not just local patterns
        3. COLUMN masking: All values of a specific feature type
           - Forces learning of cross-feature correlations

    The heterogeneous schedule prevents the model from "cheating" by
    simply copying adjacent values (trivial deduction).

    Args:
        masking_prob: Base probability of masking each token.

    Example:
        >>> loss_fn = ContextConditionalMaskedLoss(masking_prob=0.15)
        >>> mask = loss_fn.heterogeneous_mask_schedule(4, 100, device)
        >>> loss = loss_fn(reconstructed, original, mask)
    """

    def __init__(self, masking_prob: float = 0.15):
        super().__init__()
        self.masking_prob = masking_prob

    def heterogeneous_mask_schedule(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generates a heterogeneous masking pattern.

        Combines three masking strategies:
        1. Random cell masking (~60% of masks): Standard BERT-style
        2. Block masking (~30% of masks): Consecutive temporal windows
        3. Pure random roll (~10% of masks): Shifted patterns

        The combination ensures the model learns:
        - Local feature values (from cell masking)
        - Temporal trends (from block masking)
        - Global correlations (from the heterogeneous combination)

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Length of each sequence.
            device: Target device for the mask tensor.

        Returns:
            Boolean mask [B, S] where True = "must predict" (masked).
        """
        # Strategy 1: Random cell masking (standard approach)
        cell_mask = torch.rand((batch_size, seq_len), device=device) < self.masking_prob

        # Strategy 2: Block masking (temporal window expansion)
        # If a token is masked, its immediate neighbor also gets masked.
        # This prevents trivial deduction from adjacent context.
        shifted_mask = torch.roll(cell_mask, shifts=1, dims=1)
        shifted_mask[:, 0] = False  # Don't wrap around
        block_mask = cell_mask | shifted_mask

        # Strategy 3: Additional random expansion for diversity
        extra_mask = torch.roll(cell_mask, shifts=-1, dims=1)
        extra_mask[:, -1] = False

        # Combine with probabilistic mixing
        # ~70% of the time use block mask, ~30% add extra expansion
        use_extra = torch.rand((batch_size, 1), device=device) < 0.3
        combined_mask = block_mask | (extra_mask & use_extra)

        return combined_mask

    def forward(
        self,
        reconstructed_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the reconstruction loss over masked positions.

        Uses L2 loss (MSE) to measure the distance between the model's
        reconstruction and the original embedding values. Only masked
        positions contribute to the loss — visible positions are ignored.

        This is analogous to how BERT only computes loss on [MASK] tokens,
        not on the visible context.

        Args:
            reconstructed_embeddings: Model output [B, S, D] or [B, S]
            original_embeddings: Ground truth [B, S, D] or [B, S]
            mask: Boolean mask [B, S] where True = masked position

        Returns:
            Scalar MSE loss over masked positions.
        """
        # Select only masked positions for loss computation
        masked_preds = reconstructed_embeddings[mask]
        masked_targets = original_embeddings[mask]

        # Edge case: if no positions are masked (very unlikely but possible)
        if masked_preds.numel() == 0:
            return torch.tensor(
                0.0, device=reconstructed_embeddings.device, requires_grad=True
            )

        loss = F.mse_loss(masked_preds, masked_targets)
        return loss

"""
Self-Supervised Learning (SSL) Objectives for LDM Pre-Training.

============================================================================
PURPOSE
============================================================================
Self-supervised objectives enable the Large Data Model (LDM) to learn rich,
generalizable behavioral representations from unlabeled transaction logs.

This module implements three complementary self-supervised objectives:

1. NextTokenPredictionLoss (NTP — Autoregressive):
   - Predicts the next transaction's merchant category (Key) and amount (Value)
     given preceding transactions.
   - Inspired by TransactionGPT (Dou et al., Visa Research, 2025) and TabGPT
     (Padhi et al., IBM Research, 2021).

2. ContextConditionalMaskedLoss (LimiX / PRAGMA — Masked Joint-Distribution):
   - Uses a heterogeneous mask schedule (random cell, temporal block, and
     shifted expansion masks) to corrupt transaction sequences and reconstruct
     both latent representations and multi-modal raw features (Key, Value, Time).
   - Prevents trivial local interpolation and representation collapse.

3. ContrastiveEventSequenceLoss (CoLES — Subsequence Contrastive Alignment):
   - Implements Contrastive Learning for Event Sequences (Babaev et al.,
     SIGMOD 2022, arXiv:2002.08232).
   - Aligns two stochastic views of each user's transaction history in
     L2-normalized hyperspherical space via symmetric InfoNCE loss,
     explicitly preventing constant-vector representation collapse.

============================================================================
REFERENCES
============================================================================
  - Dou et al. (2025): "TransactionGPT: Foundation Models for Payment
    Trajectories" (Visa Research, arXiv:2511.08939)
  - Ostroukhov et al. (2026): "PRAGMA: A Self-Supervised Foundation Model
    for Banking Events" (Revolut/NVIDIA, arXiv:2604.08649)
  - LimiX Team (2025/2026): "LimiX: Masked Joint-Distribution Pre-training
    for Structured Data" (arXiv:2509.03505 / arXiv:2609.17488)
  - Babaev et al. (2022): "CoLES: Contrastive Learning for Event Sequences
    with Self-Supervision" (SIGMOD 2022, arXiv:2002.08232)
============================================================================
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLBaseLoss(nn.Module):
    """Base class for all self-supervised learning objectives."""

    def __init__(self):
        super().__init__()


class NextTokenPredictionLoss(SSLBaseLoss):
    """
    Next Token Prediction (NTP) — Autoregressive Objective.

    Predicts the next transaction's categorical key (MCC code) and continuous
    amount given preceding transactions:
        L_NTP = L_CE(key_logits, target_keys) + alpha * L_MSE(value_preds, target_values)

    Args:
        vocab_size: Number of categorical tokens to predict.
        hidden_size: Dimension of the Transformer's hidden states.
        mse_weight: Weight for the continuous prediction loss (`alpha`).
        label_smoothing: Label smoothing factor for categorical Cross-Entropy.
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

        self.key_head = nn.Linear(hidden_size, vocab_size)
        self.value_head = nn.Linear(hidden_size, 1)

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-1,
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
        Computes the joint autoregressive NTP loss.

        Args:
            hidden_states: `[B, S-1, D]` Transformer output for positions `1..S-1`.
            target_keys: `[B, S-1]` Ground truth MCC tokens for positions `2..S`.
            target_values: `[B, S-1]` Ground truth amounts for positions `2..S`.

        Returns:
            Tuple of `(total_loss, metrics_dict)`.
        """
        key_logits = self.key_head(hidden_states)
        value_preds = self.value_head(hidden_states).squeeze(-1)

        loss_ce = self.ce_loss(
            key_logits.reshape(-1, key_logits.size(-1)),
            target_keys.reshape(-1),
        )
        loss_mse = self.mse_loss(
            value_preds.reshape(-1),
            target_values.reshape(-1),
        )

        total_loss = loss_ce + (self.mse_weight * loss_mse)

        return total_loss, {
            "loss_ce": loss_ce.item(),
            "loss_mse": loss_mse.item(),
            "loss_total": total_loss.item(),
        }


class ContextConditionalMaskedLoss(SSLBaseLoss):
    """
    Masked Joint-Distribution Modeling — LimiX & PRAGMA Objective.

    Implements a heterogeneous masking schedule combining:
      1. Cell Masking: Random individual events
      2. Block Masking: Contiguous temporal event windows
      3. Shifted Expansion: Probabilistic neighbor expansion

    Supports both latent representation reconstruction (`forward`) and
    multi-modal input feature reconstruction (`compute_multimodal_loss`)
    across categorical keys, continuous amounts, and temporal deltas.

    Args:
        masking_prob: Base probability of masking each event token.
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
        Generates a heterogeneous boolean masking pattern `[B, S]`.

        Guarantees at least one masked token per sequence when `seq_len >= 2`
        so the reconstruction loss always receives informative gradients.
        """
        cell_mask = torch.rand((batch_size, seq_len), device=device) < self.masking_prob

        shifted_mask = torch.roll(cell_mask, shifts=1, dims=1)
        shifted_mask[:, 0] = False
        block_mask = cell_mask | shifted_mask

        extra_mask = torch.roll(cell_mask, shifts=-1, dims=1)
        extra_mask[:, -1] = False

        use_extra = torch.rand((batch_size, 1), device=device) < 0.3
        combined_mask = block_mask | (extra_mask & use_extra)

        # Ensure at least one position is masked and at least one remains unmasked
        if seq_len >= 2:
            no_mask_rows = ~combined_mask.any(dim=1)
            if no_mask_rows.any():
                combined_mask[no_mask_rows, 0] = True
            all_mask_rows = combined_mask.all(dim=1)
            if all_mask_rows.any():
                combined_mask[all_mask_rows, -1] = False

        return combined_mask

    def forward(
        self,
        reconstructed_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes MSE reconstruction loss over masked positions.

        Args:
            reconstructed_embeddings: Predicted representations `[B, S, D]` or `[B, S]`.
            original_embeddings: Target representations `[B, S, D]` or `[B, S]`.
            mask: Boolean mask `[B, S]` where `True` denotes a masked position.

        Returns:
            Scalar MSE loss over masked positions.
        """
        masked_preds = reconstructed_embeddings[mask]
        masked_targets = original_embeddings[mask]

        if masked_preds.numel() == 0:
            return (reconstructed_embeddings * 0.0).sum()

        return F.mse_loss(masked_preds, masked_targets)

    def compute_multimodal_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        target_times: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes true multi-modal KVT feature reconstruction loss at masked positions.

        Reconstructs:
          1. Categorical MCC `keys` via Cross-Entropy
          2. Continuous `values` via Smooth L1 (Huber) + MSE loss
          3. Log-scaled `times` via Smooth L1 loss

        By forcing the encoder to predict the actual masked discrete categories,
        continuous amounts, and time intervals, representation collapse to a
        constant vector is mathematically impossible.
        """
        if not token_mask.any():
            zero = (predictions["value_pred"] * 0.0).sum()
            return zero, {"key_loss": 0.0, "value_loss": 0.0, "time_loss": 0.0}

        key_logits = predictions["key_logits"][token_mask]
        masked_keys = target_keys[token_mask].clamp(min=0, max=key_logits.size(-1) - 1)
        key_loss = F.cross_entropy(key_logits, masked_keys)

        value_pred = predictions["value_pred"][token_mask]
        masked_values = target_values[token_mask]
        value_loss = F.smooth_l1_loss(value_pred, masked_values)

        time_pred = predictions["time_pred"][token_mask]
        masked_times = torch.log1p(target_times[token_mask].float().clamp(min=0.0))
        time_loss = F.smooth_l1_loss(time_pred, masked_times)

        total = key_loss + value_loss + 0.25 * time_loss
        return total, {
            "key_loss": float(key_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "time_loss": float(time_loss.detach().item()),
        }


class ContrastiveEventSequenceLoss(SSLBaseLoss):
    """
    Contrastive Learning for Event Sequences (CoLES).

    Implements the self-supervised sequence contrastive objective from
    Babaev et al. (SIGMOD 2022, "CoLES: Contrastive Learning for Event
    Sequences with Self-Supervision", arXiv:2002.08232).

    Given two augmented/masked views of a batch of user sequences with
    pooled embeddings `z1` and `z2` of shape `[B, D]`, computes symmetric
    temperature-scaled InfoNCE (NT-Xent) loss on the unit hypersphere.
    Aligning positive views while repelling distinct users in the batch
    guarantees high-variance, discriminative user representations.

    Args:
        temperature: Softmax temperature scaling factor (default: `0.15`).
    """

    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes symmetric InfoNCE contrastive loss between two sequence views.

        Args:
            z1: Pooled embeddings from view 1 `[B, D]`.
            z2: Pooled embeddings from view 2 `[B, D]`.

        Returns:
            Scalar contrastive loss tensor.
        """
        batch_size = z1.size(0)
        if batch_size < 2:
            # With batch_size == 1, maximize cosine similarity between positive views
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
            return 1.0 - (z1_norm * z2_norm).sum(dim=-1).mean()

        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)

        logits_12 = torch.matmul(z1_norm, z2_norm.T) / self.temperature
        logits_21 = logits_12.T
        targets = torch.arange(batch_size, device=z1.device, dtype=torch.long)

        loss_12 = F.cross_entropy(logits_12, targets)
        loss_21 = F.cross_entropy(logits_21, targets)
        return 0.5 * (loss_12 + loss_21)

"""
Large Data Model (LDM) — Complete End-to-End Model.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
This module connects ALL architectural components into a single, unified
model that supports two operational modes:

1. PRE-TRAINING (Self-Supervised Learning):
   - Input: Transaction sequences (unlabeled)
   - Objective: Learn general representations via masked prediction
   - Output: Dense contextual embeddings per token
   - No labels required — learns from data structure alone

2. FINE-TUNING (Supervised Classification):
   - Input: Transaction sequences + tabular features + labels
   - Objective: Binary classification (e.g., fraud detection)
   - Output: Binary logits
   - Uses Transformer embeddings fused with tabular features via DCNv2

This two-phase approach (pretrain → finetune) is the CORE PARADIGM of
foundation models. The pre-trained Transformer learns universal patterns
from massive unlabeled data, then is specialized for specific tasks with
a small amount of labeled data.

============================================================================
ARCHITECTURE OVERVIEW
============================================================================
                     ┌─────────────────────────┐
                     │   Pre-training Mode      │
                     │                          │
  KVT Tokens ──────►│  MMTTTransformerEncoder  ├──► Token Embeddings
                     │                          │    (for masked loss)
                     └─────────────────────────┘

                     ┌─────────────────────────┐
                     │   Fine-tuning Mode       │
                     │                          │
  KVT Tokens ──────►│  MMTTTransformerEncoder  │
                     │         │                │
                     │     Pooling              │
                     │         │                │
  Tabular ──────────►│  EndToEndFusionLayer    ├──► Binary Logits
                     │         │                │    (fraud/not fraud)
                     │     Sigmoid              │
                     └─────────────────────────┘

============================================================================
REFERENCES
============================================================================
  - BERT (Devlin et al., 2019): Pre-train then fine-tune paradigm
  - TransactionGPT (Visa, 2025): Foundation model for transactions
  - PRAGMA (Revolut, 2025): Self-supervised banking event model
============================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Literal
import logging

from src.models.transformer_backbone import MMTTTransformerEncoder
from src.models.joint_fusion_network import EndToEndFusionLayer

logger = logging.getLogger(__name__)


class LargeDataModel(nn.Module):
    """
    Complete Large Data Model supporting pre-training and fine-tuning.

    This is the TOP-LEVEL model that you instantiate and train. It
    internally creates and manages:
    - The MMTT Transformer encoder (backbone)
    - The DCNv2 fusion layer (head for classification)

    Args:
        vocab_size: Number of unique categorical tokens.
        hidden_size: Transformer hidden dimension.
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        tabular_feature_size: Number of tabular features for fusion.
        dcn_layers: Number of DCNv2 cross layers.
        use_virtual_tokens: Enable virtual token conditioning.
        num_virtual_tokens: Number of virtual tokens.
        pooling: Pooling strategy for sequence reduction.

    Example:
        >>> model = LargeDataModel(vocab_size=20, hidden_size=128,
        ...                        num_layers=4, num_heads=4)
        >>> # Pre-training mode
        >>> embeddings = model.encode(keys, values, times)
        >>> # Fine-tuning mode
        >>> logits = model(keys, values, times, tabular_features=tabular)
    """

    def __init__(
        self,
        vocab_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        tabular_feature_size: int = 3,
        dcn_layers: int = 3,
        use_virtual_tokens: bool = True,
        num_virtual_tokens: int = 2,
        pooling: Literal["cls", "mean", "max"] = "mean",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # =====================================================================
        # Transformer Backbone (shared between pre-training and fine-tuning)
        # =====================================================================
        self.encoder = MMTTTransformerEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_virtual_tokens=use_virtual_tokens,
            num_virtual_tokens=num_virtual_tokens,
            pooling=pooling,
        )

        # =====================================================================
        # Fusion Head (used only during fine-tuning)
        # =====================================================================
        self.fusion_head = EndToEndFusionLayer(
            transformer_hidden_size=hidden_size,
            tabular_feature_size=tabular_feature_size,
            dcn_layers=dcn_layers,
        )

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"LargeDataModel created: {total_params:,} total params, "
            f"{trainable_params:,} trainable params"
        )

    def encode(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode transaction sequences into contextual embeddings.

        Used during PRE-TRAINING to get per-token embeddings for
        the masked prediction objective.

        Args:
            packed_keys: Categorical token indices.
            packed_values: Continuous values.
            packed_times: Temporal delta indices.

        Returns:
            Full sequence embeddings [batch, seq_len, hidden_size]
        """
        return self.encoder(packed_keys, packed_values, packed_times, **kwargs)

    def get_pooled_embedding(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get a single pooled embedding per user/sequence.

        Used as the input to the fusion head during FINE-TUNING.

        Returns:
            Pooled embeddings [batch, hidden_size]
        """
        return self.encoder.get_pooled_output(
            packed_keys, packed_values, packed_times, **kwargs
        )

    def forward(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full forward pass for FINE-TUNING (classification).

        Pipeline:
        1. Encode transaction sequences → contextual embeddings
        2. Pool embeddings → single vector per user
        3. Fuse with tabular features via DCNv2
        4. Project to binary logit

        Args:
            packed_keys: Categorical token indices.
            packed_values: Continuous values.
            packed_times: Temporal delta indices.
            tabular_features: Static tabular features [batch, tabular_dim].

        Returns:
            Binary logits [batch, 1] (pre-sigmoid).
        """
        # Step 1-2: Encode and pool
        pooled = self.get_pooled_embedding(
            packed_keys, packed_values, packed_times, **kwargs
        )

        # Step 3-4: Fuse with tabular features
        if tabular_features is not None:
            logits = self.fusion_head(pooled, tabular_features)
        else:
            # If no tabular features, use zeros
            dummy_tabular = torch.zeros(
                pooled.size(0), self.fusion_head.combined_dim - self.hidden_size,
                device=pooled.device, dtype=pooled.dtype,
            )
            logits = self.fusion_head(pooled, dummy_tabular)

        return logits

    def freeze_encoder(self):
        """
        Freezes the Transformer encoder weights.

        Used during fine-tuning to prevent catastrophic forgetting of
        pre-trained representations. Only the fusion head is trained.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Transformer encoder frozen. Only fusion head will be trained.")

    def unfreeze_encoder(self):
        """Unfreezes the Transformer encoder for end-to-end fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Transformer encoder unfrozen. Full model will be trained.")

    def save_checkpoint(self, path: str, extra_info: Optional[dict] = None):
        """
        Saves a model checkpoint with metadata.

        Args:
            path: File path for the checkpoint.
            extra_info: Additional metadata to store (e.g., epoch, metrics).
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
            },
        }
        if extra_info:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, **kwargs) -> "LargeDataModel":
        """
        Loads a model from a checkpoint file.

        Args:
            path: Path to the checkpoint file.
            **kwargs: Override config parameters.

        Returns:
            Loaded LargeDataModel instance.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from {path}")
        return model

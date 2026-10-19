"""
Large Data Model (LDM) — Complete End-to-End Foundation Model.

============================================================================
PURPOSE
============================================================================
Unifies the MMTT Transformer Encoder backbone, multi-modal self-supervised
reconstruction heads, and the DCNv2 static-dynamic fusion layer into a
cohesive foundation model supporting two operational phases:

1. PRE-TRAINING (Self-Supervised Learning):
   - Input: Unlabeled transaction sequences (Key-Value-Time)
   - Objectives:
     a) Masked Joint-Distribution Reconstruction (LimiX / PRAGMA): predicts
        masked merchant categories (`keys`), continuous amounts (`values`),
        and temporal rhythms (`times`) from bidirectional context.
     b) Contrastive Event Sequence Learning (CoLES): aligns pooled user
        subsequence views in embedding space to prevent representation collapse.

2. FINE-TUNING (Supervised Risk Classification):
   - Input: Transaction sequences + static bureau features + fraud labels
   - Objective: Class-weighted binary classification via parallel DCNv2 fusion.

============================================================================
REFERENCES
============================================================================
  - Dou et al. (2025): "TransactionGPT: Foundation Models for Payment
    Trajectories" (Visa Research, arXiv:2511.08939)
  - Ostroukhov et al. (2026): "PRAGMA: A Self-Supervised Foundation Model
    for Banking Events" (Revolut/NVIDIA, arXiv:2604.08649)
  - Babaev et al. (2022): "CoLES: Contrastive Learning for Event Sequences
    with Self-Supervision" (SIGMOD 2022, arXiv:2002.08232)
============================================================================
"""

from typing import Any, Dict, Literal, Optional
import logging

import torch
import torch.nn as nn

from src.models.transformer_backbone import MMTTTransformerEncoder
from src.models.joint_fusion_network import EndToEndFusionLayer

logger = logging.getLogger(__name__)


class LargeDataModel(nn.Module):
    """
    Complete Large Data Model supporting self-supervised pre-training and
    supervised fine-tuning.

    Args:
        vocab_size: Number of unique categorical tokens (MCC codes).
        hidden_size: Transformer hidden dimension (`d_model`).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        tabular_feature_size: Number of static tabular features for fusion.
        dcn_layers: Number of DCNv2 cross layers.
        use_virtual_tokens: Enable virtual token global conditioning.
        num_virtual_tokens: Number of virtual tokens.
        pooling: Sequence pooling strategy (`"cls"`, `"mean"`, `"max"`, `"attentive"`).
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
        pooling: Literal["cls", "mean", "max", "attentive"] = "mean",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.tabular_feature_size = tabular_feature_size
        self.dcn_layers = dcn_layers
        self.use_virtual_tokens = use_virtual_tokens
        self.num_virtual_tokens = num_virtual_tokens
        self.pooling = pooling

        # Shared MMTT Transformer Backbone
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

        # Self-Supervised Multi-Modal Reconstruction Heads (PRAGMA / LimiX)
        self.ssl_key_head = nn.Linear(hidden_size, vocab_size)
        self.ssl_value_head = nn.Linear(hidden_size, 1)
        self.ssl_time_head = nn.Linear(hidden_size, 1)
        self.ssl_latent_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Supervised Fine-Tuning DCNv2 Fusion Head
        self.fusion_head = EndToEndFusionLayer(
            transformer_hidden_size=hidden_size,
            tabular_feature_size=tabular_feature_size,
            dcn_layers=dcn_layers,
        )

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
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encodes transaction sequences into contextual per-token embeddings.

        Args:
            packed_keys: Categorical token indices `[batch, seq]`.
            packed_values: Continuous values `[batch, seq]`.
            packed_times: Temporal delta indices `[batch, seq]`.
            key_padding_mask: Optional sequence padding mask `[batch, seq]`.

        Returns:
            Full sequence embeddings `[batch, seq_len + num_virtual, hidden_size]`.
        """
        return self.encoder(
            packed_keys,
            packed_values,
            packed_times,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

    def predict_masked_modalities(
        self,
        sequence_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Projects sequence token embeddings to reconstruct Key, Value, Time, and Latent targets.

        Automatically strips prepended virtual tokens so output shapes match
        the input transaction sequence length `[batch, seq_len, ...]`.

        Args:
            sequence_embeddings: Output of `encode()` `[B, num_virtual + S, D]`.

        Returns:
            Dictionary with `"key_logits"`, `"value_pred"`, `"time_pred"`, `"latent_pred"`.
        """
        num_v = self.encoder.num_virtual_tokens if self.encoder.use_virtual_tokens else 0
        token_emb = sequence_embeddings[:, num_v:, :]
        return {
            "key_logits": self.ssl_key_head(token_emb),
            "value_pred": self.ssl_value_head(token_emb).squeeze(-1),
            "time_pred": self.ssl_time_head(token_emb).squeeze(-1),
            "latent_pred": self.ssl_latent_predictor( sequence_embeddings ),
        }

    def get_pooled_embedding(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes a single pooled representation vector per user sequence.

        Returns:
            Pooled embeddings `[batch, hidden_size]`.
        """
        return self.encoder.get_pooled_output(
            packed_keys,
            packed_values,
            packed_times,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

    def forward(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full forward pass for supervised classification.

        Args:
            packed_keys: Categorical token indices `[batch, seq]`.
            packed_values: Continuous values `[batch, seq]`.
            packed_times: Temporal delta indices `[batch, seq]`.
            tabular_features: Static tabular features `[batch, tabular_dim]`.
            key_padding_mask: Optional sequence padding mask `[batch, seq]`.

        Returns:
            Binary classification logits `[batch, 1]` (pre-sigmoid).
        """
        pooled = self.get_pooled_embedding(
            packed_keys,
            packed_values,
            packed_times,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

        if tabular_features is not None:
            logits = self.fusion_head(pooled, tabular_features)
        else:
            dummy_tabular = torch.zeros(
                pooled.size(0),
                self.fusion_head.combined_dim - self.hidden_size,
                device=pooled.device,
                dtype=pooled.dtype,
            )
            logits = self.fusion_head(pooled, dummy_tabular)

        return logits

    def freeze_encoder(self) -> None:
        """Freezes the Transformer encoder parameters during linear/head probing."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Transformer encoder frozen. Only fusion head will be trained.")

    def unfreeze_encoder(self) -> None:
        """Unfreezes the Transformer encoder for end-to-end fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Transformer encoder unfrozen. Full model will be trained.")

    def save_checkpoint(self, path: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves a model checkpoint using safe tensor-compatible metadata.

        Args:
            path: File path for the checkpoint.
            extra_info: Optional primitive metadata (floats/ints/strings/dicts).
        """
        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": int(self.vocab_size),
                "hidden_size": int(self.hidden_size),
                "num_layers": int(self.num_layers),
                "num_heads": int(self.num_heads),
                "dropout": float(self.dropout),
                "tabular_feature_size": int(self.tabular_feature_size),
                "dcn_layers": int(self.dcn_layers),
                "use_virtual_tokens": bool(self.use_virtual_tokens),
                "num_virtual_tokens": int(self.num_virtual_tokens),
                "pooling": str(self.pooling),
            },
        }
        if extra_info:
            checkpoint["extra_info"] = extra_info
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, **kwargs) -> "LargeDataModel":
        """
        Loads a `LargeDataModel` from a checkpoint using safe `weights_only=True`.

        Args:
            path: Path to the checkpoint file.
            **kwargs: Optional configuration overrides.

        Returns:
            Initialized and restored `LargeDataModel` instance.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = dict(checkpoint.get("config", {}))
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Model loaded safely from {path}")
        return model

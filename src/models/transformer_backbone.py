"""
MMTT (Key-Value-Time) Transformer Backbone for Large Data Models.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
This is the CORE neural network of the LDM. It processes sequences of
financial transactions (tokenized as Key-Value-Time triplets) and produces
dense contextual embeddings that capture behavioral patterns.

Architecture overview:
  Input (KVT tokens) → Additive Fusion → [Virtual Tokens] → Transformer → Output

The architecture draws from two state-of-the-art approaches:

1. TransactionGPT (Visa, 2025) — "3D-Transformer":
   - Processes three modalities (Feature, Metadata, Temporal) hierarchically
   - Uses Virtual Token Layer for modality fusion
   - Our simplified version uses additive fusion instead of 3 separate
     Transformer branches (keeps computational cost manageable)

2. PRAGMA (Revolut/NVIDIA, 2025):
   - Key-Value-Time tokenization preserves financial data semantics
   - Two-branch encoder (profile + events)
   - Masked modeling objective for self-supervised learning

============================================================================
KEY ARCHITECTURAL DECISIONS
============================================================================
1. ADDITIVE FUSION (not concatenation):
   We ADD the three embeddings (key + value + time) instead of concatenating.
   This keeps the hidden dimension constant regardless of the number of
   modalities, avoiding quadratic attention cost growth.

2. PRE-NORM (not post-norm):
   We use norm_first=True in TransformerEncoderLayer. This places LayerNorm
   BEFORE the attention and FFN blocks (Pre-LN), which provides more stable
   gradients during training and is the standard in modern Transformers
   (GPT-2+, LLaMA, etc.).

3. VIRTUAL TOKENS (continuous prompts):
   Instead of feeding metadata as additional sequence tokens (which increases
   attention cost quadratically), we prepend a small number of learnable
   "virtual tokens" that act as global context conditioners. This is
   conceptually similar to prompt tuning in NLP.

4. CONFIGURABLE POOLING:
   Downstream tasks need a single vector per user, not per-token embeddings.
   We support three pooling strategies:
   - "cls": Use the first (virtual) token's embedding (like BERT's [CLS])
   - "mean": Average all token embeddings (robust, works well empirically)
   - "max": Max-pool across tokens (captures strongest signals)

============================================================================
REFERENCES
============================================================================
  - Vaswani et al. (2017): "Attention Is All You Need"
  - TransactionGPT (Visa Research, 2025): 3D-Transformer architecture
  - PRAGMA (Revolut/NVIDIA, 2025): KVT tokenization & masked modeling
  - Su et al. (2021): RoFormer — Rotary Position Embedding (RoPE)
============================================================================
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Literal


class VirtualTokenLayer(nn.Module):
    """
    Virtual Token Layer — Continuous Prompt Conditioning.

    Instead of flattening all metadata (user profile, graph embeddings,
    static features) into the main sequence — which would increase the
    attention computation quadratically — we represent them as a small
    number of LEARNABLE tokens prepended to the sequence.

    These virtual tokens act as "soft prompts" that globally condition
    the multi-head attention mechanism. During training, they learn to
    encode the most useful global context for the downstream task.

    Analogy:
        Think of virtual tokens as the "briefing" a fraud analyst reads
        BEFORE reviewing individual transactions. They provide context
        about the user's overall profile without examining each transaction.

    Mathematical formulation:
        input_sequence = [v_1, v_2, ..., v_k, t_1, t_2, ..., t_n]
        where v_i are virtual tokens and t_j are transaction tokens.

    Args:
        hidden_size: Dimension of the hidden representation.
        num_virtual_tokens: Number of virtual tokens to prepend.
    """

    def __init__(self, hidden_size: int, num_virtual_tokens: int = 2):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens

        # Learnable virtual token embeddings
        # Shape: [1, num_virtual_tokens, hidden_size]
        # Initialized with small random values (scaled by hidden_size)
        self.virtual_tokens = nn.Parameter(
            torch.randn(1, num_virtual_tokens, hidden_size) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Expands virtual tokens to match the batch size.

        Args:
            batch_size: Number of samples in the current batch.

        Returns:
            Tensor of shape [batch_size, num_virtual_tokens, hidden_size]
        """
        return self.virtual_tokens.expand(batch_size, -1, -1)


class MMTTTransformerEncoder(nn.Module):
    """
    MMTT Transformer Encoder — The backbone of the Large Data Model.

    Processes 1D packed sequences of financial transactions and produces
    contextual embeddings that capture complex behavioral patterns.

    Architecture:
        1. Embedding layers for each modality (Key, Value, Time)
        2. Additive fusion of the three modalities
        3. Layer normalization for training stability
        4. Optional virtual token prepending
        5. Standard Transformer Encoder with pre-norm
        6. Configurable pooling for downstream tasks

    Supports two attention modes:
        - Causal (unidirectional): For Next Token Prediction (autoregressive)
        - Bidirectional: For Masked Joint-Distribution (like BERT)

    Args:
        vocab_size: Number of unique categorical tokens (MCC codes).
        hidden_size: Dimension of the hidden representation (d_model).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability for regularization.
        use_virtual_tokens: Whether to prepend virtual tokens.
        num_virtual_tokens: Number of virtual tokens (if enabled).
        pooling: Pooling strategy — "cls", "mean", or "max".

    Example:
        >>> encoder = MMTTTransformerEncoder(
        ...     vocab_size=20, hidden_size=128, num_layers=4, num_heads=4
        ... )
        >>> keys = torch.randint(0, 20, (2, 50))     # [batch, seq_len]
        >>> values = torch.randn(2, 50)               # [batch, seq_len]
        >>> times = torch.randint(0, 1000, (2, 50))   # [batch, seq_len]
        >>> out = encoder(keys, values, times)
        >>> print(out.shape)  # [2, 52, 128] (with 2 virtual tokens)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        use_virtual_tokens: bool = True,
        num_virtual_tokens: int = 2,
        pooling: Literal["cls", "mean", "max"] = "mean",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.pooling = pooling
        self.use_virtual_tokens = use_virtual_tokens

        # =====================================================================
        # MMTT Embedding Layers
        # =====================================================================
        # Each modality gets its own embedding/projection layer:
        #
        # KEY embedding (categorical):
        #   Maps discrete MCC codes to dense vectors.
        #   Analogous to word embeddings in NLP, but for financial categories.
        #
        # VALUE projection (continuous):
        #   Projects scalar amounts into the hidden dimension.
        #   Uses a linear layer (not embedding) because amounts are continuous.
        #
        # TIME embedding (temporal):
        #   Maps temporal delta bins to dense vectors.
        #   Encodes the rhythm of user behavior (frequent vs sporadic).
        #   We use 10,000 bins covering deltas from 0 to ~417 days.
        # =====================================================================
        self.key_emb = nn.Embedding(vocab_size, hidden_size)
        self.value_proj = nn.Linear(1, hidden_size)
        self.time_emb = nn.Embedding(10000, hidden_size)

        # LayerNorm to stabilize the additive MMTT fusion
        self.mmtt_norm = nn.LayerNorm(hidden_size)

        # Dropout for regularization after embedding fusion
        self.embed_dropout = nn.Dropout(dropout)

        # =====================================================================
        # Virtual Token Layer
        # =====================================================================
        if use_virtual_tokens:
            self.virtual_token_layer = VirtualTokenLayer(
                hidden_size, num_virtual_tokens
            )
            self.num_virtual_tokens = num_virtual_tokens
        else:
            self.num_virtual_tokens = 0

        # =====================================================================
        # Transformer Encoder
        # =====================================================================
        # We use PyTorch's native TransformerEncoder with pre-norm (norm_first).
        #
        # Each layer contains:
        #   1. Multi-Head Self-Attention (captures inter-token dependencies)
        #   2. Position-wise Feed-Forward Network (non-linear transformation)
        #   3. Residual connections around both sub-layers
        #   4. LayerNorm before each sub-layer (pre-norm variant)
        #
        # The feed-forward dimension is 4× hidden_size by convention
        # (from "Attention Is All You Need").
        # =====================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,    # Input shape: [batch, seq, features]
            norm_first=True,     # Pre-LN (more stable training)
            activation="gelu",   # GELU activation (smoother than ReLU)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # Compatibility with masks
        )

        # Final LayerNorm after Transformer (standard in modern architectures)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the MMTT Transformer Encoder.

        Processing steps:
        1. Embed each modality independently (Key, Value, Time)
        2. Fuse via element-wise addition (additive fusion)
        3. Normalize and apply dropout
        4. Prepend virtual tokens (if enabled)
        5. Generate appropriate attention mask (causal or bidirectional)
        6. Pass through the Transformer encoder
        7. Apply final normalization

        Args:
            packed_keys: Categorical indices [batch, seq] or [total_tokens]
            packed_values: Continuous values [batch, seq] or [total_tokens]
            packed_times: Temporal indices [batch, seq] or [total_tokens]
            cu_seqlens: Cumulative sequence lengths for packed sequences
            max_seqlen: Maximum sequence length in this batch
            is_causal: If True → unidirectional mask (for NTP)
                       If False → bidirectional (for masked modeling)

        Returns:
            Contextual embeddings. Shape depends on input:
            - If batched: [batch, seq + num_virtual_tokens, hidden_size]
            - If packed 1D: [1, total_tokens + num_virtual_tokens, hidden_size]
        """
        # =====================================================================
        # Step 1-2: MMTT Additive Fusion
        # =====================================================================
        # Each modality is independently projected to hidden_size, then
        # ADDED together. This preserves the hidden dimension while
        # allowing the Transformer to attend across all modalities jointly.
        # =====================================================================
        k_emb = self.key_emb(packed_keys)
        v_emb = self.value_proj(packed_values.unsqueeze(-1))
        t_emb = self.time_emb(packed_times)

        # Additive fusion: the simplest and most efficient multi-modal fusion
        x = k_emb + v_emb + t_emb

        # Step 3: Normalize and dropout
        x = self.mmtt_norm(x)
        x = self.embed_dropout(x)

        # =====================================================================
        # Handle packed 1D sequences
        # =====================================================================
        # If data arrives as a 1D packed tensor (from dynamic_sequence_packing),
        # we unsqueeze to create a batch dimension. In production with
        # Flash Attention, cu_seqlens would be passed directly to CUDA kernels.
        # =====================================================================
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, S, D = x.shape

        # Step 4: Prepend Virtual Tokens
        if self.use_virtual_tokens:
            v_tokens = self.virtual_token_layer(B)
            x = torch.cat([v_tokens, x], dim=1)
            S += self.num_virtual_tokens

        # Step 5: Attention mask generation
        mask = None
        if is_causal:
            # Upper triangular mask prevents attending to future tokens.
            # Essential for autoregressive objectives (Next Token Prediction).
            mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=x.device, dtype=x.dtype
            )

        # Step 6-7: Transformer forward + final norm
        out = self.transformer(x, mask=mask, is_causal=is_causal)
        out = self.final_norm(out)

        return out

    def get_pooled_output(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass + pooling to get a single vector per user.

        This is the main interface for downstream tasks (classification,
        regression). The pooling strategy is configured in __init__.

        Pooling strategies:
        - "cls": First token's embedding (virtual token acts as [CLS])
        - "mean": Average of all token embeddings (most robust)
        - "max": Maximum across the sequence dimension

        Returns:
            Tensor of shape [batch, hidden_size]
        """
        # Full forward pass
        out = self.forward(packed_keys, packed_values, packed_times, **kwargs)

        # Apply pooling to reduce sequence dimension
        if self.pooling == "cls":
            # First token is the virtual/CLS token
            return out[:, 0, :]
        elif self.pooling == "mean":
            # Average over the sequence dimension
            return out.mean(dim=1)
        elif self.pooling == "max":
            # Max-pool over the sequence dimension
            return out.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

"""
MMTT (Key-Value-Time) Transformer Backbone for Large Data Models.

============================================================================
PURPOSE
============================================================================
Core neural network backbone of the Large Data Model (LDM). It processes
sequences of financial transactions (tokenized as Key-Value-Time triplets)
and produces contextual embeddings that capture sequential and behavioral
patterns.

Architecture overview:
  Input (KVT tokens) -> Periodic Numerical & Modality Embeddings
                     -> Additive Fusion + Positional Encoding
                     -> [Virtual Tokens] -> Pre-Norm Transformer -> Output

The architecture synthesizes four peer-reviewed approaches:

1. TransactionGPT (Dou et al., Visa Research, 2025, arXiv:2511.08939):
   - Processes three modalities (Category, Value, Temporal) jointly
   - Uses a Virtual Token Layer for global context conditioning

2. PRAGMA (Ostroukhov et al., Revolut/NVIDIA, 2026, arXiv:2604.08649):
   - Key-Value-Time (KVT) tokenization preserving financial semantics
   - Bidirectional masked modeling for self-supervised representation learning

3. Periodic Numerical Embeddings (Gorishniy et al., NeurIPS 2022, arXiv:2203.05556):
   - Maps continuous scalar transaction amounts through trainable periodic
     sinusoids followed by linear and non-linear projections (PLR),
     overcoming spectral bias on heavy-tailed financial distributions.

4. Sequence Positional Encoding (Vaswani et al., 2017 / Su et al., 2024):
   - Disambiguates sequence ordering independently of inter-transaction
     time deltas so permutation-equivariant self-attention preserves
     trajectory order.

============================================================================
KEY ARCHITECTURAL DECISIONS
============================================================================
1. ADDITIVE FUSION WITH PERIODIC NUMERICAL EMBEDDINGS:
   We sum the three modality embeddings (Key + Periodic Value + Time Delta)
   plus sinusoidal sequence position encodings. This keeps the hidden
   dimension constant regardless of modality count, avoiding quadratic
   attention growth across modalities.

2. PRE-NORM TRANSFORMER (norm_first=True):
   Places LayerNorm before attention and feed-forward sublayers (Pre-LN),
   stabilizing gradient flow during deep self-supervised pre-training.

3. VIRTUAL TOKENS (Continuous Prompts):
   Prepends a small number of learnable virtual tokens that globally
   condition multi-head self-attention without inflating sequence length.

4. CONFIGURABLE MASK-AWARE POOLING:
   Supports four pooling strategies that strictly ignore padded positions:
   - "cls": First (virtual) token representation
   - "mean": Masked average across valid sequence tokens
   - "max": Masked max-pooling across valid sequence tokens
   - "attentive": Learnable attention-weighted pooling over valid tokens
============================================================================
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicNumericalEmbedding(nn.Module):
    """
    Periodic + Linear + Activation Numerical Embedding (PLR).

    Implements the periodic numerical embedding architecture introduced by
    Gorishniy, Rubachev, & Babenko (NeurIPS 2022, "On Embeddings for
    Numerical Features in Tabular Deep Learning", arXiv:2203.05556).

    Given a continuous scalar input `x`, computes:
        p(x) = [x, sin(2*pi * c * x), cos(2*pi * c * x)]
        emb(x) = Linear(GELU(Linear(p(x))))

    Including the raw scalar `x` alongside trainable frequencies `c`
    preserves monotonic magnitude information while enabling the network
    to resolve fine-grained numerical thresholds.

    Args:
        hidden_size: Output embedding dimension.
        num_frequencies: Number of trainable sinusoidal frequency pairs.
        frequency_init_scale: Standard deviation for frequency initialization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_frequencies: int = 16,
        frequency_init_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_frequencies = num_frequencies

        self.frequencies = nn.Parameter(
            torch.randn(num_frequencies) * frequency_init_scale
        )
        in_features = 2 * num_frequencies + 1
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects continuous values into dense embeddings.

        Args:
            x: Tensor of shape `[..., 1]` or `[...]`.

        Returns:
            Tensor of shape `[..., hidden_size]`.
        """
        if x.dim() <= 2 and (torch.jit.is_tracing() or x.dim() == 0 or x.shape[-1] != 1):
            x_scalar = x.unsqueeze(-1)
        else:
            x_scalar = x

        angles = 2.0 * math.pi * x_scalar * self.frequencies
        periodic = torch.cat([x_scalar, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.linear_2(self.activation(self.linear_1(periodic)))


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal sequence positional encoding for ordered event trajectories.

    Provides sequence-order disambiguation along the event index axis
    `[0, 1, ..., S-1]`, complementing inter-transaction `time_delta` embeddings.

    Args:
        hidden_size: Embedding dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, hidden_size: int, max_len: int = 4096):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, hidden_size))
        )
        pe = torch.zeros(1, max_len, hidden_size, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if hidden_size > 1:
            pe[0, :, 1::2] = torch.cos(position * div_term[: hidden_size // 2])
        self.register_buffer("pe", pe, persistent=False)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns scaled positional encodings of shape `[1, seq_len, hidden_size]`."""
        if torch.jit.is_tracing() or seq_len <= self.max_len:
            return self.scale * self.pe[:, :seq_len, :]
        # Dynamically compute if sequence exceeds precomputed buffer
        device = self.pe.device
        dtype = self.pe.dtype
        hidden_size = self.pe.size(2)
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, hidden_size))
        )
        pe = torch.zeros(1, seq_len, hidden_size, device=device, dtype=dtype)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if hidden_size > 1:
            pe[0, :, 1::2] = torch.cos(position * div_term[: hidden_size // 2])
        return self.scale * pe


class VirtualTokenLayer(nn.Module):
    """
    Virtual Token Layer — Continuous Prompt Conditioning.

    Prepends a small number of learnable virtual tokens to each user's
    transaction sequence to act as global context conditioners in
    multi-head self-attention:
        input_sequence = [v_1, v_2, ..., v_k, t_1, t_2, ..., t_n]

    Args:
        hidden_size: Dimension of the hidden representation.
        num_virtual_tokens: Number of virtual tokens to prepend.
    """

    def __init__(self, hidden_size: int, num_virtual_tokens: int = 2):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.virtual_tokens = nn.Parameter(
            torch.randn(1, num_virtual_tokens, hidden_size) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Expands virtual tokens to match the batch size.

        Args:
            batch_size: Number of samples in the current batch.

        Returns:
            Tensor of shape `[batch_size, num_virtual_tokens, hidden_size]`.
        """
        return self.virtual_tokens.expand(batch_size, -1, -1)


class MMTTTransformerEncoder(nn.Module):
    """
    MMTT Transformer Encoder — The backbone of the Large Data Model.

    Processes packed sequences of financial transactions and produces
    contextual embeddings that capture behavioral patterns across
    merchant categories (Key), continuous amounts (Value), and temporal
    cadence (Time).

    Args:
        vocab_size: Number of unique categorical tokens (MCC codes).
        hidden_size: Dimension of the hidden representation (`d_model`).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability for regularization.
        use_virtual_tokens: Whether to prepend virtual tokens.
        num_virtual_tokens: Number of virtual tokens (if enabled).
        pooling: Pooling strategy — `"cls"`, `"mean"`, `"max"`, or `"attentive"`.
        num_frequencies: Number of periodic frequencies for numerical embedding.
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
        pooling: Literal["cls", "mean", "max", "attentive"] = "mean",
        num_frequencies: int = 16,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.use_virtual_tokens = use_virtual_tokens

        # Modality-specific embedding layers
        self.key_emb = nn.Embedding(vocab_size, hidden_size)
        self.value_proj = PeriodicNumericalEmbedding(
            hidden_size=hidden_size,
            num_frequencies=num_frequencies,
        )
        self.time_emb = nn.Embedding(10000, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size=hidden_size)

        # LayerNorm and dropout stabilizing additive MMTT fusion
        self.mmtt_norm = nn.LayerNorm(hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        # Optional Virtual Token Layer
        if use_virtual_tokens:
            self.virtual_token_layer = VirtualTokenLayer(
                hidden_size, num_virtual_tokens
            )
            self.num_virtual_tokens = num_virtual_tokens
        else:
            self.num_virtual_tokens = 0

        # Pre-Norm Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.final_norm = nn.LayerNorm(hidden_size)

        # Learnable query vector for attentive sequence pooling
        self.attention_pool_query = nn.Linear(hidden_size, 1)

    def forward(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        is_causal: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the MMTT Transformer Encoder.

        Args:
            packed_keys: Categorical indices `[batch, seq]` or `[total_tokens]`.
            packed_values: Continuous values `[batch, seq]` or `[total_tokens]`.
            packed_times: Temporal indices `[batch, seq]` or `[total_tokens]`.
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            max_seqlen: Maximum sequence length in the current batch.
            is_causal: If True, applies a causal upper-triangular mask.
            key_padding_mask: Boolean mask `[batch, seq]` where True denotes padding.

        Returns:
            Contextual embeddings of shape `[batch, seq + num_virtual_tokens, hidden_size]`.
        """
        clamped_keys = packed_keys.clamp(min=0, max=self.vocab_size - 1)
        clamped_times = packed_times.clamp(min=0, max=9999)

        k_emb = self.key_emb(clamped_keys)
        v_emb = self.value_proj(packed_values.unsqueeze(-1))
        t_emb = self.time_emb(clamped_times)

        x = k_emb + v_emb + t_emb

        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, S, _ = x.shape

        # Add sequence positional encoding along the transaction index axis
        x = x + self.pos_encoding(S)

        x = self.mmtt_norm(x)
        x = self.embed_dropout(x)

        if self.use_virtual_tokens:
            v_tokens = self.virtual_token_layer(B)
            x = torch.cat([v_tokens, x], dim=1)
            S += self.num_virtual_tokens

        mask = None
        if is_causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=x.device, dtype=x.dtype
            )

        prepared_padding_mask = key_padding_mask
        if self.use_virtual_tokens and key_padding_mask is not None:
            v_mask = torch.zeros(
                B,
                self.num_virtual_tokens,
                dtype=torch.bool,
                device=key_padding_mask.device,
            )
            prepared_padding_mask = torch.cat([v_mask, key_padding_mask], dim=1)

        out = self.transformer(
            x,
            mask=mask,
            src_key_padding_mask=prepared_padding_mask,
            is_causal=is_causal,
        )
        out = self.final_norm(out)

        return out

    def _build_full_padding_mask(
        self,
        batch_size: int,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Prepends False mask entries for virtual tokens when `key_padding_mask` is provided."""
        if key_padding_mask is None:
            return None
        if self.use_virtual_tokens:
            v_mask = torch.zeros(
                batch_size,
                self.num_virtual_tokens,
                dtype=torch.bool,
                device=key_padding_mask.device,
            )
            return torch.cat([v_mask, key_padding_mask], dim=1)
        return key_padding_mask

    def get_pooled_output(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass + mask-aware sequence pooling to produce one vector per user.

        Args:
            packed_keys: Categorical token indices `[batch, seq]`.
            packed_values: Continuous transaction values `[batch, seq]`.
            packed_times: Temporal delta indices `[batch, seq]`.
            key_padding_mask: Optional boolean padding mask `[batch, seq]`.

        Returns:
            Pooled sequence representation of shape `[batch, hidden_size]`.
        """
        out = self.forward(
            packed_keys,
            packed_values,
            packed_times,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

        B = out.shape[0]
        full_mask = self._build_full_padding_mask(B, key_padding_mask)

        if self.pooling == "cls":
            return out[:, 0, :]
        elif self.pooling == "mean":
            if full_mask is not None:
                valid_mask = (~full_mask).unsqueeze(-1).float()
                sum_embeddings = (out * valid_mask).sum(dim=1)
                num_valid = valid_mask.sum(dim=1).clamp(min=1.0)
                return sum_embeddings / num_valid
            return out.mean(dim=1)
        elif self.pooling == "max":
            if full_mask is not None:
                valid_mask = (~full_mask).unsqueeze(-1)
                masked_out = out.masked_fill(~valid_mask, -1e9)
                return masked_out.max(dim=1).values
            return out.max(dim=1).values
        elif self.pooling == "attentive":
            scores = self.attention_pool_query(out).squeeze(-1)  # [B, S_total]
            if full_mask is not None:
                scores = scores.masked_fill(full_mask, -1e9)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            return (out * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

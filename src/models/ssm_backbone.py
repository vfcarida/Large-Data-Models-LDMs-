"""
Continuous-Time Selective State-Space (Mamba / Liquid-S4) & Hybrid Sequence Backbone.

============================================================================
PURPOSE
============================================================================
Provides linear-complexity O(S) sequence modeling for ultra-long lifetime
customer transaction histories (S > 10,000 events) while accounting for
irregular, continuous-time inter-event intervals (Delta t).

Synthesizes two foundational state-space architectures:
  1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
     (Gu & Dao, COLM 2024, arXiv:2312.00752):
     - Input-dependent selection mechanism where step size Delta_t, input
       projection B_t, and output projection C_t are functions of the current
       transaction token representation x_t.
  2. Liquid Structural State-Space Models (Hasani et al., ICLR 2023,
     arXiv:2209.12951):
     - Continuous-time Zero-Order Hold (ZOH) discretization explicitly
       modulated by empirical inter-arrival times tau_t = log(1 + delta_t).

Key properties:
  - O(S) time and memory scaling in sequence length S.
  - Strict padding mask invariance: padded tokens freeze the recurrent
    hidden state (h_t = h_{t-1}) and emit zero contribution to downstream
    pooling or attention layers.
============================================================================
"""

from typing import Literal, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer_backbone import (
    PeriodicNumericalEmbedding,
    SinusoidalPositionalEncoding,
    VirtualTokenLayer,
)


class SelectiveSSMBlock(nn.Module):
    """
    Continuous-Time Selective State-Space Block (Mamba / Liquid-S4 Hybrid).

    Given an input sequence `x` of shape `[B, S, D]` and optional inter-event
    time deltas `time_deltas` of shape `[B, S]`, computes selective state-space
    recurrence with Zero-Order Hold (ZOH) discretization:
        Delta_t = softplus(W_delta(u_t) + alpha_time * log(1 + time_delta_t))
        A_bar_t = exp(-Delta_t * exp(A_log))
        B_bar_t = Delta_t * B_t
        h_t     = A_bar_t * h_{t-1} + B_bar_t * u_t
        y_t     = (h_t @ C_t) + D * u_t

    When `padding_mask[:, t]` is `True`, sets `A_bar_t = 1` and `B_bar_t = 0`
    so padded positions neither alter the recurrent state nor leak into valid
    tokens.

    Args:
        d_model: Model hidden dimension (`D`).
        d_state: Latent state-space dimension (`N`, default: 16).
        expand_factor: Inner projection expansion factor (default: 2).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Depthwise 1D causal-style local convolution (kernel_size=3)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
        )

        # Selective projections for Delta, B, and C
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # Continuous-time interval modulation weight
        self.time_delta_scale = nn.Parameter(torch.full((1, 1, self.d_inner), 0.1))

        # S4D / HiPPO-inspired log-diagonal state matrix initialization
        arange_state = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A_init = arange_state.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # Skip connection parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward channel mixer (Pre-Norm)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Executes selective state-space recurrence over `[B, S, D]`.

        Args:
            x: Input sequence tensor `[B, S, D]`.
            time_deltas: Optional continuous or integer time deltas `[B, S]`.
            padding_mask: Optional boolean mask `[B, S]` (`True` = padded position).

        Returns:
            Output sequence tensor `[B, S, D]` with strict mask invariance.
        """
        B, S, _ = x.shape
        residual = x
        x_norm = self.norm(x)

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).type_as(x_norm)
            x_norm = x_norm * valid_mask
        else:
            valid_mask = None

        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)  # Each [B, S, d_inner]

        # Depthwise local context mixing masked at padded positions
        u_conv = self.conv1d(u.transpose(1, 2)).transpose(1, 2)
        if valid_mask is not None:
            u_conv = u_conv * valid_mask
        u = F.silu(u_conv)

        # Selective parameters
        delta_raw = self.delta_proj(u)
        if time_deltas is not None:
            tau = torch.log1p(time_deltas.float().clamp(min=0.0)).unsqueeze(-1)
            delta_raw = delta_raw + self.time_delta_scale * tau

        # Clamp delta in (1e-4, 5.0) for numerical stability
        delta = F.softplus(delta_raw).clamp(min=1e-4, max=5.0)  # [B, S, d_inner]
        if valid_mask is not None:
            delta = delta * valid_mask

        B_sel = self.B_proj(u)  # [B, S, d_state]
        C_sel = self.C_proj(u)  # [B, S, d_state]

        # Diagonal negative state matrix A: [d_inner, d_state]
        A = -torch.exp(self.A_log.clamp(min=-5.0, max=4.0))

        # Vectorized ZOH discretization parameters across time
        # delta: [B, S, d_inner, 1], A: [1, 1, d_inner, d_state]
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # B_bar_u: [B, S, d_inner, d_state]
        B_bar_u = (delta * u).unsqueeze(-1) * B_sel.unsqueeze(2)

        if padding_mask is not None:
            pad_4d = padding_mask.unsqueeze(-1).unsqueeze(-1)
            A_bar = torch.where(pad_4d, torch.ones_like(A_bar), A_bar)
            B_bar_u = torch.where(pad_4d, torch.zeros_like(B_bar_u), B_bar_u)

        # Linear O(S) scan over sequence steps
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            h = A_bar[:, t] * h + B_bar_u[:, t]
            y_t = (h * C_sel[:, t].unsqueeze(1)).sum(dim=-1) + self.D * u[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, S, d_inner]
        y = y * F.silu(z)
        if valid_mask is not None:
            y = y * valid_mask

        out = residual + self.dropout(self.out_proj(y))
        out = out + self.ffn(self.ffn_norm(out))
        if valid_mask is not None:
            out = out * valid_mask

        return out


class HybridSSMTransformerEncoder(nn.Module):
    """
    Hybrid Selective State-Space (`Mamba` / `Liquid-S4`) & Transformer Backbone.

    Supports both pure linear-complexity `ssm` mode ($O(S)$ memory and compute)
    and interleaved `hybrid_ssm` mode (Jamba-style alternating `SelectiveSSMBlock`
    and `nn.TransformerEncoderLayer` layers) while sharing the exact MMTT
    input/output contract (`forward`, `get_pooled_output`).

    Args:
        vocab_size: Number of unique categorical tokens (MCC codes).
        hidden_size: Hidden representation dimension (`d_model`).
        num_layers: Number of sequence modeling layers.
        num_heads: Number of attention heads in hybrid attention layers.
        d_state: Latent SSM state dimension (`N`).
        dropout: Dropout probability.
        use_virtual_tokens: Whether to prepend learnable virtual tokens.
        num_virtual_tokens: Number of virtual tokens.
        pooling: Sequence pooling strategy (`"cls"`, `"mean"`, `"max"`, `"attentive"`).
        mode: `"hybrid_ssm"` (alternating SSM + Attention) or `"ssm"` (pure O(S) SSM).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 4,
        num_heads: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
        use_virtual_tokens: bool = True,
        num_virtual_tokens: int = 2,
        pooling: Literal["cls", "mean", "max", "attentive"] = "mean",
        mode: Literal["hybrid_ssm", "ssm"] = "hybrid_ssm",
        num_frequencies: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_state = d_state
        self.pooling = pooling
        self.use_virtual_tokens = use_virtual_tokens
        self.mode = mode

        self.key_emb = nn.Embedding(vocab_size, hidden_size)
        self.value_proj = PeriodicNumericalEmbedding(
            hidden_size=hidden_size,
            num_frequencies=num_frequencies,
        )
        self.time_emb = nn.Embedding(10000, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size=hidden_size)

        self.mmtt_norm = nn.LayerNorm(hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        if use_virtual_tokens:
            self.virtual_token_layer = VirtualTokenLayer(hidden_size, num_virtual_tokens)
            self.num_virtual_tokens = num_virtual_tokens
        else:
            self.num_virtual_tokens = 0

        self.layers = nn.ModuleList()
        self.layer_types = []
        for idx in range(num_layers):
            if mode == "ssm" or (idx % 2 == 0):
                self.layers.append(
                    SelectiveSSMBlock(
                        d_model=hidden_size,
                        d_state=d_state,
                        expand_factor=2,
                        dropout=dropout,
                    )
                )
                self.layer_types.append("ssm")
            else:
                self.layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=hidden_size * 4,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    )
                )
                self.layer_types.append("attn")

        self.final_norm = nn.LayerNorm(hidden_size)
        self.attention_pool_query = nn.Linear(hidden_size, 1)

    def _build_full_padding_mask(
        self,
        batch_size: int,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
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
        Forward pass through the Hybrid Selective SSM / Transformer Encoder.
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
        x = x + self.pos_encoding(S)
        x = self.mmtt_norm(x)
        x = self.embed_dropout(x)

        full_times = clamped_times.float()
        if self.use_virtual_tokens:
            v_tokens = self.virtual_token_layer(B)
            x = torch.cat([v_tokens, x], dim=1)
            v_times = torch.zeros(
                B, self.num_virtual_tokens, device=x.device, dtype=full_times.dtype
            )
            full_times = torch.cat([v_times, full_times], dim=1)

        prepared_padding_mask = self._build_full_padding_mask(B, key_padding_mask)

        for layer, ltype in zip(self.layers, self.layer_types):
            if ltype == "ssm":
                x = layer(
                    x,
                    time_deltas=full_times,
                    padding_mask=prepared_padding_mask,
                )
            else:
                x = layer(
                    x,
                    src_key_padding_mask=prepared_padding_mask,
                )

        return self.final_norm(x)

    def get_pooled_output(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes mask-aware pooled sequence representations `[B, hidden_size]`.
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
            scores = self.attention_pool_query(out).squeeze(-1)
            if full_mask is not None:
                scores = scores.masked_fill(full_mask, -1e9)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            return (out * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

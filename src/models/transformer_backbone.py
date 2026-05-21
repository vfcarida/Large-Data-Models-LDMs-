"""
MMTT (Key-Value-Time) Transformer Backbone for Large Data Models.

Implements the 3D-Transformer architecture described in TransactionGPT research.
Designed to handle multi-modal transactional data efficiently.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

class VirtualTokenLayer(nn.Module):
    """
    Virtual Token Layer.
    
    As proposed in 3D-Transformer architectures (e.g., TransactionGPT), instead of 
    flattening all modalities (Key, Value, Time) into an overwhelmingly large sequence,
    this layer acts as a 'Continuous Prompt'.
    
    It allows the fusion of extrinsic modalities (e.g., global user graph embeddings)
    by prepending virtual tokens to the transactional sequence. This conditions the 
    multi-head attention mechanism globally and drastically reduces the quadratic 
    attention cost associated with processing metadata dimensions simultaneously.
    """
    def __init__(self, hidden_size: int, num_virtual_tokens: int = 2):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.virtual_tokens = nn.Parameter(torch.randn(1, num_virtual_tokens, hidden_size))

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Returns the virtual tokens expanded to match the batch size.
        
        Returns:
            torch.Tensor of shape [batch_size, num_virtual_tokens, hidden_size]
        """
        return self.virtual_tokens.expand(batch_size, -1, -1)


class MMTTTransformerEncoder(nn.Module):
    """
    MMTT Transformer Encoder designed for heterogeneous tensor processing.
    
    Features:
    - Processes 1D packed sequences (from dynamic_sequence_packing_collate_fn).
    - Supports both causal masks (unidirectional for Next Token Prediction) and 
      bidirectional masks (for Masked Joint-Distribution / ContextConditionalMaskedLoss).
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 num_heads: int, 
                 dropout: float = 0.1,
                 use_virtual_tokens: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # MMTT Embeddings (Key, Value, Time)
        self.key_emb = nn.Embedding(vocab_size, hidden_size)
        self.value_proj = nn.Linear(1, hidden_size)
        self.time_emb = nn.Embedding(10000, hidden_size)  # E.g., temporal bins/deltas
        
        # Layer Normalization to stabilize the MMTT fusion
        self.mmtt_norm = nn.LayerNorm(hidden_size)
        
        # Virtual Token Layer
        self.use_virtual_tokens = use_virtual_tokens
        if use_virtual_tokens:
            self.virtual_token_layer = VirtualTokenLayer(hidden_size)
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, 
                packed_keys: torch.Tensor, 
                packed_values: torch.Tensor, 
                packed_times: torch.Tensor,
                cu_seqlens: Optional[torch.Tensor] = None,
                max_seqlen: Optional[int] = None,
                is_causal: bool = True) -> torch.Tensor:
        """
        Forward pass. Optimized for 1D packed tensors to avoid padding.
        
        In a production environment leveraging FlashAttention, `cu_seqlens` is passed directly
        to the CUDA kernels. Here, we demonstrate the architectural flow.
        
        Args:
            packed_keys (Tensor): Categorical indices (Keys).
            packed_values (Tensor): Continuous scalar values (Values).
            packed_times (Tensor): Temporal indices/deltas (Times).
            cu_seqlens (Tensor, optional): Cumulative offsets for variable sequences.
            max_seqlen (int, optional): Maximum sequence length in this batch.
            is_causal (bool): If True, applies upper triangular mask (unidirectional).
                              If False, uses bidirectional attention (for Masked Joint-Distribution).
                              
        Returns:
            torch.Tensor: Contextual embeddings.
        """
        # MMTT Additive Fusion
        k_emb = self.key_emb(packed_keys)
        v_emb = self.value_proj(packed_values.unsqueeze(-1))
        t_emb = self.time_emb(packed_times)
        
        # Heterogeneous additive fusion
        x = k_emb + v_emb + t_emb
        x = self.mmtt_norm(x)
        
        # Handling packed sequences typically requires custom Flash Attention kernels (e.g., xformers).
        # For standard PyTorch architecture demonstration, if data arrives 1D packed:
        if x.dim() == 2:
            # Fake unsqueeze to simulate batch size 1 for standard processing
            x = x.unsqueeze(0)
            
        B, S, D = x.shape
        
        # Prepend Virtual Tokens if enabled
        if self.use_virtual_tokens:
            v_tokens = self.virtual_token_layer(B)
            x = torch.cat([v_tokens, x], dim=1)
            S += self.virtual_token_layer.num_virtual_tokens

        # Causal mask generation
        mask = None
        if is_causal:
            mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)

        # Forward through the Transformer Encoder
        out = self.transformer(x, mask=mask, is_causal=is_causal)
        
        return out

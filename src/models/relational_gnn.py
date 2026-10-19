"""
Relational Entity-Merchant Bipartite Graph Neural Network (RelBench / Temporal GNN).

============================================================================
PURPOSE
============================================================================
Implements bipartite relational message passing between user entities and
merchant category hubs inspired by RelBench (Robinson et al., NeurIPS 2024)
and PRAGMA (Ostroukhov et al., 2026).

While sequential Transformers capture a single entity's temporal trajectory,
coordinated financial fraud (e.g., mule rings, merchant-level skimming, or
synchronized liquidation sweeps across crypto/wire/gift-card hubs) exhibits
strong relational structure across the entity-merchant interaction graph.

`RelationalEntityGNN` constructs a mask-aware bipartite incidence matrix
between users in a batch and the discrete Merchant Category Code (MCC) nodes:
  1. Computes volume-weighted user-to-merchant interaction affinities
     strictly over non-padded transaction events (`~key_padding_mask`).
  2. Performs multi-layer Pre-Norm bipartite message passing:
       User Embeddings (H_U) -> Merchant Hub States (H_M) -> User Embeddings (H_U)
  3. Combines persistent learnable merchant hub prototypes with dynamic
     co-transaction neighborhood aggregation, ensuring full compatibility
     with both multi-user mini-batches and single-user (`batch_size=1`)
     real-time streaming inference.

============================================================================
REFERENCES
============================================================================
  - Robinson, J., Ranjan, R., Hu, W., et al. (2024): "RelBench: A Benchmark
    for Deep Learning on Relational Databases" (NeurIPS 2024, arXiv:2407.20060)
  - Ostroukhov, P., et al. (2026): "PRAGMA: A Self-Supervised Foundation
    Model for Banking Events" (arXiv:2604.08649)
============================================================================
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BipartiteGraphConvLayer(nn.Module):
    """
    Single Pre-Norm Bipartite Message-Passing Layer (User <-> Merchant Hub).

    Args:
        embed_dim: Dimensionality of user and merchant node representations.
        dropout: Dropout probability applied to propagated messages.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.merchant_norm = nn.LayerNorm(embed_dim)
        self.user_norm = nn.LayerNorm(embed_dim)
        self.u2m_linear = nn.Linear(embed_dim, embed_dim)
        self.m2u_linear = nn.Linear(embed_dim, embed_dim)
        self.self_linear = nn.Linear(embed_dim, embed_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        user_emb: torch.Tensor,
        merchant_emb: torch.Tensor,
        u2m_adj: torch.Tensor,
        m2u_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagates relational messages User -> Merchant -> User using Pre-Norm residuals.

        Args:
            user_emb: User representations `(B, D)`.
            merchant_emb: Merchant hub representations `(V, D)`.
            u2m_adj: Row-normalized User-to-Merchant adjacency `(V, B)`.
            m2u_adj: Row-normalized Merchant-to-User adjacency `(B, V)`.

        Returns:
            Updated `(user_emb, merchant_emb)` of shapes `(B, D)` and `(V, D)`.
        """
        u_norm = self.user_norm(user_emb)
        m_norm = self.merchant_norm(merchant_emb)

        # 1. User -> Merchant aggregation: (V, B) @ (B, D) -> (V, D)
        m_messages = torch.matmul(u2m_adj, self.u2m_linear(u_norm))
        updated_merchant = merchant_emb + self.residual_scale * self.dropout(
            F.gelu(m_messages)
        )

        # 2. Merchant -> User aggregation: (B, V) @ (V, D) -> (B, D)
        m_updated_norm = self.merchant_norm(updated_merchant)
        u_messages = torch.matmul(m2u_adj, self.m2u_linear(m_updated_norm))
        u_self = self.self_linear(u_norm)
        updated_user = user_emb + self.residual_scale * self.dropout(
            F.gelu(u_self + u_messages)
        )

        return updated_user, updated_merchant


class RelationalEntityGNN(nn.Module):
    """
    Multi-Layer Bipartite Relational GNN over User-Merchant Interaction Graphs.

    Args:
        vocab_size: Number of discrete Merchant Category / Event Key hubs.
        embed_dim: Dimension of user and merchant embeddings.
        num_layers: Number of bipartite message-passing hops (default: 2).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0:
            raise ValueError("vocab_size and embed_dim must be positive.")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Learnable structural prototypes for each Merchant Category Hub
        self.merchant_prototypes = nn.Parameter(
            torch.randn(vocab_size, embed_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                BipartiteGraphConvLayer(embed_dim=embed_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def build_bipartite_adjacency(
        self,
        event_keys: torch.Tensor,
        event_values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs row-normalized bipartite adjacency matrices strictly over
        non-padded transaction events.

        Args:
            event_keys: Categorical MCC indices `(B, S)`.
            event_values: Continuous transaction amounts `(B, S)`.
            key_padding_mask: Boolean mask `(B, S)` where `True` indicates padding.

        Returns:
            Tuple `(u2m_adj, m2u_adj)`:
              - `u2m_adj`: Normalized adjacency `(V, B)`
              - `m2u_adj`: Normalized adjacency `(B, V)`
        """
        batch_size, seq_len = event_keys.shape
        device = event_keys.device
        dtype = event_values.dtype

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).to(dtype)
        else:
            valid_mask = torch.ones((batch_size, seq_len), device=device, dtype=dtype)

        event_weights = valid_mask * (1.0 + F.softplus(event_values))

        clamped_keys = event_keys.clamp(0, self.vocab_size - 1)
        zeros_inc = torch.zeros(
            (batch_size, self.vocab_size), device=device, dtype=dtype
        )
        incidence = zeros_inc.scatter_add(dim=1, index=clamped_keys, src=event_weights)

        m2u_deg = incidence.sum(dim=1, keepdim=True).clamp(min=1e-6)
        m2u_adj = incidence / m2u_deg

        inc_t = incidence.transpose(0, 1)
        u2m_deg = inc_t.sum(dim=1, keepdim=True).clamp(min=1e-6)
        u2m_adj = inc_t / u2m_deg

        return u2m_adj, m2u_adj

    def forward(
        self,
        user_embeddings: torch.Tensor,
        event_keys: torch.Tensor,
        event_values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Executes bipartite relational message passing over the user-merchant graph.
        """
        u2m_adj, m2u_adj = self.build_bipartite_adjacency(
            event_keys=event_keys,
            event_values=event_values,
            key_padding_mask=key_padding_mask,
        )

        h_user = user_embeddings
        h_merchant = self.merchant_prototypes

        for layer in self.layers:
            h_user, h_merchant = layer(
                user_emb=h_user,
                merchant_emb=h_merchant,
                u2m_adj=u2m_adj,
                m2u_adj=m2u_adj,
            )

        return h_user

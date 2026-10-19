"""
Temporal Hypergraph Contagion & Multi-Hop Collusion Ring Detection (`HGNN`).

============================================================================
PURPOSE
============================================================================
While `RelationalEntityGNN` models pairwise bipartite interactions `(User <->
Merchant Category)`, organized fraud syndicates, synthetic identity rings, and
anti-money laundering (AML) mule networks operate through higher-order
hyperedges: groups of accounts exhibiting synchronized bursts across shared
merchant categories and narrow temporal velocity windows.

This module implements:
  1. `HypergraphConvLayer` (Feng et al., AAAI 2019, "Hypergraph Neural
     Networks", arXiv:1809.09401; Chien et al., ICLR 2022, "You are AllSet:
     A Multiset Function Framework for Hypergraph Neural Networks"):
     - Normalized two-stage message passing (Node -> Hyperedge -> Node):
           X_out = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Theta
       with Pre-Norm residual connections and learnable hyperedge weights `W_e`.
  2. `TemporalHypergraphContagionModule`:
     - Dynamically constructs a mask-aware incidence matrix `H` of shape
       `[B, num_hyperedges]` where each hyperedge represents a composite
       `(merchant_sector, temporal_burst_tier)` motif.
     - Strictly masks padded positions (`key_padding_mask`) so padded tokens
       never contribute to hyperedge incidence or contagion scores.
     - Outputs both hypergraph-enriched entity representations `[B, D]` and a
       calibrated `collusion_ring_score` `[B, 1]` in `(0, 1)`.
============================================================================
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConvLayer(nn.Module):
    """
    Spectral Hypergraph Convolution Layer with Learnable Hyperedge Weights.

    Implements:
        Y = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Theta
    followed by Pre-Norm gated residual projection.

    Args:
        embed_dim: Node feature dimension (`D`).
        num_hyperedges: Number of hyperedges (`E`).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_hyperedges: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hyperedges = num_hyperedges

        self.norm = nn.LayerNorm(embed_dim)
        self.theta = nn.Linear(embed_dim, embed_dim, bias=False)
        self.hyperedge_weights = nn.Parameter(torch.ones(num_hyperedges))
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        incidence_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes node-to-hyperedge-to-node message passing.

        Args:
            node_features: Tensor `[B, D]`.
            incidence_matrix: Non-negative incidence tensor `H` of shape `[B, E]`.

        Returns:
            Updated node representations `[B, D]`.
        """
        x = self.norm(node_features)
        x_proj = self.theta(x)  # [B, D]

        w_e = F.softplus(self.hyperedge_weights).unsqueeze(0)  # [1, E]
        H = incidence_matrix

        # Hyperedge degrees D_e: [1, E]
        d_e = H.sum(dim=0, keepdim=True).clamp(min=1e-6)
        # Weighted node degrees D_v: [B, 1]
        d_v = (H * w_e).sum(dim=1, keepdim=True).clamp(min=1e-6)
        d_v_inv_sqrt = torch.rsqrt(d_v)  # [B, 1]

        x_normed = x_proj * d_v_inv_sqrt  # [B, D]
        # Aggregate nodes into hyperedges: [E, D]
        hyperedge_emb = torch.matmul((H / d_e).transpose(0, 1), x_normed)
        # Apply hyperedge weights W_e: [E, D]
        hyperedge_emb = hyperedge_emb * w_e.transpose(0, 1)
        # Broadcast back to nodes: [B, D]
        node_agg = torch.matmul(H, hyperedge_emb) * d_v_inv_sqrt

        combined = torch.cat([x, node_agg], dim=-1)
        return node_features + self.out_proj(combined)


class TemporalHypergraphContagionModule(nn.Module):
    """
    Temporal Hypergraph Contagion & Collusion Ring Scoring Module.

    Constructs higher-order hyperedges over `(MCC sector, temporal velocity tier)`
    co-occurrence patterns plus a learned soft latent hypergraph assignment so
    both explicit merchant-velocity bursts and latent behavioral clusters propagate
    multi-hop contagion signals.

    Args:
        vocab_size: Number of unique MCC tokens.
        embed_dim: Entity embedding dimension (`D`).
        num_velocity_tiers: Number of log-time velocity buckets (default: 3).
        num_layers: Number of stacked `HypergraphConvLayer` blocks (default: 2).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_velocity_tiers: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_velocity_tiers = num_velocity_tiers
        self.num_explicit_edges = vocab_size * num_velocity_tiers
        self.num_latent_edges = 8
        self.total_hyperedges = self.num_explicit_edges + self.num_latent_edges

        self.latent_edge_proj = nn.Linear(embed_dim, self.num_latent_edges)
        self.layers = nn.ModuleList(
            [
                HypergraphConvLayer(
                    embed_dim=embed_dim,
                    num_hyperedges=self.total_hyperedges,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.collusion_scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def build_incidence_matrix(
        self,
        user_embeddings: torch.Tensor,
        event_keys: torch.Tensor,
        event_values: torch.Tensor,
        event_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Builds a mask-invariant hypergraph incidence matrix `H` of shape
        `[B, total_hyperedges]`.
        """
        B, S = event_keys.shape
        device = user_embeddings.device
        dtype = user_embeddings.dtype

        clamped_keys = event_keys.clamp(min=0, max=self.vocab_size - 1)
        # Velocity tier in {0, 1, ..., num_velocity_tiers - 1} based on inter-event time
        # Tier 0: rapid burst (time_delta <= 2), Tier 1: moderate (<= 15), Tier 2: slow (> 15)
        t_float = event_times.float().clamp(min=0.0)
        tier_0 = (t_float <= 2.0).long()
        tier_1 = ((t_float > 2.0) & (t_float <= 15.0)).long() * 1
        tier_2 = (t_float > 15.0).long() * min(2, self.num_velocity_tiers - 1)
        velocity_tier = (tier_0 * 0 + tier_1 + tier_2).clamp(
            min=0, max=self.num_velocity_tiers - 1
        )

        hyperedge_idx = clamped_keys * self.num_velocity_tiers + velocity_tier  # [B, S]
        weights = F.softplus(event_values.to(dtype))

        if key_padding_mask is not None:
            weights = weights.masked_fill(key_padding_mask, 0.0)

        explicit_H = torch.zeros(
            B, self.num_explicit_edges, device=device, dtype=dtype
        )
        explicit_H.scatter_add_(dim=1, index=hyperedge_idx, src=weights)
        explicit_H = torch.log1p(explicit_H)

        # Soft latent hyperedges from mask-invariant user_embeddings
        latent_H = F.softmax(self.latent_edge_proj(user_embeddings), dim=-1)

        return torch.cat([explicit_H, latent_H], dim=1)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        event_keys: torch.Tensor,
        event_values: torch.Tensor,
        event_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes hypergraph-enriched representations and collusion ring scores.

        Returns:
            Tuple `(enriched_embeddings [B, D], collusion_ring_score [B, 1])`.
        """
        H = self.build_incidence_matrix(
            user_embeddings=user_embeddings,
            event_keys=event_keys,
            event_values=event_values,
            event_times=event_times,
            key_padding_mask=key_padding_mask,
        )

        h = user_embeddings
        for layer in self.layers:
            h = layer(h, H)

        collusion_score = torch.sigmoid(self.collusion_scorer(h))
        return h, collusion_score

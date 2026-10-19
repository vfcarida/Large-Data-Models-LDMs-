"""
Joint Fusion Network (End-to-End Fusion Layer) using DCNv2.

============================================================================
PURPOSE
============================================================================
Combines dense sequential embeddings from the MMTT Transformer with
classical static tabular features (such as credit bureau scores, annual
income, and customer age) using a parallel Deep & Cross Network v2 (DCNv2).

DCNv2 learns explicit bounded-degree polynomial feature interactions between
neural sequence representations and static attributes, while a parallel deep
MLP captures implicit non-linear interactions.

============================================================================
KEY CONCEPTS
============================================================================
1. WHY NOT SIMPLE CONCATENATION?
   Standard MLPs require many layers and samples to approximate multiplicative
   feature crosses. DCNv2 explicitly computes polynomial feature interactions
   in O(d^2) or O(d*r) parameters per layer.

2. CROSS NETWORK FORMULA:
   x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
   where:
   - x_0 is the anchored input vector [Transformer_emb || Tabular_features]
   - * denotes element-wise (Hadamard) multiplication
   - W_l is a full-rank or low-rank (U * V^T) interaction matrix

3. PARALLEL ARCHITECTURE:
   The Cross Network and Deep Network process the concatenated input in
   parallel, and their representations are concatenated prior to the final
   linear classification head.

============================================================================
REFERENCES
============================================================================
  - Wang et al. (2021): "DCN V2: Improved Deep & Cross Network and Practical
    Lessons for Web-scale Learning to Rank Systems" (WWW 2021, arXiv:2008.13535)
  - Lian et al. (2018): "xDeepFM: Combining Explicit and Implicit Feature
    Interactions for Recommender Systems" (KDD 2018)
============================================================================
"""

from typing import Optional

import torch
import torch.nn as nn


class CrossNetworkV2(nn.Module):
    """
    Deep & Cross Network V2 — Explicit Polynomial Feature Interaction.

    Mathematical formulation:
        x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l

    Args:
        input_dim: Dimension of the input feature vector.
        num_layers: Number of cross layers (polynomial degree - 1).
        low_rank: If specified, uses low-rank decomposition `W_l = U_l * V_l^T`
            with rank `low_rank`, reducing parameters from O(d^2) to O(2*d*r).
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        low_rank: Optional[int] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.low_rank = low_rank

        if low_rank is not None:
            self.U = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
                    for _ in range(num_layers)
                ]
            )
            self.V = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.cross_weights = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(input_dim, input_dim))
                    for _ in range(num_layers)
                ]
            )
            for w in self.cross_weights:
                nn.init.xavier_uniform_(w)

        self.cross_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Computes explicit polynomial feature interactions.

        Args:
            x0: Input tensor `[batch_size, input_dim]`.

        Returns:
            Output tensor `[batch_size, input_dim]` with polynomial interactions.
        """
        xl = x0
        for i in range(self.num_layers):
            if self.low_rank is not None:
                proj = torch.matmul(xl, self.V[i])
                proj = torch.matmul(proj, self.U[i].T)
            else:
                proj = torch.matmul(xl, self.cross_weights[i])

            proj = proj + self.cross_biases[i]
            xl = x0 * proj + xl

        return xl


class EndToEndFusionLayer(nn.Module):
    """
    End-to-End Parallel DCNv2 Fusion Layer for Binary Risk Classification.

    Combines:
      1. Contextual pooled embeddings from the MMTT Transformer backbone
      2. Standardized static tabular features (bureau score, income, age)

    Args:
        transformer_hidden_size: Dimension of Transformer output embeddings.
        tabular_feature_size: Number of static tabular features.
        dcn_layers: Number of DCNv2 cross layers.
        low_rank: Low-rank decomposition rank (`None` for full-rank).
        deep_dropout: Dropout rate in the parallel deep MLP.
    """

    def __init__(
        self,
        transformer_hidden_size: int,
        tabular_feature_size: int,
        dcn_layers: int = 3,
        low_rank: Optional[int] = None,
        deep_dropout: float = 0.2,
    ):
        super().__init__()

        self.combined_dim = transformer_hidden_size + tabular_feature_size
        self.tabular_norm = nn.LayerNorm(tabular_feature_size)

        self.cross_net = CrossNetworkV2(
            input_dim=self.combined_dim,
            num_layers=dcn_layers,
            low_rank=low_rank,
        )

        deep_hidden = max(self.combined_dim // 2, 16)
        deep_out = max(self.combined_dim // 4, 8)

        # Use LayerNorm instead of BatchNorm1d so batch_size=1 inference never fails
        self.deep_net = nn.Sequential(
            nn.Linear(self.combined_dim, deep_hidden),
            nn.LayerNorm(deep_hidden),
            nn.GELU(),
            nn.Dropout(deep_dropout),
            nn.Linear(deep_hidden, deep_out),
            nn.LayerNorm(deep_out),
            nn.GELU(),
            nn.Dropout(deep_dropout),
        )

        self.final_projection = nn.Linear(self.combined_dim + deep_out, 1)

    def forward(
        self,
        transformer_emb: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the parallel DCNv2 fusion head.

        Args:
            transformer_emb: `[batch, hidden_size]` pooled Transformer output.
            tabular_features: `[batch, tabular_dim]` static tabular features.

        Returns:
            `[batch, 1]` binary classification logits (pre-sigmoid).
        """
        norm_tab = self.tabular_norm(tabular_features)
        x = torch.cat([transformer_emb, norm_tab], dim=-1)

        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)

        fused = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.final_projection(fused)

        return logits

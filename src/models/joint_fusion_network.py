"""
Joint Fusion Network (End-to-End Fusion Layer) using DCNv2.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
In many real-world applications, the LDM's Transformer embeddings are not
the ONLY signal available. Financial institutions also have access to
classical tabular features such as:
  - Credit bureau scores (e.g., FICO, Serasa)
  - Income, age, employment tenure
  - Macroeconomic indicators
  - Manually engineered features from domain experts

The challenge: how to COMBINE deep Transformer representations with these
classical features without losing either signal?

SOLUTION: Deep & Cross Network v2 (DCNv2)

DCNv2 learns EXPLICIT polynomial feature interactions between the neural
embeddings and tabular features, while a parallel deep network captures
implicit non-linear relationships. The two are concatenated for the final
prediction.

============================================================================
KEY CONCEPTS (CONCEITOS-CHAVE)
============================================================================
1. WHY NOT JUST CONCATENATE?
   Simple concatenation + MLP can learn feature interactions, but
   requires many layers and parameters to discover higher-order
   interactions. DCNv2 explicitly computes polynomial feature crosses
   in O(d) parameters per layer instead of O(d²).

2. CROSS NETWORK FORMULA:
   x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
   where:
   - x_0 is the original concatenated input
   - ⊙ is element-wise multiplication
   - W_l is a learnable weight matrix
   - The residual connection (+x_l) ensures gradient flow

3. PARALLEL ARCHITECTURE (not stacked):
   We use the PARALLEL variant of DCN, where the Cross Network and
   Deep Network process the input independently, then their outputs
   are concatenated. This preserves explicit interactions from the
   Cross Network alongside implicit non-linear transformations from
   the Deep Network.

============================================================================
REFERENCES
============================================================================
  - Wang et al. (2021): "DCN V2: Improved Deep & Cross Network"
  - Lian et al. (2018): "xDeepFM: Combining Explicit and Implicit
    Feature Interactions for Recommender Systems"
  - Google Research: DCN for CTR prediction in recommendation systems
============================================================================
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossNetworkV2(nn.Module):
    """
    Deep & Cross Network V2 — Explicit Polynomial Feature Interaction.

    Unlike standard MLPs that learn IMPLICIT interactions through
    multiple non-linear layers, DCNv2 learns EXPLICIT bounded-degree
    polynomial interactions. This is more parameter-efficient and
    mathematically interpretable.

    Mathematical formulation:
        x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l

    Where:
        x_0: Original input (anchored)
        W_l: Weight matrix at layer l (full-rank or low-rank)
        b_l: Bias at layer l
        ⊙:   Element-wise multiplication (Hadamard product)

    The key insight is that the ORIGINAL input x_0 is multiplied with
    the projection at every layer, creating explicit polynomial terms:
    - Layer 1: degree-2 interactions (x_0 ⊙ Wx_0 = quadratic terms)
    - Layer 2: degree-3 interactions
    - Layer k: degree-(k+1) interactions

    Args:
        input_dim: Dimension of the input feature vector.
        num_layers: Number of cross layers (degree of polynomial - 1).
        low_rank: If not None, uses low-rank decomposition with this rank.
                  Reduces parameters from O(d²) to O(2·d·r) per layer.

    Example:
        >>> cross = CrossNetworkV2(input_dim=256, num_layers=3)
        >>> x = torch.randn(32, 256)
        >>> out = cross(x)  # [32, 256]
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
            # =====================================================================
            # Low-Rank Decomposition: W = U · V^T
            # =====================================================================
            # Instead of storing a full d×d matrix (d² parameters),
            # we decompose it into two matrices U (d×r) and V (d×r),
            # reducing parameters to 2·d·r where r << d.
            #
            # This is especially beneficial when input_dim is large
            # (e.g., 768 + tabular_features), as it prevents overfitting
            # and reduces computation.
            # =====================================================================
            self.U = nn.ParameterList([
                nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
                for _ in range(num_layers)
            ])
            self.V = nn.ParameterList([
                nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
                for _ in range(num_layers)
            ])
        else:
            # Full-rank weight matrices
            self.cross_weights = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, input_dim))
                for _ in range(num_layers)
            ])
            # Xavier initialization for numerical stability
            for w in self.cross_weights:
                nn.init.xavier_uniform_(w)

        # Bias terms (one per layer)
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Computes explicit polynomial feature interactions.

        Args:
            x0: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, input_dim] with polynomial interactions.
        """
        xl = x0
        for i in range(self.num_layers):
            if self.low_rank is not None:
                # Low-rank: W·x = U·(V^T·x)
                # Compute V^T·x first (d→r), then U·result (r→d)
                # This is O(d·r) instead of O(d²)
                proj = torch.matmul(xl, self.V[i])    # [B, r]
                proj = torch.matmul(proj, self.U[i].T)  # [B, d]
            else:
                # Full-rank: W·x directly
                proj = torch.matmul(xl, self.cross_weights[i])

            proj = proj + self.cross_biases[i]

            # Core DCNv2 formula: x_{l+1} = x_0 ⊙ proj + x_l
            xl = x0 * proj + xl

        return xl


class EndToEndFusionLayer(nn.Module):
    """
    End-to-End Fusion Layer for Binary Classification.

    Combines two complementary information sources:
    1. Contextual embeddings from the MMTT Transformer (deep behavioral patterns)
    2. Classical tabular features from legacy systems (bureau scores, demographics)

    The fusion uses PARALLEL Deep & Cross architecture:
    - Cross Network: Explicit polynomial interactions between sources
    - Deep Network: Non-linear dense transformations for implicit patterns
    - Final Projection: Concatenates both outputs for binary prediction

    Architecture diagram:
        Transformer_emb + Tabular_features → Concatenation → ┌─ CrossNet ──┐
                                                             │              │ → Concat → Linear → logit
                                                             └── DeepNet ──┘

    Args:
        transformer_hidden_size: Dimension of Transformer output embeddings.
        tabular_feature_size: Number of classical tabular features.
        dcn_layers: Number of DCNv2 cross layers (polynomial degree - 1).
        low_rank: Low-rank decomposition rank (None for full-rank).
        deep_dropout: Dropout rate in the deep network.

    Example:
        >>> fusion = EndToEndFusionLayer(
        ...     transformer_hidden_size=128,
        ...     tabular_feature_size=3,
        ...     dcn_layers=3
        ... )
        >>> transformer_emb = torch.randn(32, 128)  # From encoder pooling
        >>> tabular = torch.randn(32, 3)             # Bureau, income, age
        >>> logits = fusion(transformer_emb, tabular)
        >>> print(logits.shape)  # [32, 1]
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

        # =====================================================================
        # Cross Network: Explicit polynomial feature interactions
        # =====================================================================
        self.cross_net = CrossNetworkV2(
            input_dim=self.combined_dim,
            num_layers=dcn_layers,
            low_rank=low_rank,
        )

        # =====================================================================
        # Deep Network: Non-linear dense transformations
        # =====================================================================
        # Architecture: combined_dim → combined_dim//2 → combined_dim//4
        # Uses GELU activation (smoother than ReLU, standard in modern models)
        # BatchNorm for training stability
        # Dropout for regularization
        # =====================================================================
        deep_hidden = max(self.combined_dim // 2, 16)
        deep_out = max(self.combined_dim // 4, 8)

        self.deep_net = nn.Sequential(
            nn.Linear(self.combined_dim, deep_hidden),
            nn.BatchNorm1d(deep_hidden),
            nn.GELU(),
            nn.Dropout(deep_dropout),
            nn.Linear(deep_hidden, deep_out),
            nn.BatchNorm1d(deep_out),
            nn.GELU(),
            nn.Dropout(deep_dropout),
        )

        # =====================================================================
        # Final Projection: Cross output + Deep output → single logit
        # =====================================================================
        # Output is a single logit (no sigmoid), compatible with
        # BCEWithLogitsLoss which combines sigmoid + BCE for numerical stability.
        # =====================================================================
        self.final_projection = nn.Linear(self.combined_dim + deep_out, 1)

    def forward(
        self,
        transformer_emb: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the fusion layer.

        Args:
            transformer_emb: [batch, hidden_size] Pooled Transformer output.
            tabular_features: [batch, tabular_dim] Classical tabular features.

        Returns:
            [batch, 1] Binary logits (pre-sigmoid, for BCEWithLogitsLoss).
        """
        # Concatenate Transformer embeddings with tabular features
        x = torch.cat([transformer_emb, tabular_features], dim=-1)

        # Parallel processing
        cross_out = self.cross_net(x)     # Explicit polynomial interactions
        deep_out = self.deep_net(x)       # Implicit non-linear patterns

        # Final fusion and projection
        fused = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.final_projection(fused)

        return logits

r"""
In-Context Tabular Foundation Inference (`TabPFN v2` / `TabICL`).

Scientific & Architectural Anchors:
  - Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B.,
    Schirrmeister, R. T., & Hutter, F. (2025). "Accurate predictions on small data
    with a tabular foundation model (TabPFN v2)." *Nature*, 637, 319-326.
    DOI: 10.1038/s41586-024-08328-6
  - Qu, Y., et al. (2025). "TabICL: A Tabular Foundation Model for In-Context
    Learning on Large Data." *ICML 2025*, arXiv:2502.05564.

Performs zero-gradient Bayesian in-context inference $p(y_{\text{query}} \mid x_{\text{query}}, \mathcal{D}_{\text{support}})$
over `LargeDataModel` joint representations via Support-Label Conditioning and
Support-to-Query Cross-Attention. Enables instant adaptation to cold-start merchant
categories, newly launched payment rails, and emerging fraud rings with only a
few labeled support examples.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class InContextTabularPredictor(nn.Module):
    r"""
    Zero-gradient In-Context Learning (`ICL`) head for tabular and transactional
    foundation representations.

    Given query representations $Q \in \mathbb{R}^{B_q \times D}$, support
    representations $S \in \mathbb{R}^{B_s \times D}$, and support labels
    $y_s \in [0, 1]^{B_s}$, computes:
      1. Metric-space cosine attention logits scaled by inverse temperature $1/\tau$,
         fused with learned query/key projections.
      2. Label-conditioned support value embeddings $V_s = \phi_S(S) + \phi_Y(y_s)$.
      3. Non-parametric Bayesian kernel readout combined with a residual neural
         cross-attention refinement head.

    Args:
        repr_dim: Dimension $D$ of the LDM fused representation.
        num_heads: Number of cross-attention heads (default: `4`).
        temperature: Softmax temperature $\tau > 0$ for support-query similarity
            (default: `0.25`).
    """

    def __init__(
        self,
        repr_dim: int,
        num_heads: int = 4,
        temperature: float = 0.25,
    ):
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.num_heads = int(num_heads)
        self.temperature = max(float(temperature), 1e-3)

        self.query_norm = nn.LayerNorm(self.repr_dim)
        self.support_norm = nn.LayerNorm(self.repr_dim)

        # Support label embedding: maps y_s in [0, 1] -> R^D via [1 - y_s, y_s]
        self.label_encoder = nn.Linear(2, self.repr_dim)

        # Learned query, key, value projections for residual refinement
        self.q_proj = nn.Linear(self.repr_dim, self.repr_dim)
        self.k_proj = nn.Linear(self.repr_dim, self.repr_dim)
        self.v_proj = nn.Linear(self.repr_dim, self.repr_dim)

        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.repr_dim * 2),
            nn.Linear(self.repr_dim * 2, self.repr_dim),
            nn.GELU(),
            nn.Linear(self.repr_dim, 1),
        )
        # Initialize final residual layer near zero so initial posterior matches
        # calibrated metric-space Bayesian kernel interpolation
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        query_repr: torch.Tensor,
        support_repr: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Performs in-context posterior inference for `query_repr` conditioned on
        (`support_repr`, `support_labels`).

        Args:
            query_repr: Query embeddings `[B_q, D]`.
            support_repr: Support set embeddings `[B_s, D]`.
            support_labels: Support set binary/continuous risk targets `[B_s]` or `[B_s, 1]`.

        Returns:
            Dictionary containing:
              - `"icl_logits"`: `[B_q, 1]` posterior log-odds.
              - `"icl_probabilities"`: `[B_q, 1]` posterior fraud probability $p(y=1 \mid x_q, \mathcal{D}_s)$.
              - `"attention_weights"`: `[B_q, B_s]` normalized support attention weights $\alpha_{ij}$.
              - `"icl_predictive_entropy"`: `[B_q]` binary Shannon entropy in $[0, \ln 2]$.
              - `"effective_support_neighbors"`: `[B_q]` attention perplexity $\exp(-\sum_j \alpha_{ij}\ln\alpha_{ij})$.
        """
        y_s = support_labels.view(-1, 1).float().clamp(0.0, 1.0)
        y_two_hot = torch.cat([1.0 - y_s, y_s], dim=-1)  # [B_s, 2]

        q_normed = self.query_norm(query_repr)
        s_normed = self.support_norm(support_repr)

        # Cosine metric similarity on uncentered L2-normalized representations + learned bilinear projection
        q_unit = F.normalize(query_repr, p=2.0, dim=-1)
        s_unit = F.normalize(support_repr, p=2.0, dim=-1)
        metric_logits = torch.matmul(q_unit, s_unit.transpose(0, 1)) / self.temperature

        q_learned = F.normalize(self.q_proj(q_normed), p=2.0, dim=-1)
        k_learned = F.normalize(self.k_proj(s_normed), p=2.0, dim=-1)
        learned_logits = torch.matmul(q_learned, k_learned.transpose(0, 1))

        attn_logits = metric_logits + 0.25 * learned_logits
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B_q, B_s]

        # Non-parametric Bayesian kernel probability readout
        prior_prob = torch.matmul(attn_weights, y_s).clamp(1e-4, 1.0 - 1e-4)  # [B_q, 1]
        base_logit = torch.log(prior_prob / (1.0 - prior_prob))

        # Label-conditioned value aggregation for neural residual correction
        support_values = self.v_proj(s_normed) + self.label_encoder(y_two_hot)  # [B_s, D]
        aggregated_context = torch.matmul(attn_weights, support_values)  # [B_q, D]

        residual_logit = self.residual_head(
            torch.cat([q_normed, aggregated_context], dim=-1)
        )
        icl_logits = base_logit + residual_logit
        icl_probs = torch.sigmoid(icl_logits)

        # Predictive Shannon entropy H(p) = -p log p - (1-p) log(1-p)
        p_flat = icl_probs.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
        pred_entropy = -(
            p_flat * torch.log(p_flat) + (1.0 - p_flat) * torch.log(1.0 - p_flat)
        )

        # Effective number of support neighbors (perplexity of attention distribution)
        safe_attn = attn_weights.clamp(min=1e-8)
        attn_entropy = -torch.sum(safe_attn * torch.log(safe_attn), dim=-1)
        effective_neighbors = torch.exp(attn_entropy)

        return {
            "icl_logits": icl_logits,
            "icl_probabilities": icl_probs,
            "attention_weights": attn_weights,
            "icl_predictive_entropy": pred_entropy,
            "effective_support_neighbors": effective_neighbors,
        }

    def compute_icl_episodic_loss(
        self,
        query_repr: torch.Tensor,
        support_repr: torch.Tensor,
        support_labels: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Computes the episodic binary cross-entropy loss on query labels given a
        support context $(\mathcal{D}_{\text{support}})$ for meta-training the
        in-context cross-attention head.
        """
        out = self.forward(query_repr, support_repr, support_labels)
        targets = query_labels.view(-1, 1).float()
        return F.binary_cross_entropy_with_logits(out["icl_logits"], targets)

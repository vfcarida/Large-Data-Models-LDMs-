r"""
Multi-Currency Cross-Border Purchasing-Power & Deep CORAL Domain Alignment (`CrossBorderLDM`).

============================================================================
RESEARCH & ARCHITECTURAL ANCHORS
============================================================================
Global payment networks ingest transaction streams across heterogeneous
currencies (`USD`, `BRL`, `EUR`, `GBP`, `JPY`, `MXN`, `INR`, `SGD`) and regional
economic regimes. Directly feeding unnormalized nominal amounts or unaligned
regional embeddings causes severe false-positive spikes on cross-border
travelers and zero-shot regional deployments.

Synthesizes two domain-invariant representation learning paradigms:
  1. Purchasing-Power Parity (PPP) Log-Ratio & FiLM Conditioning (Perez et al.,
     AAAI 2018, "FiLM: Visual Reasoning with a General Conditioning Layer"):
     - Converts nominal transaction amounts into FX/PPP-adjusted log-quantities
       and modulates entity embeddings via currency-specific affine scale/shift
       $(\gamma_c, \beta_c)$.
  2. Deep CORAL Second-Order Covariance Alignment (Sun & Saenko, ECCV 2016,
     "Deep CORAL: Correlation Alignment for Deep Domain Adaptation",
     arXiv:1607.01719) + First-Order Mean Matching:
     - Aligns both first-order centroids and second-order covariance matrices
       $C_S, C_T \in \mathbb{R}^{D \times D}$ between source and target currency
       jurisdictions:
       $$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 D^2} \| C_S - C_T \|_F^2 + \|\mu_S - \mu_T\|_2^2$$
============================================================================
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference Purchasing-Power / FX conversion factors to USD baseline
DEFAULT_FX_RATES_TO_USD: Tuple[float, ...] = (
    1.0000,  # 0: USD
    0.2000,  # 1: BRL
    1.0800,  # 2: EUR
    1.2700,  # 3: GBP
    0.0067,  # 4: JPY
    0.0580,  # 5: MXN
    0.0120,  # 6: INR
    0.7400,  # 7: SGD
)


class FXPurchasingPowerNormalizer(nn.Module):
    r"""
    Normalizes multi-currency nominal transaction amounts into FX/PPP-invariant
    log-standardized values with learnable per-currency purchasing-power offsets.

    Args:
        num_currencies: Number of supported currency codes (default: `8`).
    """

    def __init__(self, num_currencies: int = 8):
        super().__init__()
        self.num_currencies = num_currencies

        rates = list(DEFAULT_FX_RATES_TO_USD[:num_currencies])
        while len(rates) < num_currencies:
            rates.append(1.0)
        self.register_buffer(
            "fx_to_usd", torch.tensor(rates, dtype=torch.float32)
        )
        # Learnable log-PPP residual adjustment per currency (initialized to 0)
        self.ppp_log_offset = nn.Parameter(torch.zeros(num_currencies))

    def forward(
        self,
        nominal_values: torch.Tensor,
        currency_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Converts nominal transaction amounts `[B, S]` into PPP-aligned log-USD
        standardized values `[B, S]`.

        Args:
            nominal_values: Raw or z-scored monetary values `[B, S]`.
            currency_ids: Currency index per user `[B]` or per transaction `[B, S]`
                in `{0, ..., num_currencies - 1}`.
            key_padding_mask: Optional boolean mask `[B, S]` (`True` = padded).

        Returns:
            PPP-standardized transaction values `[B, S]`.
        """
        c_idx = currency_ids.long().clamp(0, self.num_currencies - 1)
        if c_idx.dim() == 1:
            c_idx = c_idx.unsqueeze(1).expand_as(nominal_values)

        fx_rate = self.fx_to_usd[c_idx]
        ppp_offset = self.ppp_log_offset[c_idx]

        # Signed log1p transformation scaled by FX rate and learnable PPP offset
        abs_val = nominal_values.float().abs() * fx_rate
        signed_log = torch.sign(nominal_values.float()) * torch.log1p(abs_val)
        aligned_val = signed_log + ppp_offset

        if key_padding_mask is not None:
            aligned_val = aligned_val.masked_fill(key_padding_mask, 0.0)
        return aligned_val


class CrossBorderDomainAligner(nn.Module):
    r"""
    Aligns behavioral entity embeddings across regional payment rails using
    Jurisdiction FiLM Conditioning and Deep CORAL Covariance Alignment.

    Args:
        embed_dim: Dimension of entity embeddings $h \in \mathbb{R}^D$.
        num_currencies: Number of currency/jurisdiction domains (default: `8`).
    """

    def __init__(self, embed_dim: int, num_currencies: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_currencies = num_currencies

        self.fx_normalizer = FXPurchasingPowerNormalizer(num_currencies=num_currencies)
        self.domain_gamma = nn.Embedding(num_currencies, embed_dim)
        self.domain_beta = nn.Embedding(num_currencies, embed_dim)
        nn.init.ones_(self.domain_gamma.weight)
        nn.init.zeros_(self.domain_beta.weight)

        self.shared_adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(
        self,
        entity_embeddings: torch.Tensor,
        currency_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Maps regional `entity_embeddings` `[B, D]` into a shared FX/jurisdiction-invariant
        representation space `[B, D]`.
        """
        c_idx = currency_ids.view(-1).long().clamp(0, self.num_currencies - 1)
        gamma = self.domain_gamma(c_idx)
        beta = self.domain_beta(c_idx)

        modulated = gamma * entity_embeddings + beta
        return modulated + self.residual_scale * self.shared_adapter(modulated)

    @staticmethod
    def _compute_covariance(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if n <= 1:
            return torch.zeros(
                (x.size(1), x.size(1)), dtype=x.dtype, device=x.device
            )
        centered = x - x.mean(dim=0, keepdim=True)
        return torch.matmul(centered.transpose(0, 1), centered) / float(n - 1)

    def compute_alignment_loss(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        source_currency_ids: Optional[torch.Tensor] = None,
        target_currency_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes the Deep CORAL second-order covariance distance + first-order
        centroid distance between a source region cohort and a target region cohort.

        Returns:
            Dictionary with `"loss"`, `"coral_loss"`, and `"centroid_loss"`.
        """
        if source_currency_ids is not None:
            src_aligned = self.forward(source_embeddings, source_currency_ids)
        else:
            src_aligned = source_embeddings

        if target_currency_ids is not None:
            tgt_aligned = self.forward(target_embeddings, target_currency_ids)
        else:
            tgt_aligned = target_embeddings

        d = float(src_aligned.size(-1))
        cov_s = self._compute_covariance(src_aligned)
        cov_t = self._compute_covariance(tgt_aligned)

        coral_loss = torch.sum((cov_s - cov_t).pow(2)) / (4.0 * d * d)
        centroid_loss = F.mse_loss(
            src_aligned.mean(dim=0), tgt_aligned.mean(dim=0)
        )
        total_loss = coral_loss + centroid_loss

        return {
            "loss": total_loss,
            "coral_loss": coral_loss.detach(),
            "centroid_loss": centroid_loss.detach(),
        }

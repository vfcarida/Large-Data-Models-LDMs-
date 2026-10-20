r"""
Competing-Risks Survival Analysis & IFRS 9 / CECL Lifetime Hazard Head (`DeepHit`).

Scientific & Regulatory Anchors:
  - Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). "DeepHit: A Deep
    Learning Approach to Survival Analysis with Competing Risks." *AAAI 2018*.
  - Kvamme, H., Borgan, Ø., & Scheel, I. (2019). "Time-to-Event Prediction with
    Neural Networks and Cox Regression." *Journal of Machine Learning Research (JMLR)*,
    20(129), 1-30.
  - IASB IFRS 9 & FASB ASC 326 (CECL): Stage-1 (12-month) and Stage-2 (Lifetime)
    Expected Credit & Fraud Loss (`ECL = PD(tau) * LGD * EAD`) under competing
    default, chargeback, and voluntary attrition events.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_COMPETING_CAUSES = (
    "FRAUD_CHARGEBACK",
    "CREDIT_DEFAULT",
    "VOLUNTARY_CHURN",
)


class CompetingRisksSurvivalHead(nn.Module):
    r"""
    Discrete-time competing-risks survival and hazard head (`DeepHit`) over
    `LargeDataModel` joint representations.

    Models the joint probability distribution over $K$ competing event causes
    and $H$ discrete time horizons $\tau \in \{0, \dots, H-1\}$, plus the
    event-free survival outcome beyond horizon $H$:
    $$\sum_{k=0}^{K-1} \sum_{\tau=0}^{H-1} P(T = \tau, \text{cause}=k \mid x) + P(T \ge H \mid x) = 1$$

    Args:
        repr_dim: Dimension $D$ of the LDM fused representation.
        num_causes: Number of competing event types $K$ (default: `3`).
        num_horizons: Number of discrete time buckets $H$ (default: `12`).
        cause_names: Names of the $K$ competing causes.
        default_lgd: Default Loss Given Default (`LGD`) per cause in $[0, 1]^K$.
    """

    def __init__(
        self,
        repr_dim: int,
        num_causes: int = 3,
        num_horizons: int = 12,
        cause_names: Sequence[str] = DEFAULT_COMPETING_CAUSES,
        default_lgd: Sequence[float] = (1.00, 0.65, 0.08),
    ):
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.num_causes = int(num_causes)
        self.num_horizons = int(num_horizons)
        self.cause_names = tuple(cause_names)[: self.num_causes]

        lgd_list = list(default_lgd)[: self.num_causes]
        while len(lgd_list) < self.num_causes:
            lgd_list.append(0.50)
        self.register_buffer(
            "default_lgd",
            torch.tensor(lgd_list, dtype=torch.float32),
        )

        total_outcomes = self.num_causes * self.num_horizons + 1
        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(self.repr_dim),
            nn.Linear(self.repr_dim, self.repr_dim),
            nn.GELU(),
        )
        # Residual connection from raw representation into cause-horizon head
        self.outcome_head = nn.Linear(self.repr_dim * 2, total_outcomes)

    def forward(
        self,
        fused_repr: torch.Tensor,
        exposure_at_default: Optional[torch.Tensor] = None,
        stage1_horizon_idx: int = 2,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes cause-specific probability mass, Cumulative Incidence Functions
        ($\text{CIF}_k(\tau \mid x)$), survival curves $S(\tau \mid x)$, and
        IFRS 9 / CECL Expected Loss (`ECL`).

        Args:
            fused_repr: LDM joint representation `[B, D]`.
            exposure_at_default: Optional account exposure `EAD` `[B]`. Defaults to `1.0`.
            stage1_horizon_idx: Horizon index separating short-term (Stage 1) from
                lifetime (Stage 2) risk (default: `2`).

        Returns:
            Dictionary containing:
              - `"joint_event_probs"`: `[B, K, H]` probability $P(T = \tau, \text{cause}=k \mid x)$.
              - `"survival_beyond_horizon"`: `[B, 1]` probability $P(T \ge H \mid x)$.
              - `"cif"`: `[B, K, H]` Cumulative Incidence Function $\text{CIF}_k(\tau \mid x)$.
              - `"survival_curve"`: `[B, H]` overall survival probability $S(\tau \mid x)$.
              - `"stage1_ecl"`: `[B, 1]` short-horizon expected loss.
              - `"lifetime_ecl"`: `[B, 1]` full-horizon lifetime expected loss (CECL / IFRS 9 Stage 2).
        """
        batch_size = fused_repr.size(0)
        hidden = self.shared_trunk(fused_repr)
        concat_repr = torch.cat([fused_repr, hidden], dim=-1)

        logits = self.outcome_head(concat_repr)  # [B, K*H + 1]
        probs = F.softmax(logits, dim=-1)

        event_probs_flat = probs[:, :-1]  # [B, K*H]
        surv_beyond = probs[:, -1:]  # [B, 1]

        joint_probs = event_probs_flat.view(
            batch_size, self.num_causes, self.num_horizons
        )
        # Cumulative Incidence Function CIF_k(tau | x) = sum_{u <= tau} P(T=u, k | x)
        cif = torch.cumsum(joint_probs, dim=-1)  # [B, K, H]

        # Overall survival curve S(tau | x) = 1 - sum_k CIF_k(tau | x)
        total_cumulative_incidence = torch.sum(cif, dim=1)  # [B, H]
        survival_curve = (1.0 - total_cumulative_incidence).clamp(min=0.0, max=1.0)

        if exposure_at_default is None:
            ead = torch.ones(
                (batch_size, 1), device=fused_repr.device, dtype=fused_repr.dtype
            )
        else:
            ead = exposure_at_default.view(batch_size, 1).to(
                device=fused_repr.device, dtype=fused_repr.dtype
            )

        lgd = self.default_lgd.view(1, self.num_causes).to(
            device=fused_repr.device, dtype=fused_repr.dtype
        )
        s1_idx = min(max(int(stage1_horizon_idx), 0), self.num_horizons - 1)

        # Expected loss = EAD * sum_k CIF_k(tau) * LGD_k
        stage1_loss_rate = torch.sum(cif[:, :, s1_idx] * lgd, dim=-1, keepdim=True)
        lifetime_loss_rate = torch.sum(cif[:, :, -1] * lgd, dim=-1, keepdim=True)

        return {
            "joint_event_probs": joint_probs,
            "survival_beyond_horizon": surv_beyond,
            "cif": cif,
            "survival_curve": survival_curve,
            "stage1_ecl": ead * stage1_loss_rate,
            "lifetime_ecl": ead * lifetime_loss_rate,
        }

    def compute_deephit_loss(
        self,
        fused_repr: torch.Tensor,
        event_times: torch.Tensor,
        event_causes: torch.Tensor,
        ranking_weight: float = 0.25,
        ranking_sigma: float = 0.10,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes the DeepHit competing-risks loss under right-censoring:
        $$\mathcal{L}_{\text{DeepHit}} = \mathcal{L}_{\text{NLL-censored}} + \alpha_{\text{rank}} \mathcal{L}_{\text{rank}}$$

        Args:
            fused_repr: LDM joint representation `[B, D]`.
            event_times: Observed event or censoring time index `[B]` in $\{0, \dots, H-1\}$.
            event_causes: Cause indicator `[B]` in $\{0, 1, \dots, K\}$, where `0`
                indicates right-censored at `event_times[i]` and $k \in \{1, \dots, K\}$
                indicates cause $k-1$ occurred at `event_times[i]`.
            ranking_weight: Weight on the concordance ranking penalty.
            ranking_sigma: Smoothing scale $\sigma$ in $\exp(-(\text{CIF}_i - \text{CIF}_j)/\sigma)$.
        """
        out = self.forward(fused_repr)
        joint_probs = out["joint_event_probs"]  # [B, K, H]
        cif = out["cif"]  # [B, K, H]
        survival_curve = out["survival_curve"]  # [B, H]

        batch_size = fused_repr.size(0)
        t_idx = event_times.view(-1).long().clamp(0, self.num_horizons - 1)
        causes = event_causes.view(-1).long().clamp(0, self.num_causes)

        batch_indices = torch.arange(batch_size, device=fused_repr.device)
        uncensored_mask = causes > 0
        censored_mask = ~uncensored_mask

        nll_terms = torch.zeros(
            batch_size, device=fused_repr.device, dtype=fused_repr.dtype
        )
        if uncensored_mask.any():
            u_idx = batch_indices[uncensored_mask]
            u_cause = causes[uncensored_mask] - 1
            u_time = t_idx[uncensored_mask]
            p_event = joint_probs[u_idx, u_cause, u_time].clamp(min=1e-7)
            nll_terms[uncensored_mask] = -torch.log(p_event)

        if censored_mask.any():
            c_idx = batch_indices[censored_mask]
            c_time = t_idx[censored_mask]
            p_surv = survival_curve[c_idx, c_time].clamp(min=1e-7)
            nll_terms[censored_mask] = -torch.log(p_surv)

        nll_loss = nll_terms.mean()

        # Cause-specific concordance ranking penalty
        rank_loss = torch.tensor(0.0, device=fused_repr.device, dtype=fused_repr.dtype)
        valid_pairs = 0
        for i in range(batch_size):
            k_i = int(causes[i].item()) - 1
            if k_i < 0:
                continue
            t_i = int(t_idx[i].item())
            # Compare against samples j that survived beyond t_i
            later_mask = t_idx > t_i
            if later_mask.any():
                cif_i = cif[i, k_i, t_i]
                cif_j = cif[later_mask, k_i, t_i]
                diff = cif_i - cif_j
                rank_loss = rank_loss + torch.mean(
                    torch.exp(-diff / max(ranking_sigma, 1e-3))
                )
                valid_pairs += 1

        if valid_pairs > 0:
            rank_loss = rank_loss / float(valid_pairs)

        total_loss = nll_loss + ranking_weight * rank_loss
        return {
            "total_loss": total_loss,
            "nll_loss": nll_loss,
            "ranking_loss": rank_loss,
        }

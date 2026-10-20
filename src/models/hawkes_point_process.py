r"""
Self-Exciting Neural Marked Temporal Point Process Head (`NeuralHawkesLDM` / `THP`).

============================================================================
RESEARCH & ARCHITECTURAL ANCHORS
============================================================================
Automated card-testing bots, credential-stuffing scripts, and account-takeover
(ATO) cascades exhibit strong temporal self-excitation: each successful probe
triggers an immediate surge in subsequent transaction arrival intensity
$\lambda^*(t \mid \mathcal{H}_t)$.

Synthesizes two foundational continuous-time point process architectures:
  1. The Neural Hawkes Process (Mei & Eisner, NeurIPS 2017,
     "The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point
     Process", arXiv:1612.09328):
     - Formulates the continuous-time conditional hazard intensity as a
       softplus-modulated decay process over hidden history states $h_i$:
       $$\lambda^*(t \mid \mathcal{H}_{t_i}) = \text{softplus}\!\left(\mu + \alpha(h_i) \exp(-\beta(h_i) (t - t_i))\right)$$
  2. Transformer Hawkes Process (`THP`, Zuo et al., ICML 2020,
     "Transformer Hawkes Process", arXiv:2002.09291):
     - Couples contextualized Transformer/SSM sequence states $H \in \mathbb{R}^{B \times S \times D}$
       with exact point-process log-likelihood (log-intensity at arrival times
       minus analytical compensator integral $\int_{t_{i-1}}^{t_i} \lambda^*(t)\,dt$)
       and extracts the self-excitation branching ratio $\eta \in (0, 1)$.
============================================================================
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralHawkesProcessHead(nn.Module):
    r"""
    Continuous-time self-exciting Neural Hawkes Point Process head over
    contextualized LDM sequence representations.

    Args:
        hidden_dim: Dimension of per-token sequence embeddings $h_i \in \mathbb{R}^D$.
        num_event_types: Number of discrete MCC / event mark categories (default: `16`).
    """

    def __init__(self, hidden_dim: int, num_event_types: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_event_types = num_event_types

        # Base background intensity mu > 0
        self.base_mu = nn.Parameter(torch.tensor(0.20))

        # History-conditioned self-excitation amplitude alpha(h_i) > 0
        self.alpha_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # History-conditioned exponential decay rate beta(h_i) > 0
        self.beta_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Next event mark (MCC) distribution conditioned on intensity & state
        self.mark_head = nn.Linear(hidden_dim + 1, num_event_types)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        delta_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes per-step conditional hazard intensities $\lambda^*(t_i)$,
        sequence-level self-excitation `branching_ratio` $\in (0, 1)$, and
        expected next inter-arrival time $\mathbb{E}[\Delta t_{S+1}]$.

        Args:
            token_embeddings: Per-event contextualized states `[B, S, D]`
                (after stripping any virtual prompt tokens).
            delta_times: Non-negative inter-event time deltas `[B, S]`.
            key_padding_mask: Optional boolean mask `[B, S]` where `True`
                indicates padded positions.

        Returns:
            Dictionary containing:
              - `"conditional_intensity"`: $\lambda^*(t_i)$ `[B, S]`
              - `"excitation_alpha"`: $\alpha(h_i)$ `[B, S]`
              - `"decay_beta"`: $\beta(h_i)$ `[B, S]`
              - `"branching_ratio"`: Self-excitation index `[B, 1]` in `(0, 1)`
              - `"expected_next_delta_t"`: Predicted time to next event `[B, 1]`
              - `"next_mark_logits"`: Predicted next MCC logits `[B, num_event_types]`
        """
        dt = delta_times.float().clamp(min=0.0, max=100.0)
        # Scale discrete/raw time bins into unit continuous time intervals
        dt_scaled = dt * 0.10

        mu = F.softplus(self.base_mu) + 1e-4
        alpha = F.softplus(self.alpha_proj(token_embeddings).squeeze(-1)) + 1e-4  # [B, S]
        beta = F.softplus(self.beta_proj(token_embeddings).squeeze(-1)) + 0.50  # [B, S]

        # Conditional intensity at arrival: lambda*(t_i) = mu + alpha * exp(-beta * dt)
        decay_kernel = torch.exp(-beta * dt_scaled)
        intensity = mu + alpha * decay_kernel  # [B, S]

        # Local branching ratio alpha / (alpha + beta) in (0, 1)
        step_branching = alpha / (alpha + beta + 1e-6)

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).float()
            intensity = intensity * valid_mask
            step_branching = step_branching * valid_mask
            denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_intensity = intensity.sum(dim=1, keepdim=True) / denom
            branching_ratio = step_branching.sum(dim=1, keepdim=True) / denom
            # Pool valid token embeddings for next-mark prediction
            pooled_h = (token_embeddings * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            mean_intensity = intensity.mean(dim=1, keepdim=True)
            branching_ratio = step_branching.mean(dim=1, keepdim=True)
            pooled_h = token_embeddings.mean(dim=1)

        expected_next_dt = 1.0 / mean_intensity.clamp(min=1e-4)
        mark_input = torch.cat([pooled_h, mean_intensity], dim=-1)
        next_mark_logits = self.mark_head(mark_input)

        return {
            "conditional_intensity": intensity,
            "excitation_alpha": alpha,
            "decay_beta": beta,
            "branching_ratio": branching_ratio,
            "expected_next_delta_t": expected_next_dt,
            "next_mark_logits": next_mark_logits,
        }

    def compute_point_process_nll(
        self,
        token_embeddings: torch.Tensor,
        delta_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes the exact continuous-time point process Negative Log-Likelihood:
        $$\mathcal{L}_{\text{NLL}} = \frac{1}{N_{\text{valid}}} \sum_{i} \left[ -\log \lambda^*(t_i) + \int_{t_{i-1}}^{t_i} \lambda^*(t)\,dt \right]$$
        where the analytical compensator over $[t_{i-1}, t_i]$ with $\Delta t_i = t_i - t_{i-1}$ is:
        $$\Lambda_i = \mu \Delta t_i + \frac{\alpha_i}{\beta_i} \left(1 - \exp(-\beta_i \Delta t_i)\right)$$
        """
        out = self.forward(token_embeddings, delta_times, key_padding_mask=key_padding_mask)
        intensity = out["conditional_intensity"]
        alpha = out["excitation_alpha"]
        beta = out["decay_beta"]
        mu = F.softplus(self.base_mu) + 1e-4
        dt_scaled = delta_times.float().clamp(min=0.0, max=100.0) * 0.10

        log_lambda = torch.log(intensity.clamp(min=1e-6))
        compensator = mu * dt_scaled + (alpha / beta) * (
            1.0 - torch.exp(-beta * dt_scaled)
        )
        step_nll = -log_lambda + compensator

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).float()
            step_nll = step_nll * valid_mask
            n_valid = valid_mask.sum().clamp(min=1.0)
            nll = step_nll.sum() / n_valid
            mean_comp = (compensator * valid_mask).sum() / n_valid
        else:
            nll = step_nll.mean()
            mean_comp = compensator.mean()

        return {
            "loss": nll,
            "compensator_integral": mean_comp.detach(),
            "mean_branching_ratio": out["branching_ratio"].mean().detach(),
        }

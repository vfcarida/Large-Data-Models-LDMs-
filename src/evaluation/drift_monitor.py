"""
Production Concept Drift Monitor & Online Adaptive Conformal Inference (ACI).

============================================================================
PURPOSE
============================================================================
Provides real-time statistical dataset/embedding shift detection and online
adaptive conformal recalibration for Large Data Models (LDMs) deployed on
non-stationary financial payment streams:

  1. Population Stability Index (`compute_psi`):
     - Industry-standard SR 11-7 / Basel III distributional shift metric
       computed over reference quantile bins (`PSI < 0.10` stable,
       `0.10 <= PSI < 0.25` moderate warning, `PSI >= 0.25` critical drift).
  2. Multi-Bandwidth RBF Maximum Mean Discrepancy (`compute_mmd_rbf`):
     - Non-parametric two-sample kernel test in the high-dimensional latent
       representation space (`z_joint`) following Rabanser et al. (NeurIPS
       2019, "Failing Loudly: An Empirical Study of Methods for Detecting
       Dataset Shift", arXiv:1810.11953).
  3. 1D Wasserstein / Earth Mover's Distance (`compute_wasserstein_1d`):
     - Quantile-based transport distance on predicted risk probability
       distributions and transaction amounts.
  4. Adaptive Conformal Inference (`AdaptiveConformalAbstainer`):
     - Implements Gibbs & Candes (NeurIPS 2021, "Adaptive Conformal Inference
       Under Distribution Shift", arXiv:2106.00170), updating the effective
       miscoverage level `alpha_t` online via:
           alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
       to maintain target coverage `(1 - alpha_target)` under adversarial or
       macroeconomic concept drift without retraining the backbone.
============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence
import math

import numpy as np
import torch

from src.evaluation.metrics import ConformalRiskAbstainer


@dataclass
class DriftReport:
    """
    Structured telemetry report from `ConceptDriftMonitor`.
    """

    max_feature_psi: float
    mean_feature_psi: float
    amount_stream_psi: float
    embedding_mmd: float
    probability_wasserstein: float
    status: Literal["STABLE", "WARNING", "CRITICAL_DRIFT"]
    feature_psi_breakdown: Dict[str, float] = field(default_factory=dict)


def compute_psi(
    reference: torch.Tensor,
    current: torch.Tensor,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Computes the Population Stability Index (PSI) between two 1D distributions.

    Args:
        reference: Baseline 1D tensor from validation/calibration window.
        current: Production 1D tensor from current monitoring window.
        num_bins: Number of reference quantile buckets (default: 10).
        eps: Laplace smoothing constant to avoid division by zero.

    Returns:
        Non-negative scalar PSI value.
    """
    ref_np = reference.detach().cpu().float().view(-1).numpy()
    cur_np = current.detach().cpu().float().view(-1).numpy()

    if ref_np.size < 2 or cur_np.size < 2:
        return 0.0

    quantiles = np.linspace(0.0, 100.0, num_bins + 1)
    bin_edges = np.percentile(ref_np, quantiles)
    # Ensure strictly increasing internal edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    unique_edges = np.unique(bin_edges)
    if unique_edges.size < 3:
        # Fallback for near-constant features
        min_v = float(min(ref_np.min(), cur_np.min())) - 1e-5
        max_v = float(max(ref_np.max(), cur_np.max())) + 1e-5
        unique_edges = np.linspace(min_v, max_v, num_bins + 1)

    ref_counts, _ = np.histogram(ref_np, bins=unique_edges)
    cur_counts, _ = np.histogram(cur_np, bins=unique_edges)

    ref_pct = np.clip(ref_counts / max(1, ref_np.size), eps, 1.0)
    cur_pct = np.clip(cur_counts / max(1, cur_np.size), eps, 1.0)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


def compute_wasserstein_1d(reference: torch.Tensor, current: torch.Tensor) -> float:
    """
    Computes the 1D Earth Mover's (Wasserstein-1) distance via empirical quantiles.
    """
    ref_np = np.sort(reference.detach().cpu().float().view(-1).numpy())
    cur_np = np.sort(current.detach().cpu().float().view(-1).numpy())
    if ref_np.size == 0 or cur_np.size == 0:
        return 0.0

    q_grid = np.linspace(0.0, 1.0, 101)
    ref_q = np.quantile(ref_np, q_grid)
    cur_q = np.quantile(cur_np, q_grid)
    return float(np.mean(np.abs(ref_q - cur_q)))


def compute_mmd_rbf(
    ref_embeddings: torch.Tensor,
    cur_embeddings: torch.Tensor,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
) -> float:
    """
    Computes multi-bandwidth RBF-kernel Maximum Mean Discrepancy (MMD)
    between reference and current latent representations `[N, D]` and `[M, D]`.
    """
    with torch.no_grad():
        x = ref_embeddings.detach().float()
        y = cur_embeddings.detach().float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        if x.size(0) == 0 or y.size(0) == 0:
            return 0.0

        dist_xx = torch.cdist(x, x, p=2.0).pow(2)
        dist_yy = torch.cdist(y, y, p=2.0).pow(2)
        dist_xy = torch.cdist(x, y, p=2.0).pow(2)

        # Median heuristic scaling factor
        median_sq = float(torch.median(dist_xy).item())
        scale = max(median_sq, 1e-3)

        mmd_sq_total = 0.0
        for bw in bandwidths:
            gamma = 1.0 / (2.0 * bw * scale)
            k_xx = torch.exp(-gamma * dist_xx).mean()
            k_yy = torch.exp(-gamma * dist_yy).mean()
            k_xy = torch.exp(-gamma * dist_xy).mean()
            mmd_sq_total += float((k_xx + k_yy - 2.0 * k_xy).clamp(min=0.0).item())

        mmd_sq_avg = mmd_sq_total / max(1, len(bandwidths))
        return float(math.sqrt(max(0.0, mmd_sq_avg)))


class ConceptDriftMonitor:
    """
    Multi-modal production drift detector monitoring static tabular features,
    streaming transaction amounts, latent LDM embeddings, and output risk scores.

    Args:
        psi_warning_threshold: PSI threshold triggering `"WARNING"` (default: `0.10`).
        psi_critical_threshold: PSI threshold triggering `"CRITICAL_DRIFT"` (default: `0.25`).
        mmd_warning_threshold: Latent MMD threshold triggering `"WARNING"` (default: `0.15`).
        mmd_critical_threshold: Latent MMD threshold triggering `"CRITICAL_DRIFT"` (default: `0.30`).
        feature_names: Optional list of static feature names for telemetry breakdown.
    """

    def __init__(
        self,
        psi_warning_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        mmd_warning_threshold: float = 0.15,
        mmd_critical_threshold: float = 0.30,
        feature_names: Optional[List[str]] = None,
    ):
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_critical_threshold = psi_critical_threshold
        self.mmd_warning_threshold = mmd_warning_threshold
        self.mmd_critical_threshold = mmd_critical_threshold
        self.feature_names = feature_names

        self.ref_tabular: Optional[torch.Tensor] = None
        self.ref_amounts: Optional[torch.Tensor] = None
        self.ref_embeddings: Optional[torch.Tensor] = None
        self.ref_probs: Optional[torch.Tensor] = None

    def fit_reference(
        self,
        tabular_features: torch.Tensor,
        transaction_amounts: torch.Tensor,
        latent_embeddings: torch.Tensor,
        predicted_probs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Stores the baseline reference distribution from validation/calibration data.
        """
        self.ref_tabular = tabular_features.detach().cpu().float()
        if padding_mask is not None:
            valid_amounts = transaction_amounts[~padding_mask]
        else:
            valid_amounts = transaction_amounts.reshape(-1)
        self.ref_amounts = valid_amounts.detach().cpu().float()
        self.ref_embeddings = latent_embeddings.detach().cpu().float()
        self.ref_probs = predicted_probs.detach().cpu().float().view(-1)

    def evaluate_window(
        self,
        tabular_features: torch.Tensor,
        transaction_amounts: torch.Tensor,
        latent_embeddings: torch.Tensor,
        predicted_probs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> DriftReport:
        """
        Evaluates statistical drift between the fitted reference distribution
        and the current production window.
        """
        if (
            self.ref_tabular is None
            or self.ref_amounts is None
            or self.ref_embeddings is None
            or self.ref_probs is None
        ):
            raise RuntimeError("Call fit_reference() before evaluate_window().")

        cur_tabular = tabular_features.detach().cpu().float()
        num_features = cur_tabular.size(1)

        feature_psi_breakdown: Dict[str, float] = {}
        for idx in range(num_features):
            fname = (
                self.feature_names[idx]
                if self.feature_names and idx < len(self.feature_names)
                else f"static_feature_{idx}"
            )
            feature_psi_breakdown[fname] = compute_psi(
                self.ref_tabular[:, idx], cur_tabular[:, idx]
            )

        psi_vals = list(feature_psi_breakdown.values())
        max_f_psi = float(max(psi_vals)) if psi_vals else 0.0
        mean_f_psi = float(np.mean(psi_vals)) if psi_vals else 0.0

        if padding_mask is not None:
            cur_amounts = transaction_amounts[~padding_mask].detach().cpu().float()
        else:
            cur_amounts = transaction_amounts.reshape(-1).detach().cpu().float()

        amount_psi = compute_psi(self.ref_amounts, cur_amounts)
        emb_mmd = compute_mmd_rbf(
            self.ref_embeddings, latent_embeddings.detach().cpu().float()
        )
        prob_w1 = compute_wasserstein_1d(
            self.ref_probs, predicted_probs.detach().cpu().float().view(-1)
        )

        peak_psi = max(max_f_psi, amount_psi)
        if peak_psi >= self.psi_critical_threshold or emb_mmd >= self.mmd_critical_threshold:
            status: Literal["STABLE", "WARNING", "CRITICAL_DRIFT"] = "CRITICAL_DRIFT"
        elif peak_psi >= self.psi_warning_threshold or emb_mmd >= self.mmd_warning_threshold:
            status = "WARNING"
        else:
            status = "STABLE"

        return DriftReport(
            max_feature_psi=max_f_psi,
            mean_feature_psi=mean_f_psi,
            amount_stream_psi=amount_psi,
            embedding_mmd=emb_mmd,
            probability_wasserstein=prob_w1,
            status=status,
            feature_psi_breakdown=feature_psi_breakdown,
        )


class AdaptiveConformalAbstainer(ConformalRiskAbstainer):
    """
    Online Adaptive Conformal Inference (ACI) Abstainer under Distribution Shift.

    Implements Gibbs & Candes (NeurIPS 2021) online conformal update rule:
        alpha_{t+1} = clip(alpha_t + gamma * (alpha_target - err_t), alpha_min, alpha_max)

    When empirical selective error `err_t` exceeds `alpha_target` during a concept
    drift episode, `alpha_t` decreases, automatically widening the abstention band
    `[lower_threshold, upper_threshold]` and routing ambiguous drifted transactions
    to human review until error returns to target level `alpha_target`.

    Args:
        alpha_target: Target long-run miscoverage / selective error rate (default: `0.10`).
        gamma: Online step size for ACI adaptation (default: `0.05`).
        alpha_min: Minimum allowable effective alpha (default: `0.01`).
        alpha_max: Maximum allowable effective alpha (default: `0.45`).
    """

    def __init__(
        self,
        alpha_target: float = 0.10,
        gamma: float = 0.05,
        alpha_min: float = 0.01,
        alpha_max: float = 0.45,
    ):
        super().__init__(alpha=alpha_target)
        self.alpha_target = alpha_target
        self.effective_alpha = alpha_target
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._calib_logits: Optional[torch.Tensor] = None
        self._calib_targets: Optional[torch.Tensor] = None
        self.alpha_history: List[float] = [alpha_target]

    def calibrate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        self._calib_logits = logits.detach().clone()
        self._calib_targets = targets.detach().clone()
        self.alpha = self.effective_alpha
        return super().calibrate(self._calib_logits, self._calib_targets)

    def update_online(
        self,
        window_logits: torch.Tensor,
        window_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Executes one online ACI feedback step from delayed ground-truth labels
        and recalibrates `[lower_threshold, upper_threshold]`.

        Args:
            window_logits: Model logits `[N, 1]` or `[N]` on recent labeled window.
            window_targets: Ground-truth binary labels `[N]` on recent window.

        Returns:
            Dictionary with `observed_error_rate`, `effective_alpha`,
            `lower_threshold`, and `upper_threshold`.
        """
        with torch.no_grad():
            decisions = self.predict_with_abstention(window_logits)
            y_true = window_targets.view(-1).long()
            active_mask = decisions != -1

            if active_mask.any():
                active_preds = decisions[active_mask]
                active_targets = y_true[active_mask]
                err_t = float((active_preds != active_targets).float().mean().item())
            else:
                err_t = 0.0

            # Gibbs & Candes (2021) online step:
            # If err_t > alpha_target, effective_alpha decreases -> wider abstention band
            new_alpha = self.effective_alpha + self.gamma * (self.alpha_target - err_t)
            self.effective_alpha = float(
                max(self.alpha_min, min(self.alpha_max, new_alpha))
            )
            self.alpha = self.effective_alpha
            self.alpha_history.append(self.effective_alpha)

            # Update rolling calibration buffer and recompute quantiles
            if self._calib_logits is not None and self._calib_targets is not None:
                combined_logits = torch.cat(
                    [self._calib_logits.view(-1), window_logits.detach().view(-1)],
                    dim=0,
                )[-512:]
                combined_targets = torch.cat(
                    [self._calib_targets.view(-1), window_targets.detach().view(-1)],
                    dim=0,
                )[-512:]
            else:
                combined_logits = window_logits.detach().view(-1)
                combined_targets = window_targets.detach().view(-1)

            super().calibrate(combined_logits, combined_targets)
            self._calib_logits = combined_logits
            self._calib_targets = combined_targets

            return {
                "observed_error_rate": err_t,
                "effective_alpha": self.effective_alpha,
                "lower_threshold": float(self.lower_threshold),
                "upper_threshold": float(self.upper_threshold),
            }

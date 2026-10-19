r"""
Continual Lifelong Adaptation with Elastic Weight Consolidation (`EWC`) & Dark Experience Distillation (`DER++`).

============================================================================
RESEARCH & ARCHITECTURAL ANCHORS
============================================================================
Financial payment streams undergo non-stationary seasonal regimes (holiday
spending surges, new instant-payment rails, evolving fraud rings). Naive
fine-tuning on recent windows causes catastrophic forgetting of historical
rare fraud signatures.

Synthesizes two continual learning paradigms:
  1. Elastic Weight Consolidation (`EWC`, Kirkpatrick et al., PNAS 2017,
     "Overcoming catastrophic forgetting in neural networks",
     DOI: 10.1073/pnas.1611835114):
     - Penalizes parameter drift weighted by the online empirical diagonal
       Fisher Information Matrix $\Omega_i$ accumulated across prior regimes:
       $$\mathcal{L}_{\text{EWC}}(\theta) = \frac{\lambda_{\text{EWC}}}{2} \sum_i \Omega_i (\theta_i - \theta^*_{i,\text{old}})^2$$
  2. Dark Experience Replay / Functional Distillation (`DER++`, Buzzega et al.,
     NeurIPS 2020, "Dark Experience for General Continual Learning",
     arXiv:2004.07211):
     - Matches the pre-adaptation model's soft logit trajectory $z_{\text{old}}(x)$
       via $\alpha_{\text{DER}} \| z_\theta(x) - z_{\text{old}}(x) \|_2^2$ to preserve
       calibrated dark knowledge across task boundaries.
============================================================================
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ldm_model import LargeDataModel


@dataclass(frozen=True)
class ContinualAdaptationReport:
    """
    Telemetry report emitted after a continual regime adaptation cycle.
    """

    num_consolidated_regimes: int
    new_regime_loss_before: float
    new_regime_loss_after: float
    historical_regime_loss_before: float
    historical_regime_loss_after: float
    ewc_penalty: float
    der_distillation_loss: float
    parameter_l2_drift: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "num_consolidated_regimes": self.num_consolidated_regimes,
            "new_regime_loss_before": round(self.new_regime_loss_before, 6),
            "new_regime_loss_after": round(self.new_regime_loss_after, 6),
            "historical_regime_loss_before": round(self.historical_regime_loss_before, 6),
            "historical_regime_loss_after": round(self.historical_regime_loss_after, 6),
            "ewc_penalty": round(self.ewc_penalty, 6),
            "der_distillation_loss": round(self.der_distillation_loss, 6),
            "parameter_l2_drift": round(self.parameter_l2_drift, 6),
        }


class ContinualLDMAdapter:
    r"""
    Manages lifelong regime adaptation for a `LargeDataModel` using online
    Elastic Weight Consolidation (`Online EWC`) and Dark Experience functional
    logit distillation (`DER++`).

    Args:
        model: `LargeDataModel` instance to adapt across sequential regimes.
        ewc_lambda: Quadratic Fisher importance regularization weight $\lambda_{\text{EWC}}$
            (default: `25.0`).
        der_alpha: Functional logit-distillation weight $\alpha_{\text{DER}}$
            (default: `0.50`).
        online_gamma: Decay factor $\gamma \in (0, 1]$ for online Fisher accumulation
            across multiple historical regimes (default: `0.90`).
        fisher_damping: Minimum diagonal floor for numerical stability (default: `1e-5`).
    """

    def __init__(
        self,
        model: LargeDataModel,
        ewc_lambda: float = 25.0,
        der_alpha: float = 0.50,
        online_gamma: float = 0.90,
        fisher_damping: float = 1e-5,
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.der_alpha = der_alpha
        self.online_gamma = online_gamma
        self.fisher_damping = fisher_damping

        self.num_consolidated_regimes: int = 0
        self.anchor_params: Dict[str, torch.Tensor] = {}
        self.fisher_omega: Dict[str, torch.Tensor] = {}

    def _forward_batch_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        keys = batch["packed_keys"].to(device)
        vals = batch["packed_values"].to(device)
        times = batch["packed_times"].to(device)
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(device)
        pad = batch.get("key_padding_mask")
        if pad is not None:
            pad = pad.to(device)

        return self.model(
            keys,
            vals,
            times,
            tabular_features=tab,
            key_padding_mask=pad,
            update_memory=False,
        )

    def _compute_batch_bce(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        logits = self._forward_batch_logits(batch)
        labels = batch["labels"].to(device).float().view(-1, 1)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def consolidate_regime(self, regime_batch: Dict[str, torch.Tensor]) -> None:
        r"""
        Computes the empirical diagonal Fisher Information Matrix on `regime_batch`
        and updates the online EWC anchor parameters $\theta^*$ and importance
        weights $\Omega \leftarrow \gamma \Omega_{\text{old}} + F_{\text{current}}$.
        """
        self.model.eval()
        self.model.zero_grad()

        named_trainable = [
            (name, p) for name, p in self.model.named_parameters() if p.requires_grad
        ]
        params = [p for _, p in named_trainable]

        loss = self._compute_batch_bce(regime_batch)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

        with torch.no_grad():
            for (name, p), g in zip(named_trainable, grads):
                f_curr = (
                    g.detach().pow(2) if g is not None else torch.zeros_like(p)
                ) + self.fisher_damping
                # Normalize per-tensor Fisher scale to avoid gradient explosion
                max_f = float(f_curr.max().item())
                if max_f > 1.0:
                    f_curr = f_curr / max_f

                if name in self.fisher_omega:
                    self.fisher_omega[name] = (
                        self.online_gamma * self.fisher_omega[name] + f_curr
                    )
                else:
                    self.fisher_omega[name] = f_curr.clone()

                self.anchor_params[name] = p.detach().clone()

        self.num_consolidated_regimes += 1

    def compute_ewc_penalty(self) -> torch.Tensor:
        r"""
        Computes $\frac{\lambda_{\text{EWC}}}{2} \sum_i \Omega_i (\theta_i - \theta^*_i)^2$.
        """
        device = next(self.model.parameters()).device
        if not self.anchor_params:
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=device)
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.anchor_params:
                diff = p - self.anchor_params[name]
                omega = self.fisher_omega[name]
                penalty = penalty + 0.5 * self.ewc_lambda * (omega * diff.pow(2)).sum()
        return penalty

    def adapt_to_new_regime(
        self,
        new_regime_batch: Dict[str, torch.Tensor],
        historical_probe_batch: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = 5,
        lr: float = 2e-3,
    ) -> ContinualAdaptationReport:
        """
        Adapts the `LargeDataModel` to `new_regime_batch` while preventing
        catastrophic forgetting on prior consolidated regimes via `EWC` + `DER++`.

        Args:
            new_regime_batch: Batch from the newly arrived payment/fraud regime.
            historical_probe_batch: Optional batch from prior regime(s) for
                dark-experience logit matching and retention evaluation.
            num_steps: Number of continual optimization steps (default: `5`).
            lr: Adaptation learning rate (default: `2e-3`).

        Returns:
            `ContinualAdaptationReport` with before/after loss and drift metrics.
        """
        probe_batch = (
            historical_probe_batch
            if historical_probe_batch is not None
            else new_regime_batch
        )

        self.model.eval()
        with torch.no_grad():
            new_loss_before = float(self._compute_batch_bce(new_regime_batch).item())
            hist_loss_before = float(self._compute_batch_bce(probe_batch).item())
            # Capture pre-adaptation dark experience logits
            dark_logits_target = self._forward_batch_logits(probe_batch).detach().clone()
            initial_weights = [
                p.detach().clone() for p in self.model.parameters() if p.requires_grad
            ]

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )

        last_ewc = 0.0
        last_der = 0.0

        for _ in range(num_steps):
            optimizer.zero_grad()
            loss_new = self._compute_batch_bce(new_regime_batch)
            loss_ewc = self.compute_ewc_penalty()

            current_probe_logits = self._forward_batch_logits(probe_batch)
            loss_der = self.der_alpha * F.mse_loss(
                current_probe_logits, dark_logits_target
            )

            total_loss = loss_new + loss_ewc + loss_der
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            last_ewc = float(loss_ewc.detach().item())
            last_der = float(loss_der.detach().item())

        self.model.eval()
        with torch.no_grad():
            new_loss_after = float(self._compute_batch_bce(new_regime_batch).item())
            hist_loss_after = float(self._compute_batch_bce(probe_batch).item())
            final_weights = [
                p.detach() for p in self.model.parameters() if p.requires_grad
            ]
            l2_drift = math.sqrt(
                sum(
                    float((p - p0).float().norm(2).pow(2).item())
                    for p, p0 in zip(final_weights, initial_weights)
                )
            )

        return ContinualAdaptationReport(
            num_consolidated_regimes=self.num_consolidated_regimes,
            new_regime_loss_before=new_loss_before,
            new_regime_loss_after=new_loss_after,
            historical_regime_loss_before=hist_loss_before,
            historical_regime_loss_after=hist_loss_after,
            ewc_penalty=last_ewc,
            der_distillation_loss=last_der,
            parameter_l2_drift=l2_drift,
        )

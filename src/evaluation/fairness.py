"""
Algorithmic Fairness Governance Auditor & Equalized-Odds Post-Processor.

============================================================================
PURPOSE
============================================================================
Under U.S. Equal Credit Opportunity Act (ECOA / Regulation B), CFPB fair
lending guidance, and the EU AI Act (Annex III High-Risk Credit & Fraud
Scoring), transactional decisioning models must continuously audit and
mitigate disparate impact across protected demographic groups.

This module implements:
  1. `FairnessGovernanceAuditor`:
     - Computes multi-group fairness metrics with respect to favorable
       outcomes (legitimate approval `Y_hat = 0`) and adverse outcomes
       (fraud flag `Y_hat = 1`):
       a) Demographic Parity Difference (`DPD`): maximum absolute difference
          in approval rates across demographic groups.
       b) Disparate Impact Ratio (`DIR` — EEOC / ECOA 80% Four-Fifths Rule):
          ratio of the minimum group approval rate to the maximum group
          approval rate (`DIR >= 0.80` indicates compliance).
       c) Equalized Odds Difference (`EOD` — Hardt et al., NeurIPS 2016):
          `max(|TPR_g1 - TPR_g2|, |FPR_g1 - FPR_g2|)` across groups.
  2. `EqualizedOddsThresholdOptimizer` (Hardt, Price, & Srebro, NeurIPS 2016,
     "Equality of Opportunity in Supervised Learning", arXiv:1610.02413):
     - Calibrates group-specific probability thresholds `tau_g` on validation
       predictions to equalize False Positive Rates (legitimate customers
       wrongly declined) and/or Selection Rates across cohorts without
       modifying neural network weights.
============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import torch


@dataclass
class FairnessReport:
    """
    Structured regulatory fairness report across protected demographic cohorts.
    """

    demographic_parity_diff: float
    disparate_impact_ratio: float
    equalized_odds_diff: float
    max_fpr_gap: float
    max_tpr_gap: float
    passes_four_fifths_rule: bool
    group_approval_rates: Dict[int, float] = field(default_factory=dict)
    group_fpr: Dict[int, float] = field(default_factory=dict)
    group_tpr: Dict[int, float] = field(default_factory=dict)


class FairnessGovernanceAuditor:
    """
    Audits binary fraud/credit predictions across protected group attributes.

    Note on convention:
      - `targets == 1` denotes Fraud / Default (adverse ground truth).
      - `targets == 0` denotes Legitimate / Creditworthy (favorable ground truth).
      - Approval (`preds == 0`) is the favorable decision for ECOA Disparate
        Impact Ratio (`DIR = min_g P(Y_hat=0 | G=g) / max_g P(Y_hat=0 | G=g)`).
    """

    def __init__(self, four_fifths_threshold: float = 0.80):
        self.four_fifths_threshold = four_fifths_threshold

    @torch.no_grad()
    def audit_predictions(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> FairnessReport:
        """
        Computes DPD, DIR, and EOD from binary predictions `[N]`, ground-truth
        labels `[N]`, and integer demographic group identifiers `[N]`.
        """
        p = preds.detach().cpu().view(-1).long()
        y = targets.detach().cpu().view(-1).long()
        g = group_ids.detach().cpu().view(-1).long()

        unique_groups = sorted([int(u.item()) for u in torch.unique(g)])
        approval_rates: Dict[int, float] = {}
        fpr_map: Dict[int, float] = {}
        tpr_map: Dict[int, float] = {}

        for gid in unique_groups:
            mask_g = g == gid
            if not mask_g.any():
                continue
            p_g = p[mask_g]
            y_g = y[mask_g]

            # Favorable decision is approval (p_g == 0)
            approval_rates[gid] = float((p_g == 0).float().mean().item())

            neg_mask = y_g == 0
            pos_mask = y_g == 1
            if neg_mask.any():
                fpr_map[gid] = float((p_g[neg_mask] == 1).float().mean().item())
            else:
                fpr_map[gid] = 0.0

            if pos_mask.any():
                tpr_map[gid] = float((p_g[pos_mask] == 1).float().mean().item())
            else:
                tpr_map[gid] = 0.0

        if len(approval_rates) >= 2:
            app_vals = list(approval_rates.values())
            min_app = min(app_vals)
            max_app = max(app_vals)
            dpd = float(max_app - min_app)
            dir_ratio = float(min_app / max(max_app, 1e-6))

            fpr_vals = list(fpr_map.values())
            tpr_vals = list(tpr_map.values())
            max_fpr_gap = float(max(fpr_vals) - min(fpr_vals))
            max_tpr_gap = float(max(tpr_vals) - min(tpr_vals))
            eod = max(max_fpr_gap, max_tpr_gap)
        else:
            dpd = 0.0
            dir_ratio = 1.0
            max_fpr_gap = 0.0
            max_tpr_gap = 0.0
            eod = 0.0

        return FairnessReport(
            demographic_parity_diff=dpd,
            disparate_impact_ratio=dir_ratio,
            equalized_odds_diff=eod,
            max_fpr_gap=max_fpr_gap,
            max_tpr_gap=max_tpr_gap,
            passes_four_fifths_rule=bool(dir_ratio >= self.four_fifths_threshold),
            group_approval_rates=approval_rates,
            group_fpr=fpr_map,
            group_tpr=tpr_map,
        )

    @torch.no_grad()
    def audit_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        threshold: float = 0.5,
    ) -> FairnessReport:
        probs = torch.sigmoid(logits.float()).view(-1)
        preds = (probs >= threshold).long()
        return self.audit_predictions(preds, targets, group_ids)


class EqualizedOddsThresholdOptimizer:
    """
    Post-Processing Group Threshold Optimizer (Hardt, Price, & Srebro, NeurIPS 2016).

    Calibrates group-specific probability thresholds `tau_g` so that every
    protected group achieves a matched target approval rate or target False
    Positive Rate (FPR), eliminating disparate impact without retraining.

    Args:
        default_threshold: Fallback threshold for unseen groups (default: `0.5`).
        target_fpr: Desired maximum False Positive Rate on legitimate users
            (default: `0.10`).
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        target_fpr: float = 0.10,
    ):
        self.default_threshold = default_threshold
        self.target_fpr = target_fpr
        self.group_thresholds: Dict[int, float] = {}

    @torch.no_grad()
    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Dict[int, float]:
        """
        Calibrates `group_thresholds` on validation data to align both group
        False Positive Rates and group approval rates.
        """
        probs = torch.sigmoid(logits.detach().cpu().float()).view(-1)
        y = targets.detach().cpu().view(-1).long()
        g = group_ids.detach().cpu().view(-1).long()

        unique_groups = sorted([int(u.item()) for u in torch.unique(g)])
        # Target overall approval quantile `(1 - target_adverse_rate)`
        overall_adverse_rate = float(
            max(0.05, min(0.50, (probs >= self.default_threshold).float().mean().item()))
        )

        for gid in unique_groups:
            mask_g = g == gid
            probs_g = probs[mask_g]
            y_g = y[mask_g]
            if probs_g.numel() == 0:
                self.group_thresholds[gid] = self.default_threshold
                continue

            neg_probs = probs_g[y_g == 0]
            if neg_probs.numel() >= 3:
                # Threshold achieving target_fpr on legitimate users in group g
                q_fpr = max(0.05, min(0.95, 1.0 - self.target_fpr))
                t_fpr = float(np.quantile(neg_probs.numpy(), q_fpr))
            else:
                t_fpr = self.default_threshold

            # Threshold matching overall selection/approval rate across groups
            q_parity = max(0.05, min(0.95, 1.0 - overall_adverse_rate))
            t_parity = float(np.quantile(probs_g.numpy(), q_parity))

            # Blend FPR-equalizing and demographic-parity thresholds
            tau_g = 0.5 * t_fpr + 0.5 * t_parity
            self.group_thresholds[gid] = float(max(0.05, min(0.95, tau_g)))

        return dict(self.group_thresholds)

    @torch.no_grad()
    def predict(
        self,
        logits: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produces binary predictions `[N]` using calibrated per-group thresholds.
        """
        probs = torch.sigmoid(logits.float()).view(-1)
        g = group_ids.view(-1).long()
        preds = torch.zeros_like(probs, dtype=torch.long)

        for idx in range(probs.numel()):
            gid = int(g[idx].item())
            tau = self.group_thresholds.get(gid, self.default_threshold)
            preds[idx] = 1 if float(probs[idx].item()) >= tau else 0

        return preds

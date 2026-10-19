"""
Evaluation Metrics, Calibration Auditing, and Conformal Risk Abstention.

============================================================================
PURPOSE
============================================================================
Standard accuracy is misleading for financial risk and fraud detection models
operating under severe class imbalance (0.5%-5% positive rate).

This module implements a comprehensive evaluation and governance suite aligned
with academic best practices and banking Model Risk Management (SR 11-7 /
SR 26-2 and Gartner AI TRiSM):

1. Discrimination Metrics under Imbalance:
   - `calculate_auc`: Area Under the ROC Curve (ROC-AUC)
   - `calculate_pr_auc`: Area Under the Precision-Recall Curve (Average Precision)
   - `calculate_recall_at_fpr`: Fraud Recall at a fixed False Positive Rate budget
   - `calculate_f1_flexible` & `calculate_precision_recall`

2. Epistemic Calibration & Probability Reliability:
   - `calculate_brier_score`: Mean squared probability error
   - `calculate_ece`: Expected Calibration Error across confidence bins

3. Conformal Prediction & Human-in-the-Loop Risk Abstention:
   - `ConformalRiskAbstainer`: Calibrates finite-sample non-conformity quantiles
     to abstain (`-1`) on ambiguous predictions and route them to analyst review.

4. Self-Supervised Imputation Quality:
   - `calculate_imputation_rmse`: Reconstruction RMSE on masked positions.

============================================================================
REFERENCES
============================================================================
  - Saito & Rehmsmeier (2015): "The Precision-Recall Plot Is More Informative
    than the ROC Plot when Evaluating Binary Classifiers on Imbalanced Datasets"
  - Guo et al. (2017): "On Calibration of Modern Neural Networks" (ICML 2017)
  - Angelopoulos & Bates (2023): "Conformal Prediction: A Gentle Introduction"
============================================================================
"""

from typing import Any, Dict, Optional
import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def calculate_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates Area Under the ROC Curve (ROC-AUC) via exact Mann-Whitney U statistic.

    Args:
        logits: Model predictions (pre-sigmoid) `[N]` or `[N, 1]`.
        targets: Ground truth binary labels (`0` or `1`) `[N]` or `[N, 1]`.

    Returns:
        ROC-AUC score in `[0, 1]`. Returns `0.5` if only one class is present.
    """
    with torch.no_grad():
        preds = torch.sigmoid(logits.float()).view(-1)
        targets_flat = targets.view(-1)

        pos_preds = preds[targets_flat == 1]
        neg_preds = preds[targets_flat == 0]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            logger.warning("AUC undefined: only one class present. Returning 0.5")
            return 0.5

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
        concordant = (diff > 0).float().sum()
        ties = (diff == 0).float().sum() * 0.5

        auc = (concordant + ties) / (len(pos_preds) * len(neg_preds))
        return float(auc.item())


def calculate_pr_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates Area Under the Precision-Recall Curve (Average Precision / PR-AUC).

    Follows Saito & Rehmsmeier (2015): PR-AUC is the primary discrimination
    metric under severe class imbalance because it directly penalizes false
    positives relative to the rare positive class.

    Args:
        logits: Model predictions (pre-sigmoid) `[N]` or `[N, 1]`.
        targets: Ground truth binary labels (`0` or `1`) `[N]` or `[N, 1]`.

    Returns:
        PR-AUC (Average Precision) in `[0, 1]`.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.float()).view(-1)
        y_true = targets.view(-1).float()

        num_pos = y_true.sum().item()
        if num_pos == 0:
            return 0.0

        sorted_indices = torch.argsort(probs, descending=True)
        y_sorted = y_true[sorted_indices]

        tp_cumsum = torch.cumsum(y_sorted, dim=0)
        ranks = torch.arange(1, len(y_sorted) + 1, device=probs.device, dtype=torch.float32)
        precisions = tp_cumsum / ranks

        ap = (precisions * y_sorted).sum().item() / num_pos
        return float(ap)


def calculate_recall_at_fpr(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_fpr: float = 0.05,
) -> float:
    """
    Calculates maximum Recall achieved while keeping False Positive Rate <= `max_fpr`.

    In banking fraud operations (BCG / McKinsey / SR 11-7), investigation
    teams operate under strict false-positive alert caps (e.g., 1% or 5% FPR).

    Args:
        logits: Model predictions (pre-sigmoid).
        targets: Ground truth binary labels (`0` or `1`).
        max_fpr: Maximum permissible False Positive Rate in `(0, 1)`.

    Returns:
        Recall at `FPR <= max_fpr` in `[0, 1]`.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.float()).view(-1)
        y_true = targets.view(-1).float()

        num_pos = y_true.sum().item()
        num_neg = (1.0 - y_true).sum().item()
        if num_pos == 0 or num_neg == 0:
            return 0.0

        sorted_indices = torch.argsort(probs, descending=True)
        y_sorted = y_true[sorted_indices]

        tp_cumsum = torch.cumsum(y_sorted, dim=0)
        fp_cumsum = torch.cumsum(1.0 - y_sorted, dim=0)

        fprs = fp_cumsum / num_neg
        recalls = tp_cumsum / num_pos

        valid = fprs <= max_fpr
        if not valid.any():
            return 0.0
        return float(recalls[valid].max().item())


def calculate_brier_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the Brier Score (mean squared error of predicted probabilities).

    Brier = (1 / N) * sum_i (sigmoid(logit_i) - y_i)^2

    Lower values indicate better calibrated and more accurate probability estimates.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.float()).view(-1)
        y_true = targets.view(-1).float()
        if probs.numel() == 0:
            return 0.0
        return float(F.mse_loss(probs, y_true).item())


def calculate_ece(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Calculates Expected Calibration Error (ECE) across `n_bins` confidence intervals.

    Follows Guo et al. (ICML 2017, "On Calibration of Modern Neural Networks"):
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        logits: Model predictions (pre-sigmoid).
        targets: Ground truth binary labels (`0` or `1`).
        n_bins: Number of uniform probability bins in `[0, 1]`.

    Returns:
        Expected Calibration Error in `[0, 1]`.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.float()).view(-1)
        y_true = targets.view(-1).float()
        n = probs.numel()
        if n == 0:
            return 0.0

        bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
        ece = 0.0

        for i in range(n_bins):
            low = bin_boundaries[i]
            high = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (probs >= low) & (probs <= high)
            else:
                in_bin = (probs >= low) & (probs < high)

            prop_in_bin = in_bin.float().mean().item()
            if prop_in_bin > 0.0:
                avg_confidence = probs[in_bin].mean().item()
                avg_empirical_pos = y_true[in_bin].mean().item()
                ece += prop_in_bin * abs(avg_empirical_pos - avg_confidence)

        return float(ece)


def calculate_f1_flexible(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Calculates F1-Score with a configurable classification threshold.

    Args:
        logits: Model predictions (pre-sigmoid).
        targets: Ground truth binary labels (`0` or `1`).
        threshold: Probability threshold in `(0, 1)`.

    Returns:
        F1 score in `[0, 1]`.
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits.float()).view(-1) >= threshold).float()
        targets_flat = targets.view(-1).float()

        tp = (preds * targets_flat).sum()
        fp = (preds * (1.0 - targets_flat)).sum()
        fn = ((1.0 - preds) * targets_flat).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2.0 * (precision * recall) / (precision + recall + 1e-8)
        return float(f1.item())


def calculate_precision_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculates Precision, Recall, Specificity, Accuracy, and confusion matrix counts.

    Args:
        logits: Model predictions (pre-sigmoid).
        targets: Ground truth labels.
        threshold: Classification threshold.

    Returns:
        Dictionary with precision, recall, specificity, accuracy, tp, fp, fn, tn.
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits.float()).view(-1) >= threshold).float()
        targets_flat = targets.view(-1).float()

        tp = (preds * targets_flat).sum().item()
        fp = (preds * (1.0 - targets_flat)).sum().item()
        fn = ((1.0 - preds) * targets_flat).sum().item()
        tn = ((1.0 - preds) * (1.0 - targets_flat)).sum().item()

        return {
            "precision": float(tp / (tp + fp + 1e-8)),
            "recall": float(tp / (tp + fn + 1e-8)),
            "specificity": float(tn / (tn + fp + 1e-8)),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn + 1e-8)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }


def calculate_imputation_rmse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Calculates RMSE for missing data imputation over masked positions.

    Args:
        preds: Model predicted values.
        targets: Original ground truth values.
        mask: Boolean mask (`True` = position was masked).

    Returns:
        RMSE over masked positions (`0.0` if no positions masked).
    """
    with torch.no_grad():
        if mask.sum() == 0:
            return 0.0

        masked_preds = preds[mask]
        masked_targets = targets[mask]

        mse = F.mse_loss(masked_preds, masked_targets)
        return float(torch.sqrt(mse).item())


class ConformalRiskAbstainer:
    """
    Split-Conformal Uncertainty Guardrail & Selective Risk Abstainer.

    Implements distribution-free conformal abstention (Angelopoulos & Bates,
    2023) aligned with Gartner AI TRiSM runtime enforcement and SR 11-7
    human-in-the-loop escalation:
      - Classifies as `1` (Fraud) when predicted probability >= `upper_threshold`
      - Classifies as `0` (Legitimate) when predicted probability <= `lower_threshold`
      - Returns `-1` (`ABSTAIN` -> route to human analyst) when the prediction
        falls within the uncertain band `(lower_threshold, upper_threshold)`.

    Args:
        alpha: Target miscoverage / risk tolerance level in `(0, 0.5)` (default: `0.10`).
        default_lower: Fallback lower threshold before calibration.
        default_upper: Fallback upper threshold before calibration.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        default_lower: float = 0.30,
        default_upper: float = 0.70,
    ):
        if not (0.0 < alpha < 0.5):
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
        self.alpha = alpha
        self.lower_threshold = default_lower
        self.upper_threshold = default_upper
        self.is_calibrated = False

    def calibrate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calibrates conformal decision boundaries on validation logits and labels.

        Computes class-conditional finite-sample quantiles of predicted
        probabilities so that automated approvals (`<= lower_threshold`) and
        automated blocks (`>= upper_threshold`) control false-negative and
        false-positive error rates at level `alpha`.
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits.float()).view(-1)
            y_true = targets.view(-1).long()

            pos_probs = probs[y_true == 1]
            neg_probs = probs[y_true == 0]

            if len(pos_probs) > 0 and len(neg_probs) > 0:
                n_pos = len(pos_probs)
                n_neg = len(neg_probs)
                # Finite-sample conformal quantile levels
                q_pos = max(0.0, min(1.0, math.floor(n_pos * self.alpha) / max(1, n_pos)))
                q_neg = max(
                    0.0,
                    min(1.0, math.ceil((n_neg + 1) * (1.0 - self.alpha)) / max(1, n_neg)),
                )

                raw_lower = float(torch.quantile(pos_probs, q_pos).item())
                raw_upper = float(torch.quantile(neg_probs, q_neg).item())

                # Ensure a well-ordered abstention band [lower_threshold, upper_threshold]
                mid = 0.5 * (raw_lower + raw_upper)
                self.lower_threshold = min(raw_lower, mid - 0.05)
                self.upper_threshold = max(raw_upper, mid + 0.05)
                self.lower_threshold = max(0.05, min(0.48, self.lower_threshold))
                self.upper_threshold = max(0.52, min(0.95, self.upper_threshold))
                self.is_calibrated = True

        return {
            "lower_threshold": float(self.lower_threshold),
            "upper_threshold": float(self.upper_threshold),
        }

    def predict_with_abstention(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Produces ternary decisions: `1` (Fraud), `0` (Legitimate), `-1` (Abstain).
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits.float()).view(-1)
            decisions = torch.full_like(probs, fill_value=-1, dtype=torch.long)
            decisions[probs <= self.lower_threshold] = 0
            decisions[probs >= self.upper_threshold] = 1
            return decisions

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Evaluates selective classification metrics outside the abstention zone.

        Returns:
            Dictionary with `coverage`, `abstention_rate`, `selective_accuracy`,
            `selective_f1`, `lower_threshold`, and `upper_threshold`.
        """
        with torch.no_grad():
            decisions = self.predict_with_abstention(logits)
            y_true = targets.view(-1).long()

            active_mask = decisions != -1
            coverage = float(active_mask.float().mean().item()) if decisions.numel() > 0 else 0.0
            abstention_rate = 1.0 - coverage

            if active_mask.any():
                active_preds = decisions[active_mask].float()
                active_targets = y_true[active_mask].float()
                selective_acc = float((active_preds == active_targets).float().mean().item())

                tp = (active_preds * active_targets).sum().item()
                fp = (active_preds * (1.0 - active_targets)).sum().item()
                fn = ((1.0 - active_preds) * active_targets).sum().item()
                sel_f1 = float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
            else:
                selective_acc = 1.0
                sel_f1 = 0.0

            return {
                "conformal_coverage": coverage,
                "conformal_abstention_rate": abstention_rate,
                "conformal_selective_accuracy": selective_acc,
                "conformal_selective_f1": sel_f1,
                "conformal_lower_threshold": float(self.lower_threshold),
                "conformal_upper_threshold": float(self.upper_threshold),
            }


class RiskMetricsEvaluator:
    """
    Comprehensive discrimination, calibration, and conformal risk evaluator.

    Args:
        threshold: Default probability threshold for point classification metrics.
        conformal_alpha: Significance level for conformal risk abstention.
    """

    def __init__(self, threshold: float = 0.5, conformal_alpha: float = 0.10):
        self.threshold = threshold
        self.abstainer = ConformalRiskAbstainer(alpha=conformal_alpha)

    def evaluate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes all discrimination, calibration, and abstention metrics.
        """
        t = threshold if threshold is not None else self.threshold

        if not self.abstainer.is_calibrated:
            self.abstainer.calibrate(logits, targets)

        results: Dict[str, Any] = {
            "auc": calculate_auc(logits, targets),
            "pr_auc": calculate_pr_auc(logits, targets),
            "f1_score": calculate_f1_flexible(logits, targets, t),
            "recall_at_5pct_fpr": calculate_recall_at_fpr(logits, targets, max_fpr=0.05),
            "brier_score": calculate_brier_score(logits, targets),
            "ece": calculate_ece(logits, targets),
        }
        results.update(calculate_precision_recall(logits, targets, t))
        results.update(self.abstainer.evaluate(logits, targets))

        return results

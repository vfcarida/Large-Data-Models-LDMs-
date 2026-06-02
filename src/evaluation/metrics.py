"""
Evaluation Metrics for Risk Models and Imbalanced Classification.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
Standard accuracy is DANGEROUS for financial risk models. Consider:
  - If 1% of transactions are fraudulent
  - A model that predicts "not fraud" for EVERYTHING gets 99% accuracy
  - But catches ZERO fraud — completely useless!

This module implements metrics that properly evaluate performance
under severe class imbalance:

1. AUC (Area Under ROC Curve):
   - Threshold-independent: evaluates discrimination ability
   - 0.5 = random, 1.0 = perfect separation
   - Preferred metric for imbalanced binary classification

2. F1-Score (with flexible threshold):
   - Harmonic mean of Precision and Recall
   - Tunable threshold allows risk/reward tradeoff
   - Lower threshold → catch more fraud (but more false alarms)
   - Higher threshold → fewer false alarms (but miss more fraud)

3. Precision & Recall:
   - Precision: "Of all fraud alerts, how many were real fraud?"
   - Recall: "Of all real fraud, how many did we catch?"
   - Business decision: which error is more costly?

4. RMSE for Imputation:
   - Evaluates the model's ability to reconstruct missing data
   - Critical for data quality and pre-training evaluation

============================================================================
REFERENCES
============================================================================
  - V4FinBench: Extreme imbalance in corporate bankruptcy prediction
  - Saito & Rehmsmeier (2015): "The Precision-Recall Plot Is More
    Informative than the ROC Plot" for imbalanced data
============================================================================
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates AUC (Area Under the ROC Curve).

    The AUC measures how well the model separates positive from negative
    examples, REGARDLESS of the classification threshold. It equals the
    probability that a randomly chosen positive example is scored higher
    than a randomly chosen negative example.

    Implementation: Concordant pairs approach
    - For each (positive, negative) pair, check if positive > negative
    - AUC = (concordant + 0.5 * ties) / total_pairs

    Note: This implementation is O(N*M) in memory where N and M are the
    number of positive and negative examples. For very large datasets,
    use torchmetrics.AUROC instead.

    Args:
        logits: Model predictions (pre-sigmoid) [N] or [N, 1]
        targets: Ground truth labels (0 or 1) [N] or [N, 1]

    Returns:
        AUC score in [0, 1]. Returns 0.5 if only one class is present.
    """
    with torch.no_grad():
        preds = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)

        pos_preds = preds[targets == 1]
        neg_preds = preds[targets == 0]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            logger.warning("AUC undefined: only one class present. Returning 0.5")
            return 0.5

        # Concordant pairs matrix
        # For small datasets this is fine; for large ones use torchmetrics
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
        concordant = (diff > 0).float().sum()
        ties = (diff == 0).float().sum() * 0.5

        auc = (concordant + ties) / (len(pos_preds) * len(neg_preds))
        return auc.item()


def calculate_f1_flexible(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Calculates F1-Score with a configurable classification threshold.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    The threshold determines the operating point on the precision-recall
    curve. In fraud detection:
    - threshold=0.3: Aggressive (catch more fraud, more false alarms)
    - threshold=0.5: Balanced (standard)
    - threshold=0.7: Conservative (fewer false alarms, miss more fraud)

    The optimal threshold depends on the BUSINESS COST of each error type:
    - False Positive cost: Inconvenience to legitimate customer
    - False Negative cost: Financial loss from undetected fraud

    Args:
        logits: Model predictions (pre-sigmoid).
        targets: Ground truth labels (0 or 1).
        threshold: Classification threshold.

    Returns:
        F1 score in [0, 1].
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits).view(-1) >= threshold).float()
        targets = targets.view(-1).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1.item()


def calculate_precision_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculates Precision, Recall, and related metrics.

    Provides a detailed breakdown of classification performance:
    - Precision: TP / (TP + FP) — "quality" of positive predictions
    - Recall: TP / (TP + FN) — "coverage" of actual positives
    - Specificity: TN / (TN + FP) — "quality" of negative predictions
    - Accuracy: (TP + TN) / total

    Args:
        logits: Model predictions (pre-sigmoid).
        targets: Ground truth labels.
        threshold: Classification threshold.

    Returns:
        Dictionary with precision, recall, specificity, and accuracy.
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits).view(-1) >= threshold).float()
        targets = targets.view(-1).float()

        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()
        tn = ((1 - preds) * (1 - targets)).sum().item()

        return {
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
            "specificity": tn / (tn + fp + 1e-8),
            "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-8),
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
    RMSE for Missing Data Imputation evaluation.

    Measures how well the model can reconstruct values that were
    artificially masked during evaluation. Lower RMSE = better
    imputation quality.

    This metric evaluates the PRE-TRAINING objective quality:
    if the model can accurately impute missing values, it has learned
    meaningful representations of the data distribution.

    Args:
        preds: Model predicted values.
        targets: Original ground truth values.
        mask: Boolean mask (True = position was artificially masked).

    Returns:
        RMSE over masked positions. Returns 0.0 if no positions masked.
    """
    with torch.no_grad():
        if mask.sum() == 0:
            return 0.0

        masked_preds = preds[mask]
        masked_targets = targets[mask]

        mse = F.mse_loss(masked_preds, masked_targets)
        rmse = torch.sqrt(mse)
        return rmse.item()


class RiskMetricsEvaluator:
    """
    Comprehensive metrics evaluator for risk models.

    Aggregates all relevant metrics in a single call, providing
    a complete picture of model performance suitable for reporting
    to business stakeholders.

    Usage:
        >>> evaluator = RiskMetricsEvaluator(threshold=0.3)
        >>> results = evaluator.evaluate(logits, targets)
        >>> print(results)
        {'auc': 0.85, 'f1_score': 0.72, 'precision': 0.68, 'recall': 0.77, ...}
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes all metrics for the given predictions.

        Args:
            logits: Model predictions (pre-sigmoid).
            targets: Ground truth labels.
            threshold: Override the default threshold.

        Returns:
            Dictionary with all computed metrics.
        """
        t = threshold or self.threshold

        results = {
            "auc": calculate_auc(logits, targets),
            "f1_score": calculate_f1_flexible(logits, targets, t),
        }
        results.update(calculate_precision_recall(logits, targets, t))

        return results

"""
Benchmark Suite for Formal LDM Evaluation.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
This module runs the trained LDM through standardized evaluation protocols
against the test set, computing comprehensive metrics and generating
a formatted report.

The benchmarks evaluate:
1. Classification performance (AUC, F1, Precision, Recall)
2. Model robustness under distribution shift (Random + FGSM)
3. Imputation quality (from pre-training objective)

============================================================================
BENCHMARK DATASETS (Reference)
============================================================================
The following datasets are referenced in the LDM literature. While this
module currently evaluates on our synthetic data, the metrics and protocols
are designed to be directly comparable:

1. V4FinBench: Corporate bankruptcy forecasting (extreme imbalance)
2. OpenML-CC18: Zero-shot generalist tabular classification
3. TabArena: Few-shot tabular task benchmark
4. TALENT-REG: Tabular inference quality & interpretability
5. Adult/BankMarketing: Strategic manipulation simulation

============================================================================
REFERENCES
============================================================================
  - TabArena: https://github.com/puhsu/tabular-dl-tabr
  - OpenML-CC18: https://www.openml.org/s/99
  - V4FinBench: Corporate bankruptcy forecasting benchmark
============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import json

from src.evaluation.metrics import RiskMetricsEvaluator, calculate_auc, calculate_f1_flexible
from src.evaluation.strategic_shift_tester import StrategicShiftTester

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Orchestrates formal evaluation of the LDM across multiple metrics.

    Provides a standardized evaluation pipeline that:
    1. Runs inference on the test set
    2. Computes classification metrics
    3. Tests robustness (optional)
    4. Generates a formatted results dictionary

    Args:
        model: Trained LDM model (in eval mode).
        config: Configuration dictionary with evaluation parameters.
        device: Device for inference ("cpu" or "cuda").
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.threshold = config.get("metric_threshold", 0.5)

        # Metrics evaluator
        self.metrics_evaluator = RiskMetricsEvaluator(threshold=self.threshold)

        # Robustness tester (optional)
        self.enable_robustness = config.get("enable_strategic_shift", False)
        if self.enable_robustness:
            self.shift_tester = StrategicShiftTester(
                noise_std=config.get("noise_std", 0.1),
                adversarial_epsilon=config.get("adversarial_epsilon", 0.05),
            )

    def evaluate_classification(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Evaluates binary classification performance on the test set.

        Iterates through the full test DataLoader, collecting all
        predictions, then computes aggregate metrics.

        Args:
            dataloader: Test DataLoader with packed batches.

        Returns:
            Dictionary with AUC, F1, Precision, Recall, Accuracy.
        """
        logger.info("[Benchmark] Running classification evaluation...")

        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                keys = batch["packed_keys"].to(self.device)
                values = batch["packed_values"].to(self.device)
                times = batch["packed_times"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                labels = batch["labels"]

                # Forward pass
                logits = self.model(keys, values, times, tabular_features=tabular)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        # Aggregate predictions
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute all metrics
        results = self.metrics_evaluator.evaluate(all_logits, all_labels)

        logger.info(
            f"[Benchmark] Classification results: "
            f"AUC={results['auc']:.4f}, F1={results['f1_score']:.4f}, "
            f"Precision={results['precision']:.4f}, Recall={results['recall']:.4f}"
        )

        return results

    def evaluate_robustness(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Evaluates model robustness under distribution shift.

        Tests the model against:
        1. Random Gaussian noise (natural drift)
        2. FGSM adversarial perturbation (strategic manipulation)

        Args:
            dataloader: Test DataLoader.

        Returns:
            Dictionary with clean, shifted, and adversarial metrics.
        """
        if not self.enable_robustness:
            logger.info("[Benchmark] Robustness testing disabled in config.")
            return {}

        logger.info("[Benchmark] Running robustness evaluation...")

        # Get first batch for robustness testing
        batch = next(iter(dataloader))
        keys = batch["packed_keys"].to(self.device)
        values = batch["packed_values"].to(self.device)
        times = batch["packed_times"].to(self.device)
        tabular = batch["tabular"].to(self.device)
        labels = batch["labels"].to(self.device)

        results = self.shift_tester.evaluate_robustness(
            self.model,
            (keys, values, times),
            labels,
            tabular=tabular,
        )

        return results

    def run_all(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Runs the complete benchmark suite.

        Args:
            dataloader: Test DataLoader.

        Returns:
            Comprehensive results dictionary.
        """
        logger.info("=" * 60)
        logger.info("[Benchmark] Starting full evaluation suite")
        logger.info("=" * 60)

        results = {}

        # Classification metrics
        classification_results = self.evaluate_classification(dataloader)
        results.update(classification_results)

        # Robustness metrics (if enabled)
        if self.enable_robustness:
            robustness_results = self.evaluate_robustness(dataloader)
            results.update(robustness_results)

        logger.info("[Benchmark] Full suite completed.")
        return results

    def save_results(
        self,
        results: Dict[str, float],
        output_path: str,
    ):
        """Saves benchmark results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"[Benchmark] Results saved to {path}")


# =============================================================================
# Standalone execution
# =============================================================================
if __name__ == "__main__":
    print("[Benchmark] To run benchmarks, use: python src/main.py")
    print("[Benchmark] The main.py pipeline handles data, training, and evaluation.")

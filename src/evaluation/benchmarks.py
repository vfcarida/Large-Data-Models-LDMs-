"""
Benchmark Suite for Formal LDM Evaluation and Governance Auditing.

============================================================================
PURPOSE
============================================================================
Runs a trained Large Data Model (LDM) through standardized evaluation
protocols covering:
  1. Imbalanced Classification Discrimination (ROC-AUC, PR-AUC, F1, Recall@FPR)
  2. Epistemic Calibration Auditing (Expected Calibration Error, Brier Score)
  3. Conformal Risk Abstention (Selective Accuracy & Coverage)
  4. Strategic & Natural Distribution Shift Resilience (Gaussian Drift & FGSM)

============================================================================
REFERENCES
============================================================================
  - RelBench (Robinson et al., NeurIPS 2024, arXiv:2407.20060)
  - Federal Reserve SR 11-7 / SR 26-2 Model Risk Management Guidance
  - Gartner AI TRiSM (AI Trust, Risk, and Security Management)
============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging

import torch
import torch.nn as nn

from src.evaluation.metrics import RiskMetricsEvaluator
from src.evaluation.strategic_shift_tester import StrategicShiftTester

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Orchestrates formal evaluation of the LDM across discrimination,
    calibration, conformal abstention, and adversarial shift protocols.

    Args:
        model: Trained `LargeDataModel` instance.
        config: Configuration dictionary with evaluation parameters.
        device: Target inference device (`"cpu"` or `"cuda"`).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.threshold = float(config.get("metric_threshold", 0.5))

        self.metrics_evaluator = RiskMetricsEvaluator(
            threshold=self.threshold,
            conformal_alpha=float(config.get("conformal_alpha", 0.10)),
        )

        self.enable_robustness = bool(config.get("enable_strategic_shift", True))
        self.shift_tester = StrategicShiftTester(
            noise_std=float(config.get("noise_std", 0.1)),
            adversarial_epsilon=float(config.get("adversarial_epsilon", 0.05)),
        )

    def evaluate_classification(
        self,
        dataloader,
        val_dataloader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates classification, calibration, and conformal abstention on `dataloader`.

        If `val_dataloader` is provided, calibrates the `ConformalRiskAbstainer`
        on validation predictions before evaluating on the test dataloader.
        """
        logger.info("[Benchmark] Running classification & calibration evaluation...")
        self.model.eval()

        if val_dataloader is not None:
            val_logits, val_labels = [], []
            with torch.no_grad():
                for batch in val_dataloader:
                    keys = batch["packed_keys"].to(self.device)
                    values = batch["packed_values"].to(self.device)
                    times = batch["packed_times"].to(self.device)
                    tabular = batch["tabular"].to(self.device)
                    padding_mask = batch.get("key_padding_mask")
                    if padding_mask is not None:
                        padding_mask = padding_mask.to(self.device)
                    logits = self.model(
                        keys,
                        values,
                        times,
                        tabular_features=tabular,
                        key_padding_mask=padding_mask,
                    )
                    val_logits.append(logits.cpu())
                    val_labels.append(batch["labels"].cpu())
            if val_logits:
                self.metrics_evaluator.abstainer.calibrate(
                    torch.cat(val_logits, dim=0),
                    torch.cat(val_labels, dim=0),
                )

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                keys = batch["packed_keys"].to(self.device)
                values = batch["packed_values"].to(self.device)
                times = batch["packed_times"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                padding_mask = batch.get("key_padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(self.device)
                labels = batch["labels"]

                logits = self.model(
                    keys,
                    values,
                    times,
                    tabular_features=tabular,
                    key_padding_mask=padding_mask,
                )

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        results = self.metrics_evaluator.evaluate(all_logits_tensor, all_labels_tensor)

        logger.info(
            f"[Benchmark] Classification results: "
            f"ROC-AUC={results['auc']:.4f}, PR-AUC={results['pr_auc']:.4f}, "
            f"F1={results['f1_score']:.4f}, ECE={results['ece']:.4f}"
        )

        return results

    def evaluate_robustness(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Evaluates model robustness under random Gaussian drift and FGSM evasion.
        """
        if not self.enable_robustness:
            logger.info("[Benchmark] Robustness testing disabled in config.")
            return {}

        logger.info("[Benchmark] Running robustness evaluation across test batches...")
        self.model.eval()

        clean_logits_list = []
        shifted_logits_list = []
        adv_logits_list = []
        labels_list = []

        for batch in dataloader:
            keys = batch["packed_keys"].to(self.device)
            values = batch["packed_values"].to(self.device)
            times = batch["packed_times"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            padding_mask = batch.get("key_padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                clean_logits = self.model(
                    keys,
                    values,
                    times,
                    tabular_features=tabular,
                    key_padding_mask=padding_mask,
                )
                shifted_vals = self.shift_tester.apply_random_shift(values)
                if padding_mask is not None:
                    shifted_vals = shifted_vals.masked_fill(padding_mask, 0.0)
                shifted_logits = self.model(
                    keys,
                    shifted_vals,
                    times,
                    tabular_features=tabular,
                    key_padding_mask=padding_mask,
                )

            adv_inputs = self.shift_tester.apply_gradient_adversarial_shift(
                self.model,
                (keys, values, times),
                labels,
                tabular=tabular,
                key_padding_mask=padding_mask,
            )
            with torch.no_grad():
                adv_logits = self.model(
                    adv_inputs[0],
                    adv_inputs[1],
                    adv_inputs[2],
                    tabular_features=tabular,
                    key_padding_mask=padding_mask,
                )

            clean_logits_list.append(clean_logits.cpu())
            shifted_logits_list.append(shifted_logits.cpu())
            adv_logits_list.append(adv_logits.cpu())
            labels_list.append(labels.cpu())

        from src.evaluation.metrics import calculate_auc

        all_clean = torch.cat(clean_logits_list, dim=0)
        all_shifted = torch.cat(shifted_logits_list, dim=0)
        all_adv = torch.cat(adv_logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        clean_auc = calculate_auc(all_clean, all_labels)
        shifted_auc = calculate_auc(all_shifted, all_labels)
        adv_auc = calculate_auc(all_adv, all_labels)

        return {
            "clean_auc": float(clean_auc),
            "random_shift_auc": float(shifted_auc),
            "fgsm_auc": float(adv_auc),
            "random_degradation": float(clean_auc - shifted_auc),
            "fgsm_degradation": float(clean_auc - adv_auc),
        }

    def evaluate_serving_latency(self) -> Dict[str, Any]:
        """
        Benchmarks single-request (`batch_size=1`) TorchScript compilation,
        numerical parity, millisecond p50/p95/p99 authorization latency,
        Dynamic INT8 quantization compression, and Two-Stage Early-Exit Cascade speedup.
        """
        from src.serving.exporter import (
            benchmark_inference_latency,
            build_dummy_inference_inputs,
        )
        from src.serving.quantization_and_cascade import CascadeEarlyExitRouter

        logger.info(
            "[Benchmark] Running TorchScript, INT8 Quantization & Early-Exit Cascade benchmark..."
        )
        metrics = benchmark_inference_latency(
            self.model,
            batch_size=1,
            seq_len=32,
            num_warmup=5,
            num_runs=30,
        )

        router = CascadeEarlyExitRouter(self.model, fast_exit_threshold=0.35)
        keys, vals, times, tab, pad = build_dummy_inference_inputs(
            self.model, batch_size=16, seq_len=32, device=torch.device(self.device)
        )
        router.fit_fast_stage(keys, vals, times, tab, pad, num_steps=20)
        cascade_report = router.benchmark_cascade(
            keys, vals, times, tab, pad, num_runs=15
        )
        metrics["fp32_size_mb"] = cascade_report.fp32_size_mb
        metrics["int8_size_mb"] = cascade_report.int8_size_mb
        metrics["int8_compression_ratio"] = cascade_report.compression_ratio
        metrics["int8_max_prob_diff"] = cascade_report.int8_max_prob_diff
        metrics["cascade_fast_exit_rate"] = cascade_report.fast_exit_rate
        metrics["cascade_speedup_factor"] = cascade_report.cascade_speedup_factor
        return metrics

    def run_all(
        self,
        dataloader,
        val_dataloader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs the complete classification, calibration, robustness, and latency suite."""
        logger.info("=" * 60)
        logger.info("[Benchmark] Starting full evaluation suite")
        logger.info("=" * 60)

        results: Dict[str, Any] = {}
        results.update(self.evaluate_classification(dataloader, val_dataloader=val_dataloader))

        if self.enable_robustness:
            results.update(self.evaluate_robustness(dataloader))

        results.update(self.evaluate_serving_latency())

        logger.info("[Benchmark] Full suite completed.")
        return results

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
    ) -> None:
        """Saves benchmark results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"[Benchmark] Results saved to {path}")


if __name__ == "__main__":
    print("[Benchmark] To run benchmarks, execute: python src/main.py")

# =============================================================================
# Evaluation Sub-Package — Metrics, Benchmarks & Robustness Testing
# =============================================================================
#
# Contains:
# 1. Risk-aware metrics (AUC, F1, RMSE) optimized for imbalanced financial data
# 2. Benchmark suite for formal evaluation against standard datasets
# 3. Strategic shift tester for OOD/adversarial robustness analysis
# =============================================================================

from src.evaluation.metrics import (
    calculate_auc,
    calculate_f1_flexible,
    calculate_imputation_rmse,
    RiskMetricsEvaluator,
)
from src.evaluation.strategic_shift_tester import StrategicShiftTester, inject_lora_adaptation
from src.evaluation.benchmarks import BenchmarkSuite

__all__ = [
    "calculate_auc",
    "calculate_f1_flexible",
    "calculate_imputation_rmse",
    "RiskMetricsEvaluator",
    "StrategicShiftTester",
    "inject_lora_adaptation",
    "BenchmarkSuite",
]

# =============================================================================
# Evaluation Sub-Package — Metrics, Calibration, Conformal Abstention & Shift
# =============================================================================
#
# Contains:
# 1. Imbalanced & calibration metrics (ROC-AUC, PR-AUC, Recall@FPR, F1,
#    Brier Score, Expected Calibration Error, Imputation RMSE)
# 2. ConformalRiskAbstainer for distribution-free human-in-the-loop abstention
# 3. BenchmarkSuite for formal multi-metric evaluation
# 4. StrategicShiftTester & inject_lora_adaptation for OOD/adversarial testing
# =============================================================================

from src.evaluation.metrics import (
    calculate_auc,
    calculate_pr_auc,
    calculate_recall_at_fpr,
    calculate_brier_score,
    calculate_ece,
    calculate_f1_flexible,
    calculate_precision_recall,
    calculate_imputation_rmse,
    ConformalRiskAbstainer,
    RiskMetricsEvaluator,
)
from src.evaluation.strategic_shift_tester import StrategicShiftTester, inject_lora_adaptation
from src.evaluation.benchmarks import BenchmarkSuite

__all__ = [
    "calculate_auc",
    "calculate_pr_auc",
    "calculate_recall_at_fpr",
    "calculate_brier_score",
    "calculate_ece",
    "calculate_f1_flexible",
    "calculate_precision_recall",
    "calculate_imputation_rmse",
    "ConformalRiskAbstainer",
    "RiskMetricsEvaluator",
    "StrategicShiftTester",
    "inject_lora_adaptation",
    "BenchmarkSuite",
]

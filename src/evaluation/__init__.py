# =============================================================================
# Evaluation Sub-Package — Metrics, Calibration, Explainability & Ablation
# =============================================================================
#
# Contains:
# 1. Imbalanced & calibration metrics (ROC-AUC, PR-AUC, Recall@FPR, F1,
#    Brier Score, Expected Calibration Error, Imputation RMSE)
# 2. ConformalRiskAbstainer for distribution-free human-in-the-loop abstention
# 3. LDMExplainabilityEngine for Integrated Gradients, Adverse Action Reason
#    Codes (ECOA / SR 11-7 / GDPR Art. 22), and Counterfactual Recourse
# 4. AblationBenchmarkSuite for multi-pathway architectural ablations
# 5. BenchmarkSuite & StrategicShiftTester for formal evaluation
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
from src.evaluation.explainability import (
    AdverseActionReason,
    CounterfactualRecourseResult,
    LDMExplainabilityEngine,
)
from src.evaluation.ablation_suite import AblationBenchmarkSuite, AblationVariantMetrics
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
    "AdverseActionReason",
    "CounterfactualRecourseResult",
    "LDMExplainabilityEngine",
    "AblationBenchmarkSuite",
    "AblationVariantMetrics",
    "BenchmarkSuite",
]

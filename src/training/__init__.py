# =============================================================================
# Training Sub-Package — Objectives & Lightning Modules
# =============================================================================
#
# Contains:
# 1. Self-Supervised Objectives (NextTokenPrediction, MaskedJointDistribution)
# 2. LightningModule wrappers for pre-training and fine-tuning
# 3. Learning rate schedulers (Warmup + Cosine Annealing)
# =============================================================================

from src.training.objectives import (
    NextTokenPredictionLoss,
    ContextConditionalMaskedLoss,
)
from src.training.lightning_module import LDMPreTrainingModule, LDMFineTuningModule
from src.training.lr_scheduler import get_cosine_schedule_with_warmup

__all__ = [
    "NextTokenPredictionLoss",
    "ContextConditionalMaskedLoss",
    "LDMPreTrainingModule",
    "LDMFineTuningModule",
    "get_cosine_schedule_with_warmup",
]

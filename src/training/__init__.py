# =============================================================================
# Training Sub-Package — Objectives, Lightning Modules & Federated DP-SSL
# =============================================================================
#
# Contains:
# 1. Self-Supervised Objectives (NextTokenPredictionLoss,
#    ContextConditionalMaskedLoss, ContrastiveEventSequenceLoss)
# 2. LightningModule wrappers for pre-training and fine-tuning
# 3. Learning rate schedulers (Warmup + Cosine Annealing)
# 4. FederatedLDMCoordinator for Cross-Silo FedAvg/FedProx with (epsilon, delta)-DP
# =============================================================================

from src.training.objectives import (
    SSLBaseLoss,
    NextTokenPredictionLoss,
    ContextConditionalMaskedLoss,
    ContrastiveEventSequenceLoss,
)
from src.training.lightning_module import LDMPreTrainingModule, LDMFineTuningModule
from src.training.lr_scheduler import get_cosine_schedule_with_warmup
from src.training.federated import (
    FederatedLDMCoordinator,
    FederatedPrivacyConfig,
    FederatedRoundReport,
)
from src.training.unlearning import (
    CertifiedLDMUnlearner,
    UnlearningCertificate,
)
from src.training.continual import (
    ContinualLDMAdapter,
    ContinualAdaptationReport,
)
from src.training.federated_dp import (
    DifferentiallyPrivateFedCoordinator,
    RdpPrivacyAccountant,
)

__all__ = [
    "SSLBaseLoss",
    "NextTokenPredictionLoss",
    "ContextConditionalMaskedLoss",
    "ContrastiveEventSequenceLoss",
    "LDMPreTrainingModule",
    "LDMFineTuningModule",
    "get_cosine_schedule_with_warmup",
    "FederatedLDMCoordinator",
    "FederatedPrivacyConfig",
    "FederatedRoundReport",
    "CertifiedLDMUnlearner",
    "UnlearningCertificate",
    "ContinualLDMAdapter",
    "ContinualAdaptationReport",
    "DifferentiallyPrivateFedCoordinator",
    "RdpPrivacyAccountant",
]

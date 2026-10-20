# =============================================================================
# Models Sub-Package — Neural Network Architectures
# =============================================================================
#
# Contains the core architectural components of the LDM:
#
# 1. MMTTTransformerEncoder & PeriodicNumericalEmbedding: The backbone
#    Transformer that processes Multi-Modal-Temporal-Tabular (MMTT) sequences
#    using periodic numerical embeddings (PLR), sinusoidal sequence positional
#    encoding, and additive KVT modality fusion.
#
# 2. RelationalEntityGNN & BipartiteGraphConvLayer: Bipartite user-merchant
#    relational message passing (RelBench / Temporal GNN).
#
# 3. TabRRetrievalModule: Differentiable kNN retrieval-augmented behavioral
#    attention over historical entity trajectories (TabR, ICLR 2024).
#
# 4. EndToEndFusionLayer & CrossNetworkV2: DCNv2-based parallel fusion network
#    that combines dense Transformer sequence embeddings with classical static
#    tabular features for downstream risk classification.
#
# 5. InContextPFNHead: TabPFN v2 inspired In-Context Learning posterior
#    approximation head for cold-start payment rails.
#
# 6. LargeDataModel: The unified model connecting all subsystems.
# =============================================================================

from src.models.transformer_backbone import (
    MMTTTransformerEncoder,
    PeriodicNumericalEmbedding,
    SinusoidalPositionalEncoding,
    VirtualTokenLayer,
)
from src.models.ssm_backbone import SelectiveSSMBlock, HybridSSMTransformerEncoder
from src.models.joint_fusion_network import EndToEndFusionLayer, CrossNetworkV2
from src.models.retrieval_attention import TabRRetrievalModule
from src.models.relational_gnn import RelationalEntityGNN, BipartiteGraphConvLayer
from src.models.in_context_pfn import InContextPFNHead
from src.models.multi_task_moe import (
    MultiGateMixtureOfExpertsHead,
    HomoscedasticMultiTaskLoss,
)
from src.models.causal_uplift import (
    CausalDragonNetHead,
    DragonNetTargetedLoss,
    recommend_optimal_intervention,
)
from src.models.hypergraph_contagion import (
    HypergraphConvLayer,
    TemporalHypergraphContagionModule,
)
from src.models.flow_matching import (
    ConditionalTrajectoryFlowMatcher,
    SinusoidalFlowTimeEmbedding,
)
from src.models.hawkes_point_process import NeuralHawkesProcessHead
from src.models.cross_border_alignment import (
    FXPurchasingPowerNormalizer,
    CrossBorderDomainAligner,
)
from src.models.monotonic_constraints import (
    MonotonicConstrainedHead,
    MonotonicityCertificationAuditor,
    MonotonicityAuditReport,
)
from src.models.in_context_ldm import InContextTabularPredictor
from src.models.survival_hazard import CompetingRisksSurvivalHead
from src.models.ldm_model import LargeDataModel

__all__ = [
    "MMTTTransformerEncoder",
    "PeriodicNumericalEmbedding",
    "SinusoidalPositionalEncoding",
    "VirtualTokenLayer",
    "SelectiveSSMBlock",
    "HybridSSMTransformerEncoder",
    "EndToEndFusionLayer",
    "CrossNetworkV2",
    "TabRRetrievalModule",
    "RelationalEntityGNN",
    "BipartiteGraphConvLayer",
    "InContextPFNHead",
    "MultiGateMixtureOfExpertsHead",
    "HomoscedasticMultiTaskLoss",
    "CausalDragonNetHead",
    "DragonNetTargetedLoss",
    "recommend_optimal_intervention",
    "HypergraphConvLayer",
    "TemporalHypergraphContagionModule",
    "ConditionalTrajectoryFlowMatcher",
    "SinusoidalFlowTimeEmbedding",
    "NeuralHawkesProcessHead",
    "FXPurchasingPowerNormalizer",
    "CrossBorderDomainAligner",
    "MonotonicConstrainedHead",
    "MonotonicityCertificationAuditor",
    "MonotonicityAuditReport",
    "InContextTabularPredictor",
    "CompetingRisksSurvivalHead",
    "LargeDataModel",
]

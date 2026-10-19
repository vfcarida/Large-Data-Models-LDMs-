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
# 2. EndToEndFusionLayer & CrossNetworkV2: DCNv2-based parallel fusion network
#    that combines dense Transformer sequence embeddings with classical static
#    tabular features for downstream risk classification.
#
# 3. LargeDataModel: The unified model connecting the MMTT backbone, SSL
#    multi-modal reconstruction heads, and DCNv2 fusion layer.
# =============================================================================

from src.models.transformer_backbone import (
    MMTTTransformerEncoder,
    PeriodicNumericalEmbedding,
    SinusoidalPositionalEncoding,
    VirtualTokenLayer,
)
from src.models.joint_fusion_network import EndToEndFusionLayer, CrossNetworkV2
from src.models.ldm_model import LargeDataModel

__all__ = [
    "MMTTTransformerEncoder",
    "PeriodicNumericalEmbedding",
    "SinusoidalPositionalEncoding",
    "VirtualTokenLayer",
    "EndToEndFusionLayer",
    "CrossNetworkV2",
    "LargeDataModel",
]

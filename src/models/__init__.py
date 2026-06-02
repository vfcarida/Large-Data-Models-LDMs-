# =============================================================================
# Models Sub-Package — Neural Network Architectures
# =============================================================================
#
# Contains the three core architectural components of the LDM:
#
# 1. MMTTTransformerEncoder: The backbone Transformer that processes
#    multi-modal-temporal-tabular (MMTT) sequences using additive fusion
#    of Key, Value, and Time embeddings.
#
# 2. EndToEndFusionLayer: DCNv2-based fusion network that combines dense
#    Transformer embeddings with classical tabular features for downstream
#    binary classification tasks.
#
# 3. LargeDataModel: The complete end-to-end model that connects the
#    Transformer backbone with the fusion layer, supporting both
#    pre-training (SSL) and fine-tuning (supervised) modes.
# =============================================================================

from src.models.transformer_backbone import MMTTTransformerEncoder, VirtualTokenLayer
from src.models.joint_fusion_network import EndToEndFusionLayer, CrossNetworkV2
from src.models.ldm_model import LargeDataModel

__all__ = [
    "MMTTTransformerEncoder",
    "VirtualTokenLayer",
    "EndToEndFusionLayer",
    "CrossNetworkV2",
    "LargeDataModel",
]

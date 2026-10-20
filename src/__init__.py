# =============================================================================
# Large Data Models (LDMs) — Root Package
# =============================================================================
#
# Reference implementation of a Large Data Model (LDM) research and engineering
# platform. LDMs are self-supervised foundation models designed natively for
# structured, tabular, and temporal event sequences.
#
# Sub-packages:
#   - data:       Synthetic prior-data generation, KVT tokenization, DataModule
#   - models:     MMTT Transformer, Periodic Numerical Embeddings (PLR), DCNv2
#   - training:   LimiX/PRAGMA masked SSL, CoLES contrastive SSL, Lightning
#   - evaluation: Imbalanced metrics, ECE/Brier calibration, Conformal
#                 abstention, and FGSM strategic shift stress testing
#
# Author: Vinicius Caridá (vfcarida)
# License: MIT
# =============================================================================

__version__ = "1.0.0"
__author__ = "Vinicius Caridá"

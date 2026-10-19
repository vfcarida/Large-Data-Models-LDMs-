# =============================================================================
# Data Sub-Package — Synthetic Priors, KVT Tokenization & Lightning DataModule
# =============================================================================
#
# Provides the end-to-end data ingestion and preparation pipeline for Large
# Data Models (LDMs):
#
# 1. SyntheticTransactionGenerator: Prior-data structural transaction simulator
#    modeling Zipfian Merchant Category Codes (MCC), log-normal amounts,
#    inter-event temporal deltas, static credit bureau features, and realistic
#    multi-pattern fraud trajectories (structuring, velocity bursts, ATO).
#
# 2. FinancialEventDataset & dynamic_sequence_packing_collate_fn: Key-Value-Time
#    (KVT) user sequence dataset with leakage-free z-score normalization and
#    attention padding mask collation.
#
# 3. LDMDataModule: PyTorch Lightning DataModule enforcing stratified user-level
#    train/validation/test partitioning and zero-leakage feature scaling.
# =============================================================================

from src.data.synthetic_generator import SyntheticTransactionGenerator, MCC_VOCABULARY
from src.data.mmtt_dataset import (
    FinancialEventDataset,
    NormalizationStats,
    dynamic_sequence_packing_collate_fn,
)
from src.data.data_module import LDMDataModule

__all__ = [
    "SyntheticTransactionGenerator",
    "MCC_VOCABULARY",
    "FinancialEventDataset",
    "NormalizationStats",
    "dynamic_sequence_packing_collate_fn",
    "LDMDataModule",
]

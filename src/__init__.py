# =============================================================================
# Large Data Models (LDMs) — Root Package
# =============================================================================
#
# This package contains the complete implementation of a Large Data Model (LDM)
# research laboratory. LDMs are foundation models designed specifically for
# structured/tabular data, analogous to how LLMs (Large Language Models) are
# designed for text.
#
# The package is organized into four sub-modules:
#   - data:       Data ingestion, tokenization, and synthetic generation
#   - models:     Neural network architectures (Transformer, DCNv2, Fusion)
#   - training:   Self-supervised and supervised training objectives
#   - evaluation: Metrics, benchmarks, and robustness testing
#
# References:
#   - TransactionGPT (Visa Research): 3D-Transformer for MMTT data
#   - PRAGMA (Revolut/NVIDIA): KVT tokenization for banking events
#   - LimiX: Masked joint-distribution pre-training for tabular data
#   - TabPFN v2/2.5: Prior-data fitted networks for tabular learning
#   - Schema-1 (DLM): Native cell processing for tabular foundation models
#   - DCNv2: Deep & Cross Network v2 for explicit feature interactions
# =============================================================================

__version__ = "0.1.0"

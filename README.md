# Large Data Models (LDMs) — Behavioral & Relational Foundation Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![PyTorch Lightning 2.1+](https://img.shields.io/badge/pytorch--lightning-2.1+-orange.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Stable](https://img.shields.io/badge/status-stable_v0.2.0-success.svg)](ROADMAP.md)

**Large Data Models (LDMs)** is a reference-grade PyTorch and PyTorch Lightning framework for **self-supervised pre-training and calibrated fine-tuning of Behavioral Foundation Models over structured financial event streams and static relational tables**.

Unlike Large Language Models (LLMs) that serialize floating-point quantities into subword text tokens—destroying numerical precision and inflating context windows—**LDMs** natively ingest continuous transaction amounts, irregular temporal deltas, discrete Merchant Category Codes (MCCs), and static credit bureau profiles through a unified, end-to-end differentiable architecture.

---

## Executive Value Proposition & Differentiation

> **Differentiation Thesis:** *The open-source PyTorch Behavioral & Relational Foundation Model (`LDM`) that unifies Continuous-Time Key-Value-Time (KVT) Transformer encoders (`PLR` + `Time2Vec`), collapse-free Joint Masked-Contrastive Self-Supervised Pre-Training (`LimiX` + `PRAGMA` + `CoLES`), and End-to-End Deep & Cross Tabular Fusion (`DCNv2`) with verifiable SR 11-7 / Gartner AI TRiSM calibration and split-conformal risk abstention.*

### Why Behavioral Foundation Models Replace Brittle Feature Pipelines
1. **Zero Manual Window Engineering:** Replaces thousands of hand-crafted SQL/Spark rolling aggregations (`30d_sum`, `90d_max`) with direct attention over raw `(Key, Value, Timestamp)` event trajectories ([TransactionGPT, 2025](https://arxiv.org/abs/2511.08939); [PRAGMA, 2026](https://arxiv.org/abs/2604.08649)).
2. **High-Frequency Numerical Fidelity:** Uses **Periodic Numerical Embeddings (`PLR`)** ([Gorishniy et al., *NeurIPS 2022*](https://arxiv.org/abs/2203.05556)) to overcome neural spectral bias on heavy-tailed monetary amounts, paired with **Time2Vec** ([Kazemi et al., *JMLR 2019*](https://arxiv.org/abs/1907.05321)) for continuous-time rhythm encoding.
3. **Collapse-Free Self-Supervised Pre-Training:** Jointly optimizes **Context-Conditional Multi-Modal Masked Reconstruction** ([LimiX, 2025](https://arxiv.org/abs/2509.03505)) across `(MCC, Amount, TimeDelta)` and **Contrastive Event Sequence Learning (`CoLES`)** ([Babaev et al., *SIGMOD 2022*](https://arxiv.org/abs/2002.08232)) on the hypersphere $\mathcal{S}^{d-1}$.
4. **Regulatory-Grade Risk Governance (SR 11-7 / Gartner AI TRiSM):** Evaluates models with imbalanced risk metrics (`PR-AUC`, `Recall@5%FPR`), calibration diagnostics (`ECE`, `Brier Score`), **Split-Conformal Risk Abstention** (`ConformalRiskAbstainer`), and **FGSM Adversarial / Strategic Shift** stress testing.

---

## System Architecture

```mermaid
graph TD
    subgraph Modality1 ["1. Multi-Modal Temporal Tokenization (MMTT)"]
        A1["Discrete Event Keys (MCC)"] -->|"nn.Embedding"| B1["Key Embedding e_k"]
        A2["Continuous Amounts (z-score)"] -->|"Periodic Numerical (PLR: sin/cos + Linear)"| B2["Value Embedding e_v"]
        A3["Time Deltas (Δt)"] -->|"Discrete Bins / Continuous Time2Vec"| B3["Temporal Embedding e_t"]
        B1 & B2 & B3 -->|"Element-wise Additive Fusion + LayerNorm"| C["Composite Event Token x_t"]
    end

    subgraph Backbone ["2. Context-Conditioned Transformer Backbone"]
        V["Learnable Virtual Prompt Tokens"] -->|"Prepend"| D["Conditioned Sequence"]
        C -->|"Sinusoidal Positional Encoding"| D
        D -->|"Pre-Norm Transformer Layers"| E["Contextualized Representations H"]
        M["Key Padding Mask"] -->|"Strictly Blocks Padded Positions"| E
        E -->|"Mask-Aware Pooling (CLS / Mean / Max / Attentive)"| F["Sequence Embedding h_seq"]
    end

    subgraph SSL ["3a. Phase 1: Hybrid Self-Supervised Pre-Training"]
        E -->|"Masked Positions Only"| S1["Multi-Modal Heads (Key CE + Value SmoothL1 + Time SmoothL1)"]
        E -->|"Asymmetric Predictor"| S2["Latent Stop-Grad Cosine Alignment"]
        F -->|"Sub-Trajectory Augmented Views"| S3["CoLES Contrastive InfoNCE Loss"]
    end

    subgraph Fusion ["3b. Phase 2: Supervised DCNv2 Joint Fusion & Governance"]
        T["Static Tabular Features (Bureau, Income, Age)"] -->|"Train-Only Z-Score Scaling"| G["Normalized Tabular Vector x_tab"]
        F & G -->|"Concatenate x_0"| H["Joint State [h_seq || x_tab]"]
        H -->|"Bounded-Degree Polynomial Crosses"| I["CrossNetworkV2 (Full or Low-Rank)"]
        H -->|"Deep Non-Linear MLP"| J["Deep Network + LayerNorm"]
        I & J -->|"Concatenate + Output Projection"| K["Fraud Logits & Probabilities"]
        K -->|"Split-Conformal Calibration"| L["Auto-Approve / Auto-Decline / Human Review Abstention"]
    end
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/vfcarida/Large-Data-Models-LDMs-.git
cd Large-Data-Models-LDMs-

# Install core package in editable mode
pip install -e .

# Optional: Install visualization, tracking, or PEFT extras
pip install -e ".[dev,viz,tracking,peft]"
```

**Docker Execution:**
```bash
docker build -t ldm-research -f docker/Dockerfile .
docker run --gpus all -it ldm-research
```

### 2. Run the End-to-End Pipeline

The CLI orchestrates synthetic data generation, Phase 1 self-supervised pre-training, Phase 2 supervised DCNv2 fine-tuning, and the Phase 3 governance benchmark suite:

```bash
# Fast CPU verification run (~25 seconds on CPU; writes reports to results/)
python src/main.py --config small

# Full production configuration
python src/main.py --config default

# Override specific hyperparameters from the CLI
python src/main.py --config small --pretrain-epochs 5 --finetune-epochs 10 --seed 42
```

### 3. Run the Automated Test Suite

```bash
pytest -v
```

---

## Verified Benchmark Results (`small` CPU Configuration)

Reproducible metrics generated by running `python src/main.py --config small` (`seed: 42`, 400 users: 280 train / 60 validation / 60 test, 12% positive fraud rate across three stealth and velocity fraud archetypes; full report in [`results/metrics_report.md`](results/metrics_report.md)):

| Category | Metric | Value | Governance Interpretation |
| :--- | :--- | :---: | :--- |
| **Discrimination** | `ROC-AUC` | **0.8491** | Strong separation across clean and camouflaged fraud trajectories |
| **Imbalanced Precision-Recall** | `PR-AUC` | **0.4314** | **3.6x lift** over the 12.0% prior fraud base rate |
| **Operational Specificity** | `Specificity` | **0.8491** | High legitimate-user approval rate at standard operating threshold |
| **Overall Accuracy** | `Standard Accuracy` | **0.8000** | Baseline uncalibrated accuracy prior to conformal abstention |
| **Probabilistic Calibration** | `Brier Score` | **0.1531** | Proper quadratic probability scoring error |
| **Probabilistic Calibration** | `Expected Calibration Error (ECE)` | **0.1852** | 10-bin empirical confidence-to-accuracy alignment |
| **Conformal Risk Governance** | `Conformal Coverage` | **81.67%** | Automated decisions issued on 81.67% of test instances (`18.33%` routed to review) |
| **Conformal Risk Governance** | `Conformal Selective Accuracy` | **87.76%** | **+7.76% accuracy gain** on automated decisions after abstaining on borderline band `[0.1348, 0.8460]` |
| **Distributional Robustness** | `Random Covariate Shift ROC-AUC` | **0.8464** | Minimal degradation (`-0.0027` AUC) under Gaussian covariate noise ($\sigma=0.10$) |
| **Adversarial Robustness** | `FGSM Adversarial Shift ROC-AUC` | **0.8275** | Resilient (`-0.0216` AUC degradation) under worst-case gradient perturbation ($\epsilon=0.05$) |

---

## Repository Structure

```text
Large-Data-Models-LDMs-/
├── configs/
│   ├── config.yaml                  # Full production hyperparameter configuration
│   └── config_small.yaml            # Fast CPU verification configuration (<30s runtime)
├── docker/
│   └── Dockerfile                   # CUDA 12.1 + Python 3.10 container specification
├── docs/
│   ├── 01_fundamentos_ldm.md        # Foundations of Behavioral & Tabular Large Data Models
│   ├── 02_arquitetura_mmtt.md       # MMTT Tokenization, Periodic Embeddings (PLR) & Time2Vec
│   ├── 03_pretraining_limix.md      # Hybrid Self-Supervised Pre-Training (LimiX + PRAGMA + CoLES)
│   ├── 04_finetuning_dcnv2.md       # Supervised Fine-Tuning & Deep-Cross Fusion (DCNv2)
│   ├── 05_avaliacao_robustez.md     # Calibrated Evaluation, Conformal Abstention & Shift Testing
│   └── references.md                # Consolidated Scientific & Executive Bibliography
├── results/
│   ├── config.json                  # Reproducible configuration snapshot from latest run
│   └── metrics_report.md            # Generated evaluation & governance report
├── src/
│   ├── data/
│   │   ├── data_module.py           # LDMDataModule (stratified user split & train-only scaling)
│   │   ├── mmtt_dataset.py          # FinancialEventDataset & dynamic sequence packing collate
│   │   └── synthetic_generator.py   # Multi-archetype synthetic transaction prior generator
│   ├── evaluation/
│   │   ├── benchmarks.py            # BenchmarkSuite (Classification + Calibration + Conformal + Shift)
│   │   ├── metrics.py               # ROC-AUC, PR-AUC, Recall@FPR, Brier, ECE & ConformalRiskAbstainer
│   │   └── strategic_shift_tester.py# Progressive covariate & adversarial concept shift simulator
│   ├── models/
│   │   ├── joint_fusion_network.py  # CrossNetworkV2 (full/low-rank) & EndToEndFusionLayer
│   │   ├── ldm_model.py             # Unified LargeDataModel (MMTT + SSL Heads + DCNv2)
│   │   ├── transformer_backbone.py  # PeriodicNumericalEmbedding, ContinuousTimeEncoding & MMTT
│   │   └── virtual_token_Injector.py# Learnable Virtual Token Prompting & MultiModalTokenFusion
│   ├── training/
│   │   ├── lightning_module.py      # LDMPreTrainingModule & LDMFineTuningModule
│   │   ├── lr_scheduler.py          # CosineWarmupScheduler
│   │   └── objectives.py            # ContextConditionalMaskedLoss & ContrastiveEventSequenceLoss
│   └── main.py                      # End-to-end CLI entrypoint
├── tests/
│   └── test_ldm.py                  # Comprehensive unit & integration test suite (pytest)
├── CITATION.cff                     # Machine-readable academic citation metadata
├── LICENSE                          # MIT License
├── pyproject.toml                   # PEP 621 build system, dependencies & tool configuration
├── requirements.txt                 # Core runtime dependencies
├── ROADMAP.md                       # Architectural roadmap & maturity governance
└── setup.py                         # Legacy setuptools compatibility shim
```

---

## Documentation & Architecture Guides

| Guide | Focus Area |
| :--- | :--- |
| [**01 — Foundations of Large Data Models**](docs/01_fundamentos_ldm.md) | Why LLMs fail on numerical/temporal streams; synthetic prior generation; enterprise AI TRiSM governance. |
| [**02 — MMTT Architecture & Tokenization**](docs/02_arquitetura_mmtt.md) | Periodic Numerical Embeddings (`PLR`), `Time2Vec`, Virtual Token Prompting, and mask-aware pooling. |
| [**03 — Hybrid Self-Supervised Pre-Training**](docs/03_pretraining_limix.md) | Mathematical formulation of `LimiX`/`PRAGMA` multi-modal masked reconstruction and `CoLES` contrastive loss. |
| [**04 — Supervised Fine-Tuning & DCNv2 Fusion**](docs/04_finetuning_dcnv2.md) | Full-rank vs. low-rank `CrossNetworkV2` polynomial interactions and zero-leakage static feature scaling. |
| [**05 — Evaluation, Conformal Governance & Robustness**](docs/05_avaliacao_robustez.md) | `PR-AUC`, `Recall@FPR`, `ECE`, `Brier Score`, `ConformalRiskAbstainer`, and FGSM adversarial stress testing. |
| [**Academic & Industry References**](docs/references.md) | Tier-verified bibliography across *Nature*, *NeurIPS*, *ICML*, *ICLR*, *SIGMOD*, *WWW*, Gartner, and SR 11-7. |
| [**Roadmap & Maturity Governance**](ROADMAP.md) | Maturity quality gates (`Stable` / `Beta` / `Planned`) and upcoming horizon tracks (`TabR`, `RelBench`, `TabPFN v2`). |

---

## Roadmap Summary

See [`ROADMAP.md`](ROADMAP.md) for the complete milestone matrix and verification gates.

| Subsystem / Milestone | Status | Verification Gate |
| :--- | :---: | :--- |
| **Zero-Leakage Data Subsystem & Synthetic Prior Generator** (`src/data/`) | `Stable` | `tests/test_ldm.py::TestDataSubsystem` |
| **MMTT Backbone + Periodic Numerical Embeddings (`PLR`)** (`src/models/`) | `Stable` | `tests/test_ldm.py::TestMMTTBackbone` |
| **Deep & Cross Network v2 (`DCNv2`) Joint Fusion** (`src/models/`) | `Stable` | `tests/test_ldm.py::TestFusionNetwork` |
| **Hybrid SSL (`LimiX` + `PRAGMA` + `CoLES`)** (`src/training/`) | `Stable` | `tests/test_ldm.py::TestSSLAndFineTuning` |
| **Calibrated Risk Governance & Split-Conformal Abstention** (`src/evaluation/`) | `Stable` | `tests/test_ldm.py::TestEvaluationAndGovernance` |
| **Retrieval-Augmented Tabular Attention (`TabR` Integration)** | `Planned (v0.3.0)` | Nearest-neighbor context retrieval |
| **Relational Entity Graph Message Passing (`RelBench` / Temporal GNN)** | `Planned (v0.3.0)` | Multi-table graph-sequence fusion |
| **ONNX / TorchScript Export & Low-Latency Streaming Inference** | `Planned (v0.4.0)` | Sub-10ms p99 authorization benchmark |

---

## Citation

If you use this software or architectural blueprint in your research or enterprise systems, please cite it using the metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{carida2026ldm,
  author       = {Carid{\'a}, Vinicius},
  title        = {{Large Data Models (LDMs): Foundation Models for Structured Tabular and Continuous-Time Transactional Sequences}},
  year         = {2026},
  version      = {0.2.0},
  url          = {https://github.com/vfcarida/Large-Data-Models-LDMs-},
  license      = {MIT}
}
```

---

## License & Attribution

Released under the [MIT License](LICENSE).

*Designed and architected by [Vinicius Caridá](https://github.com/vfcarida). Implementation and verification accelerated with AI-assisted engineering workflows.*

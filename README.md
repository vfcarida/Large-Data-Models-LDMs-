# Large Data Models (LDMs) — Behavioral & Relational Foundation Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![PyTorch Lightning 2.1+](https://img.shields.io/badge/pytorch--lightning-2.1+-orange.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Stable](https://img.shields.io/badge/status-stable_v0.5.0-success.svg)](ROADMAP.md)

**Large Data Models (LDMs)** is a reference-grade PyTorch and PyTorch Lightning framework for **self-supervised & differentially-private federated pre-training, relational-retrieval-augmented fine-tuning, selective state-space (`Mamba` / `Liquid-S4`) sequence modeling, multi-task Mixture-of-Experts (`MMoE`), in-context posterior inference, regulatory explainability, online concept-drift monitoring (`ACI`), and real-time INT8/cascade streaming authorization of Behavioral Foundation Models over structured financial event streams and static relational tables**.

Unlike Large Language Models (LLMs) that serialize floating-point quantities into subword text tokens—destroying numerical precision and inflating context windows—**LDMs** natively ingest continuous transaction amounts, irregular temporal deltas, discrete Merchant Category Codes (MCCs), bipartite user-merchant interaction graphs, and static credit bureau profiles through a unified, end-to-end differentiable architecture.

---

## Executive Value Proposition & Differentiation

> **Differentiation Thesis:** *The open-source PyTorch Behavioral & Relational Foundation Model (`LDM v0.5.0`) that unifies Continuous-Time Key-Value-Time (KVT) Transformer & Selective State-Space (`Mamba` / `Liquid-S4`) encoders (`PLR` + `Time2Vec`), collapse-free Joint Masked-Contrastive & Cross-Silo Federated Pre-Training (`LimiX` + `PRAGMA` + `CoLES` + `FedProx`/`DP`), Bipartite Entity-Merchant Graph Message Passing (`RelBench`), Differentiable kNN Context Retrieval (`TabR`), End-to-End Deep & Cross Tabular Fusion (`DCNv2`), Multi-Gate Mixture-of-Experts (`MMoE` for Fraud + Credit Default + LTV), In-Context Prior-Data Fitted Inference (`TabPFN v2`), Integrated Gradients Adverse Action Explainability & Counterfactual Recourse, Online Concept Drift Monitoring (`PSI` / `MMD` / `ACI`), and TorchScript / Dynamic INT8 / Early-Exit Cascade streaming authorization.*

### Core Capabilities (`v0.5.0`)
1. **Zero Manual Window Engineering & Linear $O(S)$ State-Space Scaling:** Replaces thousands of hand-crafted SQL/Spark rolling aggregations (`30d_sum`, `90d_max`) with direct attention or continuous-time **Selective State-Space (`Mamba` / `Liquid-S4`)** recurrence (`SelectiveSSMBlock`, `HybridSSMTransformerEncoder`) over raw `(Key, Value, Timestamp)` event trajectories ([TransactionGPT, 2025](https://arxiv.org/abs/2511.08939); [Gu & Dao, *COLM 2024*](https://arxiv.org/abs/2312.00752)).
2. **High-Frequency Numerical Fidelity:** Uses **Periodic Numerical Embeddings (`PLR`)** ([Gorishniy et al., *NeurIPS 2022*](https://arxiv.org/abs/2203.05556)) to overcome neural spectral bias on heavy-tailed monetary amounts, paired with **Time2Vec** ([Kazemi et al., *JMLR 2019*](https://arxiv.org/abs/1907.05321)) for continuous-time rhythm encoding.
3. **Collapse-Free Self-Supervised & Cross-Silo Federated Pre-Training:** Jointly optimizes **Context-Conditional Multi-Modal Masked Reconstruction** ([LimiX, 2025](https://arxiv.org/abs/2509.03505)) and **Contrastive Event Sequence Learning (`CoLES`)** ([Babaev et al., *SIGMOD 2022*](https://arxiv.org/abs/2002.08232)), and supports multi-institution **`FedProx` + $(\epsilon, \delta)$-Differential Privacy** via `FederatedLDMCoordinator`.
4. **Relational Graph & Nearest-Neighbor Retrieval (`RelBench` + `TabR`):** Enriches sequence representations via Pre-Norm bipartite user-merchant graph message passing ([Robinson et al., *NeurIPS 2024*](https://arxiv.org/abs/2407.20060)) and differentiable top-$k$ historical trajectory retrieval ([Gorishniy et al., *ICLR 2024*](https://arxiv.org/abs/2307.14338)).
5. **Multi-Task Mixture-of-Experts (`MMoE`) & Cold-Start In-Context Inference (`TabPFN v2`):** Simultaneously predicts Real-Time Fraud Risk, 90-Day Credit Delinquency, and Customer LTV via `MultiGateMixtureOfExpertsHead` + `HomoscedasticMultiTaskLoss` ([Ma et al., *KDD 2018*](https://dl.acm.org/doi/10.1145/3219819.3220007); [Kendall et al., *CVPR 2018*](https://arxiv.org/abs/1705.07115)), alongside single-pass cold-start posterior inference via `InContextPFNHead` ([Hollmann et al., *Nature 2025*](https://doi.org/10.1038/s41586-024-08328-6)).
6. **Regulatory Explainability, Concept Drift Telemetry (`ACI`) & INT8 Cascade Serving:** Provides mask-aware **Integrated Gradients** adverse action reason codes, sparse $L_1/L_2$ counterfactual recourse (`LDMExplainabilityEngine`), statistical drift detection (`ConceptDriftMonitor` + `AdaptiveConformalAbstainer` — [Gibbs & Candès, *NeurIPS 2021*](https://arxiv.org/abs/2106.00170)), and `DynamicINT8Linear` + `CascadeEarlyExitRouter` (`13.03x` fast-exit cascade speedup and `1.20x` INT8 compression).

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

    subgraph Backbone ["2. Context-Conditioned Transformer & Selective SSM Backbone"]
        V["Learnable Virtual Prompt Tokens"] -->|"Prepend"| D["Conditioned Sequence"]
        C -->|"Sinusoidal Positional Encoding"| D
        D -->|"Pre-Norm Transformer or Hybrid Mamba/Liquid-S4 O(S) Layers"| E["Contextualized Representations H"]
        M["Key Padding Mask"] -->|"Strictly Blocks Padded Positions"| E
        E -->|"Mask-Aware Pooling (CLS / Mean / Max / Attentive)"| F["Sequence Embedding h_seq"]
    end

    subgraph SSL ["3a. Phase 1: Hybrid SSL & Federated DP-SSL"]
        E -->|"Masked Positions Only"| S1["Multi-Modal Heads (Key CE + Value SmoothL1 + Time SmoothL1)"]
        E -->|"Asymmetric Predictor"| S2["Latent Stop-Grad Cosine Alignment"]
        F -->|"Sub-Trajectory Augmented Views"| S3["CoLES Contrastive InfoNCE Loss"]
        S1 & S2 & S3 -.->|"Cross-Silo FedProx + L2 Clip + Gaussian DP"| FED["FederatedLDMCoordinator"]
    end

    subgraph RelationalRetrieval ["3b. Relational GNN & kNN Context Retrieval"]
        F -->|"Volume-Weighted User-MCC Graph"| GNN["RelationalEntityGNN (RelBench Bipartite GNN)"]
        GNN -->|"Scaled L2 Top-k Historical Bank"| TABR["TabRRetrievalModule (Differentiable kNN Attention)"]
    end

    subgraph Fusion ["4. Phase 2: DCNv2, Multi-Task MMoE, TabPFN & Governance"]
        T["Static Tabular Features (Bureau, Income, Age)"] -->|"Train-Only Z-Score Scaling"| G["Normalized Tabular Vector x_tab"]
        TABR & G -->|"Concatenate x_0"| H["Joint State [h_enriched || x_tab]"]
        H -->|"Bounded-Degree Polynomial Crosses"| I["CrossNetworkV2 + Deep MLP (DCNv2)"]
        H -.->|"Multi-Gate Mixture-of-Experts"| MMOE["MMoE Head (Fraud + 90d Credit Default + LTV)"]
        H -.->|"Cold-Start Support Set Conditioning"| PFN["InContextPFNHead (TabPFN v2 Posterior)"]
        I -->|"TorchScript / Dynamic INT8 / Cascade"| K["Fraud Logits & Probabilities"]
        K -->|"RealTimeAuthorizationService + ACI Drift Monitor"| L["APPROVE / DECLINE / HUMAN_REVIEW + IG Adverse Reasons"]
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

The CLI orchestrates synthetic data generation, Phase 1 self-supervised pre-training, Phase 2 supervised DCNv2 + Relational GNN + TabR + PFN + MMoE fine-tuning, and the Phase 3 governance, ablation, INT8 quantization, and cascade SLA benchmark suite:

```bash
# Fast CPU verification run (writes reports to results/)
python src/main.py --config small

# Full production configuration
python src/main.py --config default
```

### 3. Run the Automated Test Suite (24 Tests)

```bash
pytest -v
```

---

## Verified Benchmark Results (`small` CPU Configuration, `v0.5.0`)

Reproducible metrics generated by running `python src/main.py --config small` (`seed: 42`, 400 users: 280 train / 60 validation / 60 test, 12% positive fraud rate across three stealth and velocity fraud archetypes; full report in [`results/metrics_report.md`](results/metrics_report.md)):

| Category | Metric | Value | Governance & SLA Interpretation |
| :--- | :--- | :---: | :--- |
| **Discrimination** | `ROC-AUC` | **0.8140** | Strong separation across clean and camouflaged fraud trajectories |
| **Imbalanced Precision-Recall** | `PR-AUC` | **0.3219** | **2.7x lift** over the 12.0% prior fraud base rate |
| **Operational Specificity** | `Specificity` | **0.9245** | High legitimate-user approval rate at standard operating threshold |
| **Overall Accuracy** | `Standard Accuracy` | **0.8333** | Baseline uncalibrated accuracy prior to conformal abstention |
| **Probabilistic Calibration** | `Brier Score` | **0.1366** | Proper quadratic probability scoring error |
| **Probabilistic Calibration** | `Expected Calibration Error (ECE)` | **0.1581** | 10-bin empirical confidence-to-accuracy alignment |
| **Conformal Risk Governance** | `Conformal Coverage` | **86.67%** | Automated decisions issued on 86.67% of test instances (`13.33%` routed to review) |
| **Conformal Risk Governance** | `Conformal Selective Accuracy` | **90.38%** | **+7.05% accuracy gain** on automated decisions after abstaining on uncertain band `[0.0613, 0.9290]` |
| **Distributional Robustness** | `Random Covariate Shift ROC-AUC` | **0.8140** | **Zero degradation (`0.0000`)** under Gaussian covariate noise ($\sigma=0.10$) |
| **Adversarial Robustness** | `FGSM Adversarial Shift ROC-AUC` | **0.7898** | Resilient (`-0.0243` AUC degradation) under worst-case gradient perturbation ($\epsilon=0.05$) |
| **INT8 Quantization & Cascade** | `INT8 Compression / Cascade Speedup` | **1.20x / 13.03x** | `3.75 MB` FP32 $\to$ `3.13 MB` INT8 (`0.0000` max prob diff) & `13.03x` two-stage cascade speedup |
| **Graph Export Integrity** | `TorchScript Numerical Parity` | **0.0000** | Exact floating-point parity (`< 1e-7` max diff) between Eager and TorchScript graphs |

### Scientific Architectural Ablation Study (`AblationBenchmarkSuite`)

| Architectural Variant | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: |
| **1. Static Tabular + DCNv2 Only** *(No event sequence)* | `0.4582` | `0.1205` | `0.2374` | `0.3474` |
| **2. MMTT + PLR + DCNv2** *(+Sequential Transformer & PLR)* | `0.8383` | `0.4150` | `0.1109` | `0.1013` |
| **3. MMTT + RelBench Bipartite GNN + DCNv2** *(+Entity-MCC Graph)* | `0.8383` | `0.4201` | `0.1248` | `0.1324` |
| **4. Full LDM (`MMTT + GNN + TabR + DCNv2`)** *(+kNN Memory Bank)* | `0.8140` | `0.3219` | `0.1366` | `0.1581` |
| **5. In-Context PFN Head (`TabPFN v2` Cold-Start ICL)** *(Single-Pass)* | `0.7763` | `0.3080` | `0.1616` | `0.1719` |
| **6. Multi-Task MMoE Head (`Fraud + Credit + LTV`)** *(Joint 3-Task)* | `0.8032` | `0.3215` | `0.1392` | `0.1447` |

---

## Repository Structure

```text
Large-Data-Models-LDMs-/
├── .github/workflows/
│   └── ci.yml                        # Automated pytest & CPU benchmark gate (Python 3.10-3.12)
├── configs/
│   ├── config.yaml                   # Full production hyperparameter configuration
│   └── config_small.yaml             # Fast CPU verification configuration
├── docker/
│   └── Dockerfile                    # CUDA 12.1 + Python 3.10 container specification
├── docs/
│   ├── 01_fundamentos_ldm.md         # Foundations of Behavioral & Tabular Large Data Models
│   ├── 02_arquitetura_mmtt.md        # MMTT Tokenization, Periodic Embeddings (PLR) & Time2Vec
│   ├── 03_pretraining_limix.md       # Hybrid Self-Supervised Pre-Training (LimiX + PRAGMA + CoLES)
│   ├── 04_finetuning_dcnv2.md        # Supervised Fine-Tuning & Deep-Cross Fusion (DCNv2)
│   ├── 05_avaliacao_robustez.md      # Calibrated Evaluation, Conformal Abstention & Shift Testing
│   └── references.md                 # Consolidated Scientific & Executive Bibliography
├── results/
│   ├── config.json                   # Reproducible configuration snapshot from latest run
│   └── metrics_report.md             # Generated evaluation, ablation & SLA latency report
├── src/
│   ├── data/
│   │   ├── data_module.py            # LDMDataModule (stratified user split & train-only scaling)
│   │   ├── mmtt_dataset.py           # FinancialEventDataset & dynamic sequence packing collate
│   │   └── synthetic_generator.py    # Multi-archetype synthetic transaction prior generator
│   ├── evaluation/
│   │   ├── ablation_suite.py         # AblationBenchmarkSuite (6-pathway scientific ablation study)
│   │   ├── benchmarks.py             # BenchmarkSuite (Classification + Conformal + Shift + Latency)
│   │   ├── drift_monitor.py          # ConceptDriftMonitor (PSI, MMD, Wasserstein) & AdaptiveConformalAbstainer
│   │   ├── explainability.py         # LDMExplainabilityEngine (Integrated Gradients & Recourse)
│   │   ├── metrics.py                # ROC-AUC, PR-AUC, Recall@FPR, Brier, ECE & ConformalRiskAbstainer
│   │   └── strategic_shift_tester.py # Progressive covariate & adversarial concept shift simulator
│   ├── models/
│   │   ├── in_context_pfn.py         # InContextPFNHead (TabPFN v2 cold-start posterior head)
│   │   ├── joint_fusion_network.py   # CrossNetworkV2 (full/low-rank) & EndToEndFusionLayer
│   │   ├── ldm_model.py              # Unified LargeDataModel (MMTT/SSM + GNN + TabR + DCNv2 + PFN + MMoE)
│   │   ├── multi_task_moe.py         # MultiGateMixtureOfExpertsHead & HomoscedasticMultiTaskLoss
│   │   ├── relational_gnn.py         # RelationalEntityGNN & BipartiteGraphConvLayer (RelBench)
│   │   ├── retrieval_attention.py    # TabRRetrievalModule (Differentiable kNN memory retrieval)
│   │   ├── ssm_backbone.py           # SelectiveSSMBlock & HybridSSMTransformerEncoder (Mamba / Liquid-S4)
│   │   ├── transformer_backbone.py   # PeriodicNumericalEmbedding, ContinuousTimeEncoding & MMTT
│   │   └── virtual_token_Injector.py # Learnable Virtual Token Prompting & MultiModalTokenFusion
│   ├── serving/
│   │   ├── authorization_service.py  # RealTimeAuthorizationService (TorchScript + Conformal + IG)
│   │   ├── exporter.py               # LDMInferenceWrapper, TorchScript/ONNX export & SLA benchmark
│   │   └── quantization_and_cascade.py # DynamicINT8Linear & CascadeEarlyExitRouter
│   ├── training/
│   │   ├── federated.py              # FederatedLDMCoordinator (Cross-Silo FedProx + Gaussian DP)
│   │   ├── lightning_module.py       # LDMPreTrainingModule & LDMFineTuningModule
│   │   ├── lr_scheduler.py           # CosineWarmupScheduler
│   │   └── objectives.py             # ContextConditionalMaskedLoss & ContrastiveEventSequenceLoss
│   └── main.py                       # End-to-end CLI entrypoint
├── tests/
│   └── test_ldm.py                   # Comprehensive unit & invariant test suite (24 tests in pytest)
├── CITATION.cff                      # Machine-readable academic citation metadata
├── LICENSE                           # MIT License
├── pyproject.toml                    # PEP 621 build system, dependencies & tool configuration
├── requirements.txt                  # Core runtime dependencies
├── ROADMAP.md                        # Architectural roadmap & maturity governance
└── setup.py                          # Legacy setuptools compatibility shim
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
| [**Roadmap & Maturity Governance**](ROADMAP.md) | Maturity quality gates (`Stable` / `Beta` / `Planned`) and verification matrix across all 16 subsystems. |

---

## Roadmap Summary (`v0.5.0`)

See [`ROADMAP.md`](ROADMAP.md) for the complete milestone matrix and verification gates.

| Subsystem / Milestone | Status | Verification Gate |
| :--- | :---: | :--- |
| **Zero-Leakage Data Subsystem & Synthetic Prior Generator** (`src/data/`) | `Stable` | `tests/test_ldm.py::test_synthetic_generator_and_dataset_pipeline` |
| **MMTT Backbone + Periodic Numerical Embeddings (`PLR`)** (`src/models/`) | `Stable` | `tests/test_ldm.py::test_periodic_numerical_embedding_and_positional_encoding` |
| **Selective State-Space & Hybrid Backbone (`Mamba` / `Liquid-S4`)** (`src/models/ssm_backbone.py`) | `Stable` | `tests/test_ldm.py::test_selective_ssm_and_hybrid_backbone_mask_invariance` |
| **Deep & Cross Network v2 (`DCNv2`) Joint Fusion** (`src/models/`) | `Stable` | `tests/test_ldm.py::test_dcnv2_full_rank_and_low_rank_and_batch_size_one` |
| **Multi-Gate Mixture-of-Experts (`MMoE`) Multi-Task Hierarchy** (`src/models/multi_task_moe.py`) | `Stable` | `tests/test_ldm.py::test_mmoe_multitask_head_and_homoscedastic_loss` |
| **Hybrid SSL (`LimiX` + `PRAGMA` + `CoLES`)** (`src/training/`) | `Stable` | `tests/test_ldm.py::test_multimodal_ssl_and_coles_contrastive_loss` |
| **Retrieval-Augmented Behavioral Attention (`TabR`)** (`src/models/retrieval_attention.py`) | `Stable` | `tests/test_ldm.py::test_tabr_retrieval_self_masking_and_memory_bank` |
| **Relational Entity Graph Message Passing (`RelBench` GNN)** (`src/models/relational_gnn.py`) | `Stable` | `tests/test_ldm.py::test_relational_entity_gnn_mask_invariance_and_b1` |
| **In-Context Prior-Data Fitted Network Head (`TabPFN v2`)** (`src/models/in_context_pfn.py`) | `Stable` | `tests/test_ldm.py::test_in_context_pfn_head_query_order_invariance` |
| **TorchScript / ONNX Export, Dynamic INT8 & Early-Exit Cascade** (`src/serving/`) | `Stable` | `tests/test_ldm.py::test_dynamic_int8_quantization_and_cascade_early_exit_router` |
| **Regulatory Explainability & Counterfactual Recourse** (`src/evaluation/explainability.py`) | `Stable` | `tests/test_ldm.py::test_explainability_integrated_gradients_and_adverse_reasons` |
| **Cross-Silo Federated Pre-Training with Differential Privacy** (`src/training/federated.py`) | `Stable` | `tests/test_ldm.py::test_federated_dp_ssl_coordinator_and_privacy_accountant` |
| **Concept Drift Telemetry & Online Adaptive Conformal Inference (`ACI`)** (`src/evaluation/drift_monitor.py`) | `Stable` | `tests/test_ldm.py::test_concept_drift_monitor_and_adaptive_conformal_abstainer` |
| **Scientific Ablation Suite & Real-Time Authorization Service** (`src/evaluation/ablation_suite.py`) | `Stable` | `tests/test_ldm.py::test_ablation_suite_and_realtime_authorization_service` |
| **Calibrated Risk Governance & Split-Conformal Abstention** (`src/evaluation/`) | `Stable` | `tests/test_ldm.py::test_calibration_and_conformal_risk_abstainer` |

---

## Citation

If you use this software or architectural blueprint in your research or enterprise systems, please cite it using the metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{carida2026ldm,
  author       = {Carid{\'a}, Vinicius},
  title        = {{Large Data Models (LDMs): Foundation Models for Structured Tabular and Continuous-Time Transactional Sequences}},
  year         = {2026},
  version      = {0.5.0},
  url          = {https://github.com/vfcarida/Large-Data-Models-LDMs-},
  license      = {MIT}
}
```

---

## License & Attribution

Released under the [MIT License](LICENSE).

*Designed and architected by [Vinicius Caridá](https://github.com/vfcarida). Implementation and verification accelerated with AI-assisted engineering workflows.*

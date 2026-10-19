# Architectural Roadmap & Maturity Governance

This document establishes the technical roadmap, maturity governance framework, and capability milestone matrix for **Large Data Models (LDMs)**. Every milestone is anchored in peer-reviewed research and enterprise model-risk management standards, and progresses through objective verification gates.

---

## 1. Vision & Architectural North Star

Enterprise transactional decisioning—across real-time fraud prevention, credit underwriting, and anti-money laundering (AML)—operates at the intersection of **heterogeneous relational tables** and **continuous-time behavioral event streams**. Traditional architectures force organizations to maintain brittle, hand-engineered SQL/Spark aggregation pipelines that suffer from feature staleness, train-serve skew, and rapid decay under adversarial concept drift.

The **Large Data Models (LDMs)** architecture bridges this gap by unifying:
1. **Multi-Modal Temporal Transformer (MMTT) Encoders** with **Periodic Numerical Embeddings (PLR)** ([Gorishniy et al., *NeurIPS 2022*](https://arxiv.org/abs/2203.05556)) to ingest raw `(Key, Value, Timestamp)` event streams without manual window aggregation.
2. **Hybrid Self-Supervised Pre-Training** combining **Context-Conditional Multi-Modal Masked Modeling** ([LimiX, 2025](https://arxiv.org/abs/2509.03505); [PRAGMA, 2026](https://arxiv.org/abs/2604.08649)) and **Contrastive Event Sequence Learning (`CoLES`)** ([Babaev et al., *SIGMOD 2022*](https://arxiv.org/abs/2002.08649)) to prevent representation collapse and learn transferable behavioral manifolds.
3. **End-to-End Deep & Cross Fusion (DCNv2)** ([Wang et al., *WWW 2021*](https://arxiv.org/abs/2008.13535)) to capture bounded-degree polynomial interactions between deep temporal trajectories and static entity attributes.
4. **Calibrated Governance & Split-Conformal Risk Abstention** aligned with **Gartner AI TRiSM** and **SR 11-7** model risk management standards, enabling guaranteed error-rate control under distributional and strategic shift.

---

## 2. Maturity Model & Quality Gates

All subsystems in this repository are classified using an objective five-tier maturity taxonomy. A module may only advance to a higher tier when all corresponding verification criteria are satisfied by automated tests.

| Maturity Tier | Engineering & Verification Requirements |
| :--- | :--- |
| **`Stable`** | Complete implementation; strict type annotations; deterministic execution under `seed_everything`; unit and integration tests passing in `pytest`; zero known numerical stability defects; documented in `docs/`. |
| **`Beta`** | Feature-complete implementation with unit test coverage; actively undergoing multi-dataset or stress-test calibration before API freeze. |
| **`Alpha`** | Functional prototype integrated into the codebase; basic smoke tests passing; interfaces subject to refinement across minor releases. |
| **`Experimental`** | Research exploration or ablation track; isolated from default production paths. |
| **`Planned`** | Architectural specification and research anchors defined; scheduled for upcoming evolution rounds. |

---

## 3. Capability & Milestone Matrix (Shipped & Active)

| Track / Subsystem | Architectural Scope & Key Modules | Research & Industry Anchors | Maturity Status | Verification Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Data Subsystem & Zero-Leakage Pipeline** | `SyntheticTransactionGenerator`, `FinancialEventDataset`, `dynamic_sequence_packing_collate_fn`, `LDMDataModule` (`src/data/`) | Zero train-test leakage (`scikit-learn` / `PyTorch Frame`); dynamic padding mask isolation | **`Stable`** | `tests/test_ldm.py::TestDataSubsystem` (stratified user split, train-only z-score statistics, padding mask invariants) |
| **MMTT Backbone & Periodic Numerical Embeddings** | `PeriodicNumericalEmbedding` (`PLR`), `ContinuousTimeEncoding`, `MultiModalTemporalTransformer` (`src/models/transformer_backbone.py`) | Gorishniy et al. (*NeurIPS 2022* PLR); Kazemi et al. (*JMLR 2019* Time2Vec); Vaswani et al. (*NeurIPS 2017*) | **`Stable`** | `tests/test_ldm.py::TestMMTTBackbone` (PLR shape/finiteness, padding mask invariance across `cls`, `mean`, `max`, `attentive` pooling) |
| **Deep & Cross Joint Fusion Network** | `CrossNetworkV2`, `EndToEndFusionLayer` with `LayerNorm` stabilization (`src/models/joint_fusion_network.py`) | Wang et al. (*WWW 2021* DCNv2); Ba et al. (2016 LayerNorm) | **`Stable`** | `tests/test_ldm.py::TestFusionNetwork` (batch-size-1 inference compatibility, gradient flow verification) |
| **Hybrid Self-Supervised Pre-Training** | `ContextConditionalMaskedLoss` (multi-modal KVT + latent predictor), `ContrastiveEventSequenceLoss` (`CoLES`), `LDMPreTrainingModule` (`src/training/`) | LimiX (`arXiv:2509.03505`); PRAGMA (`arXiv:2604.08649`); Babaev et al. (*SIGMOD 2022* CoLES) | **`Stable`** | `tests/test_ldm.py::TestSSLAndFineTuning` (non-collapsed embedding variance, multi-modal loss convergence) |
| **Calibrated Evaluation & Conformal Risk Abstention** | `ROC-AUC`, `PR-AUC`, `Recall@FPR`, `Brier Score`, `ECE`, `ConformalRiskAbstainer`, `BenchmarkSuite` (`src/evaluation/`) | Vovk et al. (2005 Conformal Prediction); Gartner AI TRiSM; SR 11-7 Model Risk Governance | **`Stable`** | `tests/test_ldm.py::TestEvaluationAndGovernance` (metric bounds, split-conformal coverage guarantees, strategic shift harness) |
| **Adversarial & Strategic Concept Shift Testing** | `StrategicShiftTester` (`src/evaluation/strategic_shift_tester.py`) simulating progressive covariate perturbation (`0.05` to `0.50`) | Dal Pozzolo et al. (*IEEE TNNLS 2018* Concept Drift in Fraud Detection) | **`Stable`** | `tests/test_ldm.py::TestEvaluationAndGovernance::test_strategic_shift_tester_full_loader` |

---

## 4. Next-Horizon Frontier Tracks (Upcoming Rounds)

### Horizon Track 1 — Retrieval-Augmented Tabular & Behavioral Attention (`TabR` Integration)
- **Status:** `Planned` (Target: `v0.3.0`)
- **Research Anchor:** Gorishniy et al., *TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023* (*ICLR 2024*, `arXiv:2307.14338`).
- **Architectural Scope:** Introduce a differentiable nearest-neighbor retrieval module over historical entity embeddings so the fusion network can attend directly to analogous past behavioral trajectories and confirmed fraud cases during inference.

### Horizon Track 2 — Relational Entity Graph Message Passing (`RelBench` / Temporal GNNs)
- **Status:** `Planned` (Target: `v0.3.0`)
- **Research Anchor:** Robinson et al., *RelBench: A Benchmark for Deep Learning on Relational Databases* (*NeurIPS 2024*, `arXiv:2407.20060`); Ostroukhov et al., *PRAGMA* (`arXiv:2604.08649`).
- **Architectural Scope:** Extend `LDMModel` to ingest bipartite cardholder–merchant–device interaction graphs alongside sequential transactions, enabling detection of coordinated mule rings and graph-level collusion.

### Horizon Track 3 — In-Context Prior-Data Fitted Inference (`TabPFN v2` Head)
- **Status:** `Planned` (Target: `v0.4.0`)
- **Research Anchor:** Hollmann et al., *Accurate predictions on small data with a tabular foundation model* (*Nature* 637:319–326, 2025, `DOI: 10.1038/s41586-024-08328-6`).
- **Architectural Scope:** Provide an optional Prior-Data Fitted Network (PFN) classification head over frozen MMTT embeddings for rapid cold-start adaptation on newly launched payment rails with fewer than 1,000 labeled fraud examples.

### Horizon Track 4 — ONNX / TorchScript Export & Low-Latency Streaming Serving
- **Status:** `Planned` (Target: `v0.4.0`)
- **Architectural Scope:** Deterministic export of `LDMModel` to TorchScript and ONNX graphs with pre-allocated KV-caching for sub-10ms p99 real-time payment authorization SLAs.

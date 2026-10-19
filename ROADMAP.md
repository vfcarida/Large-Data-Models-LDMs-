# Architectural Roadmap & Maturity Governance

This document establishes the technical roadmap, maturity governance framework, and capability milestone matrix for **Large Data Models (LDMs)**. Every milestone is anchored in peer-reviewed research and enterprise model-risk management standards, and progresses through objective verification gates.

---

## 1. Vision & Architectural North Star

Enterprise transactional decisioning—across real-time fraud prevention, credit underwriting, and anti-money laundering (AML)—operates at the intersection of **heterogeneous relational tables**, **bipartite entity-merchant graphs**, and **continuous-time behavioral event streams**. Traditional architectures force organizations to maintain brittle, hand-engineered SQL/Spark aggregation pipelines that suffer from feature staleness, train-serve skew, and rapid decay under adversarial concept drift.

The **Large Data Models (LDMs)** architecture (`v0.5.0`) unifies:
1. **Multi-Modal Temporal Transformer (MMTT) & Selective State-Space (`Mamba` / `Liquid-S4`) Backbones** with **Periodic Numerical Embeddings (PLR)** ([Gorishniy et al., *NeurIPS 2022*](https://arxiv.org/abs/2203.05556); [Gu & Dao, *COLM 2024*](https://arxiv.org/abs/2312.00752)) to ingest raw `(Key, Value, Timestamp)` event streams at both standard and linear $O(S)$ complexity.
2. **Hybrid Self-Supervised & Federated DP Pre-Training** combining **Context-Conditional Multi-Modal Masked Modeling** ([LimiX, 2025](https://arxiv.org/abs/2509.03505); [PRAGMA, 2026](https://arxiv.org/abs/2604.08649)), **Contrastive Event Sequence Learning (`CoLES`)** ([Babaev et al., *SIGMOD 2022*](https://arxiv.org/abs/2002.08232)), and **Cross-Silo Federated Optimization (`FedProx` + $(\epsilon, \delta)$-Differential Privacy)** ([Li et al., *MLSys 2020*](https://arxiv.org/abs/1812.06127)).
3. **Bipartite Entity-Merchant Graph Message Passing (`RelBench`)** ([Robinson et al., *NeurIPS 2024*](https://arxiv.org/abs/2407.20060)) and **Differentiable kNN Context Retrieval (`TabR`)** ([Gorishniy et al., *ICLR 2024*](https://arxiv.org/abs/2307.14338)) to capture cross-entity collusion and analogous historical fraud trajectories.
4. **End-to-End Deep & Cross Fusion (`DCNv2`)**, **Multi-Gate Mixture-of-Experts (`MMoE`)** ([Ma et al., *KDD 2018*](https://dl.acm.org/doi/10.1145/3219819.3220007)), and **In-Context Prior-Data Fitted Inference (`TabPFN v2`)** ([Hollmann et al., *Nature 2025*](https://doi.org/10.1038/s41586-024-08328-6)) for single-task, multi-task (Fraud + Credit Default + LTV), and cold-start few-shot adaptation.
5. **Calibrated Governance, Online Adaptive Conformal Inference (`ACI`), Concept Drift Monitoring (`PSI` / `MMD` / `Wasserstein`), Integrated Gradients Explainability, Counterfactual Recourse & Sub-10ms TorchScript / INT8 Cascade Authorization** aligned with **Gartner AI TRiSM**, **SR 11-7**, **ECOA**, and **GDPR Article 22** standards.

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

## 3. Capability & Milestone Matrix (Shipped & Verified in `v0.5.0`)

| Track / Subsystem | Architectural Scope & Key Modules | Research & Industry Anchors | Maturity Status | Verification Gate |
| :--- | :--- | :--- | :---: | :--- |
| **Data Subsystem & Zero-Leakage Pipeline** | `SyntheticTransactionGenerator`, `FinancialEventDataset`, `dynamic_sequence_packing_collate_fn`, `LDMDataModule` (`src/data/`) | Zero train-test leakage (`scikit-learn` / `PyTorch Frame`); dynamic padding mask isolation | **`Stable`** | `tests/test_ldm.py::test_synthetic_generator_and_dataset_pipeline` |
| **MMTT Backbone & Periodic Numerical Embeddings** | `PeriodicNumericalEmbedding` (`PLR`), `ContinuousTimeEncoding`, `MMTTTransformerEncoder` (`src/models/transformer_backbone.py`) | Gorishniy et al. (*NeurIPS 2022* PLR); Kazemi et al. (*JMLR 2019* Time2Vec); Vaswani et al. (*NeurIPS 2017*) | **`Stable`** | `tests/test_ldm.py::test_periodic_numerical_embedding_and_positional_encoding`, `test_attention_and_pooling_padding_invariance` |
| **Selective State-Space & Hybrid Sequence Backbone (`Mamba` / `Liquid-S4`)** | `SelectiveSSMBlock`, `HybridSSMTransformerEncoder` (`src/models/ssm_backbone.py`) | Gu & Dao (*COLM 2024* Mamba, `arXiv:2312.00752`); Hasani et al. (*ICLR 2023* Liquid-S4) | **`Stable`** | `tests/test_ldm.py::test_selective_ssm_and_hybrid_backbone_mask_invariance` |
| **Deep & Cross Joint Fusion Network** | `CrossNetworkV2`, `EndToEndFusionLayer` with `LayerNorm` stabilization (`src/models/joint_fusion_network.py`) | Wang et al. (*WWW 2021* DCNv2); Ba et al. (2016 LayerNorm) | **`Stable`** | `tests/test_ldm.py::test_dcnv2_full_rank_and_low_rank_and_batch_size_one` |
| **Multi-Gate Mixture-of-Experts (`MMoE`) Multi-Task Hierarchy** | `MultiGateMixtureOfExpertsHead`, `HomoscedasticMultiTaskLoss`, `LargeDataModel.forward_multitask` (`src/models/multi_task_moe.py`) | Ma et al. (*KDD 2018* MMoE); Kendall et al. (*CVPR 2018* Uncertainty Weighting) | **`Stable`** | `tests/test_ldm.py::test_mmoe_multitask_head_and_homoscedastic_loss` |
| **Hybrid Self-Supervised Pre-Training** | `ContextConditionalMaskedLoss` (multi-modal KVT + latent predictor), `ContrastiveEventSequenceLoss` (`CoLES`), `LDMPreTrainingModule` (`src/training/`) | LimiX (`arXiv:2509.03505`); PRAGMA (`arXiv:2604.08649`); Babaev et al. (*SIGMOD 2022* CoLES) | **`Stable`** | `tests/test_ldm.py::test_multimodal_ssl_and_coles_contrastive_loss` |
| **Retrieval-Augmented Behavioral Attention (`TabR`)** | `TabRRetrievalModule` (`src/models/retrieval_attention.py`) with leave-one-out training mask & persistent circular memory bank | Gorishniy et al. (*ICLR 2024* TabR, `arXiv:2307.14338`) | **`Stable`** | `tests/test_ldm.py::test_tabr_retrieval_self_masking_and_memory_bank` |
| **Relational Entity-Merchant Graph GNN (`RelBench`)** | `RelationalEntityGNN`, `BipartiteGraphConvLayer` (`src/models/relational_gnn.py`) over mask-aware bipartite user-MCC incidence graphs | Robinson et al. (*NeurIPS 2024* RelBench, `arXiv:2407.20060`) | **`Stable`** | `tests/test_ldm.py::test_relational_entity_gnn_mask_invariance_and_b1` |
| **In-Context Prior-Data Fitted Inference (`TabPFN v2`)** | `InContextPFNHead`, `LargeDataModel.forward_in_context` (`src/models/in_context_pfn.py`) with asymmetric support-to-query attention | Hollmann et al. (*Nature* 637:319–326, 2025, `DOI: 10.1038/s41586-024-08328-6`) | **`Stable`** | `tests/test_ldm.py::test_in_context_pfn_head_query_order_invariance` |
| **TorchScript / ONNX Export, Dynamic INT8 & Early-Exit Cascade** | `LDMInferenceWrapper`, `export_to_torchscript`, `quantize_ldm_dynamic_int8`, `CascadeEarlyExitRouter` (`src/serving/`) | Sub-10ms p99 payment SLA, dynamic INT8 weight quantization & two-stage cascade routing | **`Stable`** | `tests/test_ldm.py::test_torchscript_export_parity_and_latency_benchmark`, `test_dynamic_int8_quantization_and_cascade_early_exit_router` |
| **Regulatory Explainability & Counterfactual Recourse** | `LDMExplainabilityEngine`, `AdverseActionReason`, `CounterfactualRecourseResult` (`src/evaluation/explainability.py`) | Sundararajan et al. (*ICML 2017* Integrated Gradients); Wachter et al. (*Harvard JOLT 2017*); ECOA / GDPR Art. 22 | **`Stable`** | `tests/test_ldm.py::test_explainability_integrated_gradients_and_adverse_reasons`, `test_counterfactual_recourse_probability_reduction` |
| **Cross-Silo Federated Pre-Training with Differential Privacy** | `FederatedLDMCoordinator`, `FederatedPrivacyConfig`, `FederatedRoundReport` (`src/training/federated.py`) | McMahan et al. (*AISTATS 2017* FedAvg); Li et al. (*MLSys 2020* FedProx); $(\epsilon, \delta)$-DP Gaussian Mechanism | **`Stable`** | `tests/test_ldm.py::test_federated_dp_ssl_coordinator_and_privacy_accountant` |
| **Concept Drift Telemetry & Online Adaptive Conformal Inference (`ACI`)** | `ConceptDriftMonitor`, `AdaptiveConformalAbstainer`, `compute_psi`, `compute_mmd_rbf`, `compute_wasserstein_1d` (`src/evaluation/drift_monitor.py`) | Gibbs & Candès (*NeurIPS 2021* ACI); Rabanser et al. (*NeurIPS 2019* Failing Loudly); SR 11-7 PSI | **`Stable`** | `tests/test_ldm.py::test_concept_drift_monitor_and_adaptive_conformal_abstainer` |
| **Scientific Ablation & Real-Time Authorization Service** | `AblationBenchmarkSuite` (`src/evaluation/ablation_suite.py`), `RealTimeAuthorizationService` (`src/serving/authorization_service.py`) | 6-pathway architectural ablation matrix & real-time conformal authorization payload | **`Stable`** | `tests/test_ldm.py::test_ablation_suite_and_realtime_authorization_service` |
| **Calibrated Evaluation & Conformal Risk Abstention** | `ROC-AUC`, `PR-AUC`, `Recall@FPR`, `Brier Score`, `ECE`, `ConformalRiskAbstainer`, `BenchmarkSuite` (`src/evaluation/`) | Vovk et al. (2005 Conformal Prediction); Gartner AI TRiSM; SR 11-7 Model Risk Governance | **`Stable`** | `tests/test_ldm.py::test_calibration_and_conformal_risk_abstainer` |
| **Adversarial & Strategic Concept Shift Testing** | `StrategicShiftTester` (`src/evaluation/strategic_shift_tester.py`) simulating progressive covariate perturbation (`0.05` to `0.50`) | Dal Pozzolo et al. (*IEEE TNNLS 2018* Concept Drift in Fraud Detection) | **`Stable`** | `tests/test_ldm.py::test_strategic_shift_and_benchmark_suite` |

---

## 4. Next-Horizon Frontier Tracks (`v0.6.0+`)

### Horizon Track A — Causal Counterfactual Treatment Uplift & Policy Optimization (`DoubleML` / `DragonNet` Head)
- **Status:** `Planned` (Target: `v0.6.0`)
- **Architectural Scope:** Causal propensity-score and conditional average treatment effect (`CATE`) estimation heads atop the LDM representation to optimize intervention policies (e.g., step-up 2FA authentication vs. hard block vs. credit-limit adjustment).

### Horizon Track B — Temporal Hypergraph Contagion & Multi-Hop Collusion Ring Detection
- **Status:** `Planned` (Target: `v0.6.0`)
- **Architectural Scope:** Higher-order hypergraph message passing connecting shared devices, IP subnets, beneficiary accounts, and merchants to uncover coordinated synthetic identity and money-mule rings.

# Architectural Roadmap & Maturity Governance

This document establishes the technical roadmap, maturity governance framework, and capability milestone matrix for **Large Data Models (LDMs)**. Every milestone is anchored in peer-reviewed research and enterprise model-risk management standards, and progresses through objective verification gates.

---

## 1. Vision & Architectural North Star

Enterprise transactional decisioning—across real-time fraud prevention, credit underwriting, and anti-money laundering (AML)—operates at the intersection of **heterogeneous relational tables**, **bipartite entity-merchant graphs**, and **continuous-time behavioral event streams**. Traditional architectures force organizations to maintain brittle, hand-engineered SQL/Spark aggregation pipelines that suffer from feature staleness, train-serve skew, and rapid decay under adversarial concept drift.

The **Large Data Models (LDMs)** architecture (`v0.7.0`) unifies:
1. **Multi-Modal Temporal Transformer (MMTT) & Selective State-Space (`Mamba` / `Liquid-S4`) Backbones** with **Periodic Numerical Embeddings (PLR)** ([Gorishniy et al., *NeurIPS 2022*](https://arxiv.org/abs/2203.05556); [Gu & Dao, *COLM 2024*](https://arxiv.org/abs/2312.00752)) to ingest raw `(Key, Value, Timestamp)` event streams at both standard and linear $O(S)$ complexity.
2. **Hybrid Self-Supervised, Federated DP, Continual Lifelong Adaptation (`EWC` + `DER++`) & Certified Machine Unlearning** combining **Context-Conditional Multi-Modal Masked Modeling** ([LimiX, 2025](https://arxiv.org/abs/2509.03505); [PRAGMA, 2026](https://arxiv.org/abs/2604.08649)), **Contrastive Event Sequence Learning (`CoLES`)** ([Babaev et al., *SIGMOD 2022*](https://arxiv.org/abs/2002.08232)), **Cross-Silo Federated Optimization (`FedProx` + $(\epsilon, \delta)$-Differential Privacy)** ([Li et al., *MLSys 2020*](https://arxiv.org/abs/1812.06127)), **Elastic Weight Consolidation & Dark Experience Distillation** ([Kirkpatrick et al., *PNAS 2017*](https://doi.org/10.1073/pnas.1611835114); [Buzzega et al., *NeurIPS 2020*](https://arxiv.org/abs/2004.07211)), and **Certified Fisher/Memory Unlearning** ([Golatkar et al., *CVPR 2020*](https://arxiv.org/abs/1911.04933)).
3. **Bipartite Entity-Merchant Graph (`RelBench`), Temporal Hypergraph Contagion (`HGNN`) & kNN Context Retrieval (`TabR`)** ([Robinson et al., *NeurIPS 2024*](https://arxiv.org/abs/2407.20060); [Feng et al., *AAAI 2019*](https://arxiv.org/abs/1809.09401); [Gorishniy et al., *ICLR 2024*](https://arxiv.org/abs/2307.14338)) to capture pairwise and multi-hop collusion rings.
4. **End-to-End Deep & Cross Fusion (`DCNv2`)**, **Multi-Gate Mixture-of-Experts (`MMoE`)** ([Ma et al., *KDD 2018*](https://dl.acm.org/doi/10.1145/3219819.3220007)), **Causal Counterfactual Treatment Uplift (`DragonNet`)** ([Shi et al., *NeurIPS 2019*](https://arxiv.org/abs/1906.02120)), **Conditional Optimal-Transport Flow Matching (`FlowLDM` / `TabDDPM`)** ([Lipman et al., *ICLR 2023*](https://arxiv.org/abs/2210.02747); [Kotelnikov et al., *ICML 2023*](https://arxiv.org/abs/2209.15421)), and **In-Context Prior-Data Fitted Inference (`TabPFN v2`)** ([Hollmann et al., *Nature 2025*](https://doi.org/10.1038/s41586-024-08328-6)).
5. **Calibrated Governance, Algorithmic Fairness (`EqualizedOdds`), Tamper-Evident `HMAC-SHA256` Audit Ledger & Statistical Watermarking (`EU AI Act Art. 12 & 50`), Online Adaptive Conformal Inference (`ACI`), Concept Drift Monitoring (`PSI` / `MMD` / `Wasserstein`), Integrated Gradients Explainability, Counterfactual Recourse & Sub-10ms TorchScript / INT8 / 4-Bit `NF4` / `DoRA` Cascade Authorization** aligned with **Gartner AI TRiSM**, **SR 11-7**, **ECOA**, **GDPR Art. 17 & 22**, and **LGPD Art. 18** standards.

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

## 3. Capability & Milestone Matrix (Shipped & Verified in `v0.7.0`)

| Track / Subsystem | Architectural Scope & Key Modules | Research & Industry Anchors | Maturity Status | Verification Gate |
| :--- | :--- | :--- | :---: | :--- |
| **Data Subsystem & Zero-Leakage Pipeline** | `SyntheticTransactionGenerator`, `FinancialEventDataset`, `dynamic_sequence_packing_collate_fn`, `LDMDataModule` (`src/data/`) | Zero train-test leakage (`scikit-learn` / `PyTorch Frame`); dynamic padding mask isolation | **`Stable`** | `tests/test_ldm.py::test_synthetic_generator_and_dataset_pipeline` |
| **MMTT Backbone & Periodic Numerical Embeddings** | `PeriodicNumericalEmbedding` (`PLR`), `ContinuousTimeEncoding`, `MMTTTransformerEncoder` (`src/models/transformer_backbone.py`) | Gorishniy et al. (*NeurIPS 2022* PLR); Kazemi et al. (*JMLR 2019* Time2Vec); Vaswani et al. (*NeurIPS 2017*) | **`Stable`** | `tests/test_ldm.py::test_periodic_numerical_embedding_and_positional_encoding`, `test_attention_and_pooling_padding_invariance` |
| **Selective State-Space & Hybrid Sequence Backbone (`Mamba` / `Liquid-S4`)** | `SelectiveSSMBlock`, `HybridSSMTransformerEncoder` (`src/models/ssm_backbone.py`) | Gu & Dao (*COLM 2024* Mamba, `arXiv:2312.00752`); Hasani et al. (*ICLR 2023* Liquid-S4) | **`Stable`** | `tests/test_ldm.py::test_selective_ssm_and_hybrid_backbone_mask_invariance` |
| **Deep & Cross Joint Fusion Network** | `CrossNetworkV2`, `EndToEndFusionLayer` with `LayerNorm` stabilization (`src/models/joint_fusion_network.py`) | Wang et al. (*WWW 2021* DCNv2); Ba et al. (2016 LayerNorm) | **`Stable`** | `tests/test_ldm.py::test_dcnv2_full_rank_and_low_rank_and_batch_size_one` |
| **Multi-Gate Mixture-of-Experts (`MMoE`) Multi-Task Hierarchy** | `MultiGateMixtureOfExpertsHead`, `HomoscedasticMultiTaskLoss`, `LargeDataModel.forward_multitask` (`src/models/multi_task_moe.py`) | Ma et al. (*KDD 2018* MMoE); Kendall et al. (*CVPR 2018* Uncertainty Weighting) | **`Stable`** | `tests/test_ldm.py::test_mmoe_multitask_head_and_homoscedastic_loss` |
| **Causal Counterfactual Treatment Uplift (`DragonNet`)** | `CausalDragonNetHead`, `DragonNetTargetedLoss`, `recommend_optimal_intervention` (`src/models/causal_uplift.py`) | Shi, Blei, & Veitch (*NeurIPS 2019* DragonNet, `arXiv:1906.02120`) | **`Stable`** | `tests/test_ldm.py::test_causal_dragonnet_uplift_and_policy_optimizer` |
| **Conditional Flow-Matching Trajectory Synthesis (`FlowLDM`)** | `ConditionalTrajectoryFlowMatcher`, `LargeDataModel.synthesize_counterfactual_trajectories` (`src/models/flow_matching.py`) | Lipman et al. (*ICLR 2023* OT-CFM); Kotelnikov et al. (*ICML 2023* TabDDPM) | **`Stable`** | `tests/test_ldm.py::test_conditional_flow_matching_loss_and_heun_synthesis` |
| **Temporal Hypergraph Contagion & Collusion Rings (`HGNN`)** | `HypergraphConvLayer`, `TemporalHypergraphContagionModule` (`src/models/hypergraph_contagion.py`) | Feng et al. (*AAAI 2019* HGNN); Chien et al. (*ICLR 2022* AllSet) | **`Stable`** | `tests/test_ldm.py::test_temporal_hypergraph_contagion_mask_invariance` |
| **Hybrid Self-Supervised Pre-Training** | `ContextConditionalMaskedLoss` (multi-modal KVT + latent predictor), `ContrastiveEventSequenceLoss` (`CoLES`), `LDMPreTrainingModule` (`src/training/`) | LimiX (`arXiv:2509.03505`); PRAGMA (`arXiv:2604.08649`); Babaev et al. (*SIGMOD 2022* CoLES) | **`Stable`** | `tests/test_ldm.py::test_multimodal_ssl_and_coles_contrastive_loss` |
| **Retrieval-Augmented Behavioral Attention (`TabR`)** | `TabRRetrievalModule` (`src/models/retrieval_attention.py`) with leave-one-out training mask & persistent circular memory bank | Gorishniy et al. (*ICLR 2024* TabR, `arXiv:2307.14338`) | **`Stable`** | `tests/test_ldm.py::test_tabr_retrieval_self_masking_and_memory_bank` |
| **Relational Entity-Merchant Graph GNN (`RelBench`)** | `RelationalEntityGNN`, `BipartiteGraphConvLayer` (`src/models/relational_gnn.py`) over mask-aware bipartite user-MCC incidence graphs | Robinson et al. (*NeurIPS 2024* RelBench, `arXiv:2407.20060`) | **`Stable`** | `tests/test_ldm.py::test_relational_entity_gnn_mask_invariance_and_b1` |
| **In-Context Prior-Data Fitted Inference (`TabPFN v2`)** | `InContextPFNHead`, `LargeDataModel.forward_in_context` (`src/models/in_context_pfn.py`) with asymmetric support-to-query attention | Hollmann et al. (*Nature* 637:319–326, 2025, `DOI: 10.1038/s41586-024-08328-6`) | **`Stable`** | `tests/test_ldm.py::test_in_context_pfn_head_query_order_invariance` |
| **TorchScript / ONNX Export, Dynamic INT8 & Early-Exit Cascade** | `LDMInferenceWrapper`, `export_to_torchscript`, `quantize_ldm_dynamic_int8`, `CascadeEarlyExitRouter` (`src/serving/`) | Sub-10ms p99 payment SLA, dynamic INT8 weight quantization & two-stage cascade routing | **`Stable`** | `tests/test_ldm.py::test_torchscript_export_parity_and_latency_benchmark`, `test_dynamic_int8_quantization_and_cascade_early_exit_router` |
| **PEFT (`LoRA` / `DoRA`) & Blockwise 4-Bit `NF4` Quantization** | `LoRALinear`, `BlockwiseNF4Linear`, `inject_lora_adapters`, `merge_lora_adapters`, `quantize_ldm_nf4` (`src/serving/peft_and_nf4.py`) | Hu et al. (*ICLR 2022* LoRA); Liu et al. (*ICML 2024 Oral* DoRA); Dettmers et al. (*NeurIPS 2023* QLoRA NF4) | **`Stable`** | `tests/test_ldm.py::test_peft_dora_adapters_zero_latency_merge_and_nf4_quantization` |
| **Regulatory Explainability & Counterfactual Recourse** | `LDMExplainabilityEngine`, `AdverseActionReason`, `CounterfactualRecourseResult` (`src/evaluation/explainability.py`) | Sundararajan et al. (*ICML 2017* Integrated Gradients); Wachter et al. (*Harvard JOLT 2017*); ECOA / GDPR Art. 22 | **`Stable`** | `tests/test_ldm.py::test_explainability_integrated_gradients_and_adverse_reasons`, `test_counterfactual_recourse_probability_reduction` |
| **Cross-Silo Federated Pre-Training with Differential Privacy** | `FederatedLDMCoordinator`, `FederatedPrivacyConfig`, `FederatedRoundReport` (`src/training/federated.py`) | McMahan et al. (*AISTATS 2017* FedAvg); Li et al. (*MLSys 2020* FedProx); $(\epsilon, \delta)$-DP Gaussian Mechanism | **`Stable`** | `tests/test_ldm.py::test_federated_dp_ssl_coordinator_and_privacy_accountant` |
| **Continual Lifelong Adaptation (`EWC` + `DER++`)** | `ContinualLDMAdapter`, `ContinualAdaptationReport` (`src/training/continual.py`) | Kirkpatrick et al. (*PNAS 2017* EWC); Buzzega et al. (*NeurIPS 2020* DER++) | **`Stable`** | `tests/test_ldm.py::test_continual_ewc_and_dark_experience_lifelong_adaptation` |
| **Certified Machine Unlearning (`GDPR Art. 17` / `LGPD Art. 18`)** | `CertifiedLDMUnlearner`, `UnlearningCertificate` (`src/training/unlearning.py`) | Golatkar et al. (*CVPR 2020* Fisher Scrubbing); Bourtoule et al. (*IEEE S&P 2021* Unlearning) | **`Stable`** | `tests/test_ldm.py::test_certified_machine_unlearning_tabr_and_fisher_scrubbing` |
| **Algorithmic Fairness Auditor & Equalized-Odds Optimizer** | `FairnessGovernanceAuditor`, `EqualizedOddsThresholdOptimizer` (`src/evaluation/fairness.py`) | Hardt, Price, & Srebro (*NeurIPS 2016* Equalized Odds); ECOA 80% Four-Fifths Rule | **`Stable`** | `tests/test_ldm.py::test_fairness_governance_auditor_and_equalized_odds_optimizer` |
| **Tamper-Evident Audit Ledger & Statistical Output Watermarking** | `TamperEvidentAuditLedger`, `AuditRecord`, `StatisticalLogitWatermarker` (`src/evaluation/audit_ledger.py`) | EU AI Act Articles 12 (Record-Keeping) & 50 (Watermarking); SR 11-7 Audit Trail | **`Stable`** | `tests/test_ldm.py::test_tamper_evident_audit_ledger_and_logit_watermarking` |
| **Concept Drift Telemetry & Online Adaptive Conformal Inference (`ACI`)** | `ConceptDriftMonitor`, `AdaptiveConformalAbstainer`, `compute_psi`, `compute_mmd_rbf`, `compute_wasserstein_1d` (`src/evaluation/drift_monitor.py`) | Gibbs & Candès (*NeurIPS 2021* ACI); Rabanser et al. (*NeurIPS 2019* Failing Loudly); SR 11-7 PSI | **`Stable`** | `tests/test_ldm.py::test_concept_drift_monitor_and_adaptive_conformal_abstainer` |
| **Scientific Ablation & Real-Time Authorization Service** | `AblationBenchmarkSuite` (`src/evaluation/ablation_suite.py`), `RealTimeAuthorizationService` (`src/serving/authorization_service.py`) | 6-pathway architectural ablation matrix & real-time conformal authorization payload | **`Stable`** | `tests/test_ldm.py::test_ablation_suite_and_realtime_authorization_service` |
| **Calibrated Evaluation & Conformal Risk Abstention** | `ROC-AUC`, `PR-AUC`, `Recall@FPR`, `Brier Score`, `ECE`, `ConformalRiskAbstainer`, `BenchmarkSuite` (`src/evaluation/`) | Vovk et al. (2005 Conformal Prediction); Gartner AI TRiSM; SR 11-7 Model Risk Governance | **`Stable`** | `tests/test_ldm.py::test_calibration_and_conformal_risk_abstainer` |
| **Adversarial & Strategic Concept Shift Testing** | `StrategicShiftTester` (`src/evaluation/strategic_shift_tester.py`) simulating progressive covariate perturbation (`0.05` to `0.50`) | Dal Pozzolo et al. (*IEEE TNNLS 2018* Concept Drift in Fraud Detection) | **`Stable`** | `tests/test_ldm.py::test_strategic_shift_and_benchmark_suite` |

---

## 4. Next-Horizon Frontier Tracks (`v0.8.0+`)

### Horizon Track A — Hawkes Process Neural Point-Process Intensity Head (`NeuralHawkesLDM`)
- **Status:** `Planned` (Target: `v0.8.0`)
- **Architectural Scope:** Self-exciting continuous-time hazard intensity modeling $\lambda^*(t)$ for predicting exact next-transaction arrival times and burst cascades.

### Horizon Track B — Multi-Currency Cross-Border Hierarchical Entity Alignment (`CrossBorderLDM`)
- **Status:** `Planned` (Target: `v0.8.0`)
- **Architectural Scope:** Contrastive entity resolution and FX-invariant purchasing-power parity embeddings across multi-region payment rails.

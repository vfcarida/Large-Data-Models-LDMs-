# Scientific References & Executive Governance Bibliography

All architectural components, self-supervised objectives, numerical embeddings, and risk evaluation protocols in **LDM Research Lab** are anchored in peer-reviewed literature and enterprise risk governance standards.

---

## 1. Tabular & Transactional Foundation Models

* **TabPFN v2 — Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025).**
  *Accurate predictions on small data with a tabular foundation model.* **Nature**, 637, 319–326. `DOI: 10.1038/s41586-024-08328-6`
  * **Relevance:** Establishes synthetic prior-data generation with structural causal and behavioral priors as a core pillar of tabular foundation models.
* **TransactionGPT — Dou, Y., Jiang, Z., Zhang, T., Hu, M., Xu, Z., et al. (Visa Research, 2025).**
  *TransactionGPT: Foundation Models for Payment Trajectories.* `arXiv:2511.08939`
  * **Relevance:** Introduces 3D-Transformer / Multi-Modal-Temporal-Tabular (MMTT) representations and Virtual Token Layers for global context conditioning on payment trajectories.
* **PRAGMA — Ostroukhov, M., Mikhailov, R., Iashin, V., Sokolov, A., et al. (Revolut & NVIDIA, 2026).**
  *PRAGMA: A Self-Supervised Foundation Model for Banking Events.* `arXiv:2604.08649`
  * **Relevance:** Formalizes Key-Value-Time (KVT) tokenization and multi-modal masked event reconstruction for banking sequences.
* **LimiX — LimiX Team (2025/2026).**
  *LimiX: Masked Joint-Distribution Pre-training for Structured Data.* `arXiv:2509.03505` / *LimiX-2* `arXiv:2609.17488`
  * **Relevance:** Introduces heterogeneous cell and contiguous block masking across the joint distribution to prevent trivial local interpolation during pre-training.
* **CoLES — Babaev, D., Ovsov, N., Kireev, I., Ivanova, M., Gusev, G., Nazarov, I., & Tuzhilin, A. (2022).**
  *CoLES: Contrastive Learning for Event Sequences with Self-Supervision.* **Proceedings of the ACM SIGMOD International Conference on Management of Data (SIGMOD '22)**, `arXiv:2002.08232`
  * **Relevance:** Prevents representation collapse in self-supervised event sequence encoders via subsequence contrastive alignment (`ContrastiveEventSequenceLoss`).
* **TabFormer / TabGPT — Padhi, I., Schiff, Y., Melnyk, I., Rigotti, M., Mroueh, Y., et al. (IBM Research, 2021).**
  *Tabular Transformers for Modeling Multivariate Time Series.* **IEEE ICASSP 2021**, `arXiv:2011.01843`
  * **Relevance:** Pioneered hierarchical tabular transformers and synthetic credit card transaction benchmarks for fraud detection.

---

## 2. Numerical Embeddings, Retrieval & Hybrid Tabular Fusion

* **Periodic Numerical Embeddings (PLR) — Gorishniy, Y., Rubachev, I., & Babenko, A. (2022).**
  *On Embeddings for Numerical Features in Tabular Deep Learning.* **Advances in Neural Information Processing Systems (NeurIPS 2022)**, `arXiv:2203.05556`
  * **Relevance:** Proves that Periodic + Linear + Activation (`PeriodicNumericalEmbedding`) overcomes spectral bias on continuous tabular scalars.
* **TabR — Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., & Babenko, A. (2024).**
  *TabR: Tabular Deep Learning Meets Nearest Neighbors.* **International Conference on Learning Representations (ICLR 2024)**, `arXiv:2307.14338`
  * **Relevance:** Demonstrates the power of attention-weighted context aggregation in tabular deep learning.
* **CARTE — Kim, M. J., Grinsztajn, L., & Varoquaux, G. (2024).**
  *CARTE: Pretraining and Transfer for Tabular Learning.* **International Conference on Machine Learning (ICML 2024)**, `arXiv:2402.16785`
  * **Relevance:** Formalizes transfer learning and differential fine-tuning protocols for structured tables.
* **RelBench — Robinson, J., Ranjan, R., Hu, W., Fey, M., Leskovec, J., et al. (2024).**
  *RelBench: A Benchmark for Deep Learning on Relational Databases.* **NeurIPS 2024 Datasets and Benchmarks Track**, `arXiv:2407.20060`
  * **Relevance:** Defines entity-centric splitting and temporal event aggregation standards.
* **DCN V2 — Wang, R., Shivanna, R., Cheng, D., Jain, S., Lin, D., Hong, L., & Chi, E. (Google Research, 2021).**
  *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems.* **Proceedings of the Web Conference (WWW 2021)**, `arXiv:2008.13535`
  * **Relevance:** Foundation for `CrossNetworkV2` and `EndToEndFusionLayer`, computing explicit bounded-degree polynomial feature interactions.

---

## 3. Imbalanced Evaluation, Calibration, Conformal Prediction & Robustness

* **Precision-Recall under Imbalance — Saito, T., & Rehmsmeier, M. (2015).**
  *The Precision-Recall Plot Is More Informative than the ROC Plot when Evaluating Binary Classifiers on Imbalanced Datasets.* **PLOS ONE**, 10(3), e0118432.
  * **Relevance:** Anchors `calculate_pr_auc` and `calculate_recall_at_fpr` in `src/evaluation/metrics.py`.
* **Neural Calibration — Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).**
  *On Calibration of Modern Neural Networks.* **International Conference on Machine Learning (ICML 2017)**, `arXiv:1706.04599`
  * **Relevance:** Anchors `calculate_ece` (Expected Calibration Error) and `calculate_brier_score`.
* **Conformal Prediction — Angelopoulos, A. N., & Bates, S. (2023).**
  *Conformal Prediction: A Gentle Introduction.* **Foundations and Trends in Machine Learning**, 16(4), 494–591, `arXiv:2107.07511`
  * **Relevance:** Anchors `ConformalRiskAbstainer` for finite-sample selective risk abstention.
* **Adversarial Examples (FGSM) — Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).**
  *Explaining and Harnessing Adversarial Examples.* **ICLR 2015**, `arXiv:1412.6572`
  * **Relevance:** Anchors `StrategicShiftTester.apply_gradient_adversarial_shift`.
* **LoRA — Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022).**
  *LoRA: Low-Rank Adaptation of Large Language Models.* **ICLR 2022**, `arXiv:2106.09685`
  * **Relevance:** Anchors `inject_lora_adaptation` for rapid post-deployment adaptation under strategic shift.
* **DoRA — Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024).**
  *DoRA: Weight-Decomposed Low-Rank Adaptation.* **ICML 2024**, `arXiv:2402.09353`
  * **Relevance:** Anchors `DoRALinear` and `inject_dora_adaptation`, decoupling weight magnitude from directional low-rank updates.
* **QLoRA / NF4 — Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).**
  *QLoRA: Efficient Finetuning of Quantized LLMs.* **NeurIPS 2023**, `arXiv:2305.14314`
  * **Relevance:** Anchors `NF4QuantizedLinear` and `quantize_ldm_nf4` for 4-bit NormalFloat quantization of frozen LDM backbones.

---

## 4. Continuous-Time Generative Dynamics, Point Processes & Continual Learning

* **Flow Matching — Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023).**
  *Flow Matching for Generative Modeling.* **ICLR 2023**, `arXiv:2210.02747`
  * **Relevance:** Anchors `ConditionalFlowMatchingHead` for continuous-time ODE velocity field forecasting and counterfactual trajectory generation.
* **Neural Hawkes Process — Mei, H., & Eisner, J. M. (2017).**
  *The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process.* **NeurIPS 2017**, `arXiv:1612.09328`
  * **Relevance:** Anchors `NeuralHawkesProcessHead` for self-exciting continuous-time hazard intensity $\lambda^*(t \mid \mathcal{H}_t)$ and bursty botnet velocity scoring.
* **Transformer Hawkes Process — Zuo, S., Jiang, H., Li, Z., Zhao, T., & Zha, H. (2020).**
  *Transformer Hawkes Process.* **ICML 2020**, `arXiv:2002.09291`
  * **Relevance:** Anchors attention-conditioned temporal point process intensity parameterization on MMTT trajectory states.
* **Elastic Weight Consolidation (EWC) — Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., et al. (2017).**
  *Overcoming catastrophic forgetting in neural networks.* **PNAS**, 114(13), 3521–3526, `arXiv:1612.00796`
  * **Relevance:** Anchors `ContinualLdmTrainer` quadratic Fisher Information regularization across non-stationary fraud regimes.
* **Dark Experience Replay (DER++) — Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020).**
  *Dark Experience for General Continual Learning: a Strong, Simple Baseline.* **NeurIPS 2020**, `arXiv:2004.07211`
  * **Relevance:** Anchors `ReservoirDarkReplayBuffer` for bounded-memory dark logit distillation.

---

## 5. Cross-Border Domain Alignment, Certified Monotonicity & Contextual Bandits

* **Deep CORAL — Sun, B., & Saenko, K. (2016).**
  *Deep CORAL: Correlation Alignment for Deep Domain Adaptation.* **ECCV 2016**, `arXiv:1607.01719`
  * **Relevance:** Anchors `CrossBorderDomainAligner` and `FXPurchasingPowerNormalizer` for cross-border multi-currency representation invariance.
* **Deep Lattice & Partial Monotonic Networks — You, S., Ding, D., Canini, K., Pfeifer, J., & Gupta, M. (2017).**
  *Deep Lattice Networks and Partial Monotonic Functions.* **NeurIPS 2017**, `arXiv:1709.06680`
  * **Relevance:** Anchors `MonotonicConstrainedHead` and `MonotonicityCertificationAuditor` for zero-violation partial monotonicity under SR 11-7 / ECOA.
* **Constrained Monotonic Neural Networks — Runje, D., & Shankaranarayana, S. M. (2023).**
  *Constrained Monotonic Neural Networks.* **ICML 2023**, `arXiv:2205.11775`
  * **Relevance:** Anchors non-negative weight projection (`softplus`) and monotone hidden activations for certified risk heads.
* **LinUCB — Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).**
  *A Contextual-Bandit Approach to Personalized News Article Recommendation.* **WWW 2010**, `arXiv:1003.0146`
  * **Relevance:** Anchors `NeuralLinUCBRouter` for upper-confidence-bound multi-action transaction routing with Sherman-Morrison online updates.
* **Doubly Robust Off-Policy Evaluation (DR-OPE) — Dudík, M., Langford, J., & Li, L. (2011).**
  *Doubly Robust Policy Evaluation and Learning.* **ICML 2011**, `arXiv:1103.4601`
  * **Relevance:** Anchors `DoublyRobustOffPolicyEvaluator` for unbiased counterfactual policy value estimation prior to production rollout.
* **TabICL — Qu, Y., et al. (2025).**
  *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.* **ICML 2025**, `arXiv:2502.05564`
  * **Relevance:** Anchors `InContextTabularPredictor` for Bayesian support-query cross-attention over `LargeDataModel` representations.
* **DeepHit — Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018).**
  *DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks.* **AAAI 2018**.
  * **Relevance:** Anchors `CompetingRisksSurvivalHead` for right-censored cause-specific Cumulative Incidence Functions (`CIF`) and IFRS 9 / CECL Expected Credit Loss (`ECL`).
* **Rényi Differential Privacy (RDP) — Mironov, I. (2017).**
  *Rényi Differential Privacy.* **IEEE 30th Computer Security Foundations Symposium (CSF 2017)**, `arXiv:1702.07476`
  * **Relevance:** Anchors `RdpPrivacyAccountant` and `DifferentiallyPrivateFedCoordinator` for cross-silo inter-bank federated training (`DP-FedAvg`).
* **Randomized Smoothing — Cohen, J., Rosenfeld, E., & Kolter, Z. (2019).**
  *Certified Adversarial Robustness via Randomized Smoothing.* **ICML 2019**, `arXiv:1902.02918`
  * **Relevance:** Anchors `RandomizedSmoothingCertifier` and `clopper_pearson_lower_bound` for provable $\ell_2$ adversarial radius certificates.
* **Adversarial Classification — Dalvi, N., Domingos, P., Mausam, Sanghai, S., & Verma, D. (2004).**
  *Adversarial Classification.* **KDD 2004**.
  * **Relevance:** Anchors `CostAwareStackelbergRedTeam` asymmetric feature manipulation costs ($c_j$) and rational adversary utility maximization.
* **Stackelberg Prediction Games — Brückner, M., Kanzow, C., & Scheffer, T. (2012).**
  *Static Prediction Games for Adversarial Learning Problems.* **JMLR**, 13, 2617–2654.
  * **Relevance:** Anchors `CostAwareStackelbergRedTeam` bilevel Leader-Follower Stackelberg equilibrium thresholding and minimax hardening loss.
* **Strategic Classification — Hardt, M., Megiddo, N., Papadimitriou, C., & Wootters, M. (2016).**
  *Strategic Classification.* **ITCS 2016**, `arXiv:1506.06980`
  * **Relevance:** Anchors economic break-even evasion analysis (`break_even_rate`) under heterogeneous manipulation cost constraints.
* **Delegating Computation / Verifiable Proofs — Goldwasser, S., Kalai, Y. T., & Rothblum, G. N. (2015).**
  *Delegating Computation: Interactive Proofs for Muggles.* **Journal of the ACM (JACM)**, 62(4), Article 27.
  * **Relevance:** Anchors `VerifiableLDMCommitmentEngine` layer-by-layer execution trace verification and cryptographic soundness for delegated inference.
* **ZKML Verifiable Inference — Kang, D., Hashimoto, T., Stoica, I., & Sun, Y. (2022).**
  *Scaling up Trustless DNN Inference with Zero-Knowledge Proofs.* **arXiv:2210.08674 / PVLDB**.
  * **Relevance:** Anchors `VerifiableLDMCommitmentEngine` and `VerifiableInferenceReceipt` binding model weight Merkle roots ($\mathcal{R}_W$), blinded input commitments ($\mathcal{C}_{\text{in}}$), and forward-hook intermediate activation trace Merkle roots ($\mathcal{R}_H$).

---

## 6. Enterprise Risk & Regulatory Governance Frameworks

* **Board of Governors of the Federal Reserve System & OCC (2011 / 2026).** *SR 11-7: Guidance on Model Risk Management* & *SR 26-2 Interagency AI Risk Guidance*.
* **Gartner (2024–2026).** *AI TRiSM (AI Trust, Risk, and Security Management)* — Runtime inspection, epistemic abstention guardrails, and adversarial resilience.
* **Boston Consulting Group (BCG, 2024–2026).** *AI at Scale in Financial Services ("Risk-First, Value-Fast")* — Operational false-positive budget governance (`Recall@FPR`).
* **McKinsey & Company / QuantumBlack (2024–2026).** *Next-Generation Financial Crime & Hybrid Foundation Risk Modeling*.
* **RFC 6962 — Laurie, B., Langley, A., & Kasper, E. (2013).** *Certificate Transparency* — Anchors `CryptographicAuditLedger` and `VerifiableLDMCommitmentEngine` SHA-256 hash-chain and Merkle root verification.

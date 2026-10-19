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

---

## 4. Enterprise Risk & Regulatory Governance Frameworks

* **Board of Governors of the Federal Reserve System & OCC (2011 / 2026).** *SR 11-7: Guidance on Model Risk Management* & *SR 26-2 Interagency AI Risk Guidance*.
* **Gartner (2024–2026).** *AI TRiSM (AI Trust, Risk, and Security Management)* — Runtime inspection, epistemic abstention guardrails, and adversarial resilience.
* **Boston Consulting Group (BCG, 2024–2026).** *AI at Scale in Financial Services ("Risk-First, Value-Fast")* — Operational false-positive budget governance (`Recall@FPR`).
* **McKinsey & Company / QuantumBlack (2024–2026).** *Next-Generation Financial Crime & Hybrid Foundation Risk Modeling*.

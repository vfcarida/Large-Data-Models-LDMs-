# 1. Foundations: What Are Large Data Models (LDMs)?

Welcome to the **LDM Research Lab**. This guide introduces the scientific and architectural foundations of **Large Data Models (LDMs)**—self-supervised foundation models purpose-built for structured, tabular, and temporal transaction sequences.

---

## Why Standard Large Language Models (LLMs) Fail on Transactional Data

While autoregressive Large Language Models (LLMs) excel at natural language processing, serializing structured financial tables and high-frequency transaction logs into text strings introduces three fundamental bottlenecks:

1. **Loss of Numerical Geometry and Spectral Bias:**
   Subword tokenizers (BPE, WordPiece) fragment continuous floating-point numbers into arbitrary discrete tokens (e.g., `1542.50` $\rightarrow$ `[15, 42, ., 50]`), destroying Euclidean distance and monotonic magnitude relationships. Even when raw scalars are passed into standard linear layers, neural networks suffer from spectral bias—failing to resolve high-frequency decision boundaries that tree ensembles capture naturally (Gorishniy et al., *NeurIPS 2022*).
2. **Quadratic Context Inflation:**
   Converting a 200-transaction customer history into verbose text templates (`"Customer spent $45.20 at merchant category 5411 on..."`) inflates sequence length by $10\times$–$30\times$, causing $\mathcal{O}(S^2)$ self-attention memory and latency explosions.
3. **Non-Linguistic Joint Distributions:**
   Financial behavior is governed by heavy-tailed log-normal amount distributions, Zipfian merchant category frequencies, circadian/payday temporal rhythms, and cross-sectional credit bureau correlations rather than natural language syntax.

---

## What Is a Large Data Model (LDM)?

A **Large Data Model (LDM)** is a foundation model architecture designed natively for **Multi-Modal-Temporal-Tabular (MMTT)** event trajectories and relational attributes:

* **Native Multi-Modal Tokenization (KVT):** Rather than converting rows to text, each event is encoded as a **Key-Value-Time** triplet—preserving discrete categorical semantics, continuous numerical precision (via Periodic Numerical Embeddings), and inter-event temporal dynamics (Ostroukhov et al., *PRAGMA*, 2026; Dou et al., *TransactionGPT*, 2025).
* **Prior-Data Structural Grounding:** Inspired by **TabPFN v2** (Hollmann et al., *Nature* 2025), LDMs can be pre-trained and stress-tested on synthetic prior-data generators that encode realistic structural causal mechanisms and multi-pattern fraud trajectories.
* **Hybrid Static-Dynamic Fusion:** Enterprise risk decisions depend on both dynamic event trajectories (card swipes, wire transfers) and static tabular attributes (credit bureau scores, verified income, account tenure). LDMs fuse pooled sequence representations with static features via explicit polynomial feature crossing (**DCNv2**, Wang et al., *WWW* 2021).

---

## The Two-Phase Foundation Model Lifecycle

### Phase 1: Collapse-Free Self-Supervised Pre-Training (Unlabeled Data)
In financial services, confirmed fraud and default labels are scarce, delayed, and expensive to audit, whereas unlabeled transaction logs are abundant. During pre-training, the LDM learns behavioral representations without human labels by combining:
1. **Masked Joint-Distribution Reconstruction (LimiX / PRAGMA):** Corrupting contiguous blocks and cells of Key-Value-Time sequences and reconstructing the true underlying merchant categories, amounts, and temporal deltas.
2. **Contrastive Event Sequence Learning (CoLES):** Aligning stochastic subsequence views of the same user on the unit hypersphere while separating distinct users (Babaev et al., *SIGMOD* 2022), mathematically preventing representation collapse.

### Phase 2: Supervised Fine-Tuning & Calibrated Risk Governance (Labeled Data)
The pre-trained `MMTTTransformerEncoder` is paired with the `EndToEndFusionLayer` (`CrossNetworkV2` + parallel deep MLP) and fine-tuned using differential learning rates and class-weighted Binary Cross-Entropy loss. Beyond raw discrimination, the model is audited for **Expected Calibration Error (ECE)**, **Precision-Recall AUC (PR-AUC)**, **Recall at Fixed False Positive Rate (`Recall@FPR`)**, and **Conformal Risk Abstention** for human-in-the-loop governance.

---

**Next Step:** Read [02_arquitetura_mmtt.md](02_arquitetura_mmtt.md) to explore how raw transactions are transformed into Key-Value-Time tensors and Periodic Numerical Embeddings.

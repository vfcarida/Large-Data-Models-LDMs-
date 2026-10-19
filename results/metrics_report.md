# LDM Training & Governance Evaluation Report

**Generated**: 2026-10-19 15:43:49
**Config**: small

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | 64 |
| Transformer Layers | 2 |
| Attention Heads | 2 |
| Numerical Embedding | Periodic Numerical Embedding (PLR, 16 frequencies) |
| DCNv2 Cross Layers | 2 |
| Vocabulary Size | 20 |
| Dropout | 0.1 |

## Dataset

| Parameter | Value |
|-----------|-------|
| Number of Users | 400 |
| Transactions/User | 12-50 |
| Fraud Rate | 12.0% |
| Max Sequence Length | 64 |

## Training Summary

| Phase | Epochs | Learning Rate | Duration |
|-------|--------|--------------|----------|
| Pre-training (LimiX + PRAGMA + CoLES) | 3 | 0.001 | 17.4s |
| Fine-tuning (DCNv2 Classification) | 8 | 0.002 | 28.2s |

## Test & Governance Evaluation Results

| Metric | Value |
|--------|-------|
| test/loss | 1.8318 |
| test/auc | 0.8167 |
| test/pr_auc | 0.3100 |
| test/f1 | 0.2222 |
| test/recall_at_5pct_fpr | 0.1429 |
| test/brier_score | 0.1860 |
| test/ece | 0.1951 |
| benchmark/auc | 0.8167 |
| benchmark/pr_auc | 0.3100 |
| benchmark/f1_score | 0.2222 |
| benchmark/recall_at_5pct_fpr | 0.1429 |
| benchmark/brier_score | 0.1860 |
| benchmark/ece | 0.1951 |
| benchmark/precision | 0.1818 |
| benchmark/recall | 0.2857 |
| benchmark/specificity | 0.8302 |
| benchmark/accuracy | 0.7667 |
| benchmark/tp | 2 |
| benchmark/fp | 9 |
| benchmark/fn | 5 |
| benchmark/tn | 44 |
| benchmark/conformal_coverage | 0.8667 |
| benchmark/conformal_abstention_rate | 0.1333 |
| benchmark/conformal_selective_accuracy | 0.8269 |
| benchmark/conformal_selective_f1 | 0.1818 |
| benchmark/conformal_lower_threshold | 0.1217 |
| benchmark/conformal_upper_threshold | 0.9500 |
| benchmark/clean_auc | 0.8167 |
| benchmark/random_shift_auc | 0.8167 |
| benchmark/fgsm_auc | 0.8032 |
| benchmark/random_degradation | 0.0000 |
| benchmark/fgsm_degradation | 0.0135 |
| benchmark/eager_p50_ms | 6.9719 |
| benchmark/eager_p95_ms | 10.5880 |
| benchmark/eager_p99_ms | 25.1241 |
| benchmark/torchscript_p50_ms | 4.2061 |
| benchmark/torchscript_p95_ms | 7.1361 |
| benchmark/torchscript_p99_ms | 14.4487 |
| benchmark/torchscript_throughput_rps | 202.1621 |
| benchmark/torchscript_parity_max_diff | 0.0000 |

## Architectural Ablation Study

| Architectural Variant | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: |
| 1. Static Tabular + DCNv2 Only | 0.5121 | 0.1530 | 0.2171 | 0.3262 |
| 2. MMTT + PLR + DCNv2 | 0.8194 | 0.3827 | 0.1482 | 0.1563 |
| 3. MMTT + RelBench Bipartite GNN + DCNv2 | 0.8140 | 0.3779 | 0.1723 | 0.1869 |
| 4. Full LDM (MMTT + GNN + TabR + DCNv2) | 0.8167 | 0.3100 | 0.1860 | 0.1951 |
| 5. In-Context PFN Head (Cold-Start ICL) | 0.8005 | 0.2623 | 0.1810 | 0.1955 |

---

## Files Generated

- `ldm_pretrained.pt` — Self-supervised pre-trained encoder checkpoint
- `ldm_finetuned.pt` — Fine-tuned classification model checkpoint
- `metrics_report.md` — Comprehensive evaluation & governance report
- `config.json` — Reproducible training configuration snapshot

## Methodology

### Pre-training (Phase 1)
Collapse-free self-supervised modeling combining Masked Joint-Distribution
Reconstruction (LimiX / PRAGMA) across categorical keys, continuous amounts,
and temporal deltas with Contrastive Event Sequence Learning (CoLES).

### Fine-tuning & Risk Governance (Phase 2)
Supervised binary fraud classification combining pooled MMTT Transformer
embeddings with standardized static bureau features via a parallel Deep &
Cross Network v2 (DCNv2) head, evaluated with ROC-AUC, PR-AUC, Recall@FPR,
Expected Calibration Error (ECE), Brier Score, Conformal Risk Abstention,
and FGSM adversarial stress testing.

---

## References
1. Hollmann et al. (2025) — TabPFN v2 (*Nature* 637:319-326, DOI: 10.1038/s41586-024-08328-6)
2. Dou et al. (2025) — TransactionGPT (Visa Research, arXiv:2511.08939)
3. Ostroukhov et al. (2026) — PRAGMA (Revolut/NVIDIA, arXiv:2604.08649)
4. LimiX Team (2025/2026) — LimiX Masked Joint-Distribution (arXiv:2509.03505)
5. Gorishniy et al. (2022) — Periodic Numerical Embeddings (*NeurIPS 2022*, arXiv:2203.05556)
6. Babaev et al. (2022) — CoLES Contrastive Event Sequences (*SIGMOD 2022*, arXiv:2002.08232)
7. Wang et al. (2021) — Deep & Cross Network V2 (*WWW 2021*, arXiv:2008.13535)

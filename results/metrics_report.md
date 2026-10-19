# LDM Training & Governance Evaluation Report

**Generated**: 2026-10-19 14:31:44
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
| Pre-training (LimiX + PRAGMA + CoLES) | 3 | 0.001 | 13.8s |
| Fine-tuning (DCNv2 Classification) | 8 | 0.002 | 13.0s |

## Test & Governance Evaluation Results

| Metric | Value |
|--------|-------|
| test/loss | 1.4661 |
| test/auc | 0.8491 |
| test/pr_auc | 0.4314 |
| test/f1 | 0.3333 |
| test/recall_at_5pct_fpr | 0.1429 |
| test/brier_score | 0.1531 |
| test/ece | 0.1852 |
| benchmark/auc | 0.8491 |
| benchmark/pr_auc | 0.4314 |
| benchmark/f1_score | 0.3333 |
| benchmark/recall_at_5pct_fpr | 0.1429 |
| benchmark/brier_score | 0.1531 |
| benchmark/ece | 0.1852 |
| benchmark/precision | 0.2727 |
| benchmark/recall | 0.4286 |
| benchmark/specificity | 0.8491 |
| benchmark/accuracy | 0.8000 |
| benchmark/tp | 3 |
| benchmark/fp | 8 |
| benchmark/fn | 4 |
| benchmark/tn | 45 |
| benchmark/conformal_coverage | 0.8167 |
| benchmark/conformal_abstention_rate | 0.1833 |
| benchmark/conformal_selective_accuracy | 0.8776 |
| benchmark/conformal_selective_f1 | 0.2500 |
| benchmark/conformal_lower_threshold | 0.1348 |
| benchmark/conformal_upper_threshold | 0.8460 |
| benchmark/clean_auc | 0.8491 |
| benchmark/random_shift_auc | 0.8464 |
| benchmark/fgsm_auc | 0.8275 |
| benchmark/random_degradation | 0.0027 |
| benchmark/fgsm_degradation | 0.0216 |

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

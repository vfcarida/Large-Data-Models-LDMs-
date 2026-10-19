# LDM Training & Governance Evaluation Report

**Generated**: 2026-10-19 16:30:14
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
| Pre-training (LimiX + PRAGMA + CoLES) | 3 | 0.001 | 124.8s |
| Fine-tuning (DCNv2 Classification) | 8 | 0.002 | 626.3s |

## Test & Governance Evaluation Results

| Metric | Value |
|--------|-------|
| test/loss | 2.4407 |
| test/auc | 0.8140 |
| test/pr_auc | 0.3219 |
| test/f1 | 0.1667 |
| test/recall_at_5pct_fpr | 0.1429 |
| test/brier_score | 0.1366 |
| test/ece | 0.1581 |
| benchmark/auc | 0.8140 |
| benchmark/pr_auc | 0.3219 |
| benchmark/f1_score | 0.1667 |
| benchmark/recall_at_5pct_fpr | 0.1429 |
| benchmark/brier_score | 0.1366 |
| benchmark/ece | 0.1581 |
| benchmark/precision | 0.2000 |
| benchmark/recall | 0.1429 |
| benchmark/specificity | 0.9245 |
| benchmark/accuracy | 0.8333 |
| benchmark/tp | 1 |
| benchmark/fp | 4 |
| benchmark/fn | 6 |
| benchmark/tn | 49 |
| benchmark/conformal_coverage | 0.8667 |
| benchmark/conformal_abstention_rate | 0.1333 |
| benchmark/conformal_selective_accuracy | 0.9038 |
| benchmark/conformal_selective_f1 | 0.2857 |
| benchmark/conformal_lower_threshold | 0.0613 |
| benchmark/conformal_upper_threshold | 0.9290 |
| benchmark/clean_auc | 0.8140 |
| benchmark/random_shift_auc | 0.8140 |
| benchmark/fgsm_auc | 0.7898 |
| benchmark/random_degradation | 0.0000 |
| benchmark/fgsm_degradation | 0.0243 |
| benchmark/eager_p50_ms | 11.3461 |
| benchmark/eager_p95_ms | 19.6803 |
| benchmark/eager_p99_ms | 20.3743 |
| benchmark/torchscript_p50_ms | 12.2560 |
| benchmark/torchscript_p95_ms | 23.9463 |
| benchmark/torchscript_p99_ms | 31.8403 |
| benchmark/torchscript_throughput_rps | 75.9820 |
| benchmark/torchscript_parity_max_diff | 0.0000 |
| benchmark/fp32_size_mb | 3.7471 |
| benchmark/int8_size_mb | 3.1347 |
| benchmark/int8_compression_ratio | 1.1954 |
| benchmark/int8_max_prob_diff | 0.0000 |
| benchmark/cascade_fast_exit_rate | 1.0000 |
| benchmark/cascade_speedup_factor | 13.0295 |

## Architectural Ablation Study

| Architectural Variant | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: |
| 1. Static Tabular + DCNv2 Only | 0.4582 | 0.1205 | 0.2374 | 0.3474 |
| 2. MMTT + PLR + DCNv2 | 0.8383 | 0.4150 | 0.1109 | 0.1013 |
| 3. MMTT + RelBench Bipartite GNN + DCNv2 | 0.8383 | 0.4201 | 0.1248 | 0.1324 |
| 4. Full LDM (MMTT + GNN + TabR + DCNv2) | 0.8140 | 0.3219 | 0.1366 | 0.1581 |
| 5. In-Context PFN Head (Cold-Start ICL) | 0.7763 | 0.3080 | 0.1616 | 0.1719 |
| 6. Multi-Task MMoE Head (Fraud + Credit + LTV) | 0.8032 | 0.3215 | 0.1392 | 0.1447 |

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

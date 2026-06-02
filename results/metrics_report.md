# LDM Training Results Report

**Generated**: 2026-06-01 23:16:12
**Config**: small

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | 64 |
| Transformer Layers | 2 |
| Attention Heads | 2 |
| DCNv2 Cross Layers | 2 |
| Vocabulary Size | 20 |
| Dropout | 0.1 |

## Dataset

| Parameter | Value |
|-----------|-------|
| Number of Users | 200 |
| Transactions/User | 10-50 |
| Fraud Rate | 5.0% |
| Max Sequence Length | 64 |

## Training Summary

| Phase | Epochs | Learning Rate | Duration |
|-------|--------|--------------|----------|
| Pre-training (LimiX) | 3 | 0.001 | 3.8s |
| Fine-tuning (Classification) | 5 | 0.001 | 6.0s |

## Test Results (Final Evaluation)

| Metric | Value |
|--------|-------|
| test/loss | 305338417152.0000 |
| test/auc | 0.5000 |
| test/f1 | 0.0000 |

---

## Files Generated

- `ldm_pretrained.pt` — Pre-trained encoder checkpoint
- `ldm_finetuned.pt` — Fine-tuned model (ready for inference)
- `metrics_report.md` — This report
- `config.json` — Training configuration

## Methodology

### Pre-training (Phase 1)
Self-supervised masked modeling using the LimiX objective. The model learns
to reconstruct randomly masked transaction features from visible context.
No labels required.

### Fine-tuning (Phase 2)
Supervised binary classification for fraud detection. Uses the pre-trained
Transformer encoder combined with a DCNv2 fusion head that integrates
static tabular features (bureau score, income, age).

### Key Technical Features
- **KVT Tokenization**: Key-Value-Time encoding preserving financial data semantics
- **Dynamic Sequence Packing**: Zero-padding-free batching for GPU efficiency
- **Virtual Token Conditioning**: Continuous prompts for global context
- **DCNv2 Fusion**: Explicit polynomial feature interactions
- **Class-Weighted Loss**: Handles extreme fraud rate imbalance

---

## References
1. TransactionGPT (Visa Research, 2025)
2. PRAGMA (Revolut/NVIDIA, 2025)
3. LimiX — Masked Joint-Distribution Pre-training
4. TabPFN v2/2.5 — Prior-Data Fitted Networks
5. Schema-1 (DLM) — Data Language Model
6. DCNv2 — Deep & Cross Network V2

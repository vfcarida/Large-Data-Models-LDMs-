"""
LDM Research Lab — Main Entrypoint.

============================================================================
USAGE (USO)
============================================================================
    # Run full pipeline (generate data → pretrain → finetune → evaluate)
    python src/main.py

    # Run with small config (fast, for testing)
    python src/main.py --config small

    # Run only evaluation on a trained model
    python src/main.py --eval-only --checkpoint results/ldm_finetuned.pt

============================================================================
PIPELINE OVERVIEW
============================================================================
    1. GENERATE: Create synthetic financial transaction data
    2. PRETRAIN: Self-supervised masked modeling (LimiX objective)
    3. FINETUNE: Supervised fraud classification with DCNv2 fusion
    4. EVALUATE: Compute final metrics (AUC, F1, Precision, Recall)
    5. REPORT:   Save results, metrics, and training curves

============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        RichProgressBar,
        LearningRateMonitor,
    )
except ImportError:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        RichProgressBar,
        LearningRateMonitor,
    )

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_module import LDMDataModule
from src.models.ldm_model import LargeDataModel
from src.training.lightning_module import LDMPreTrainingModule, LDMFineTuningModule
from src.evaluation.metrics import calculate_auc, calculate_f1_flexible

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LDM")


# =============================================================================
# Configuration Presets
# =============================================================================
# Two configs: "default" for real training, "small" for quick testing
# =============================================================================

CONFIGS = {
    "default": {
        # Data
        "data_path": "data/synthetic_transactions.parquet",
        "num_users": 1000,
        "min_transactions": 20,
        "max_transactions": 200,
        "fraud_rate": 0.02,
        "batch_size": 32,
        "max_seq_len": 256,
        "num_workers": 0,
        # Model
        "vocab_size": 20,
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
        "tabular_feature_size": 3,
        "dcn_layers": 3,
        # Pre-training
        "pretrain_epochs": 10,
        "pretrain_lr": 1e-3,
        "ssl_masking_prob": 0.15,
        # Fine-tuning
        "finetune_epochs": 20,
        "finetune_lr": 1e-3,
        "pos_weight": 10.0,
        "metric_threshold": 0.5,
        # General
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "seed": 42,
    },
    "small": {
        # Minimal config for quick testing on CPU
        "data_path": "data/synthetic_small.parquet",
        "num_users": 200,
        "min_transactions": 10,
        "max_transactions": 50,
        "fraud_rate": 0.05,
        "batch_size": 16,
        "max_seq_len": 64,
        "num_workers": 0,
        # Tiny model
        "vocab_size": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "tabular_feature_size": 3,
        "dcn_layers": 2,
        # Fewer epochs
        "pretrain_epochs": 3,
        "pretrain_lr": 1e-3,
        "ssl_masking_prob": 0.15,
        "finetune_epochs": 5,
        "finetune_lr": 1e-3,
        "pos_weight": 5.0,
        "metric_threshold": 0.5,
        "weight_decay": 0.01,
        "warmup_steps": 20,
        "seed": 42,
    },
}


def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    logger.info(f"Random seed set to {seed}")


def create_results_dir() -> Path:
    """Create timestamped results directory."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_pretraining(config: dict, data_module: LDMDataModule, results_dir: Path) -> LargeDataModel:
    """
    Phase 1: Self-Supervised Pre-Training.

    Trains the Transformer encoder using masked prediction (LimiX).
    No labels required — the model learns from the structure of
    transaction sequences.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: SELF-SUPERVISED PRE-TRAINING (LimiX)")
    logger.info("=" * 70)

    # Create model
    model = LargeDataModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        tabular_feature_size=config["tabular_feature_size"],
        dcn_layers=config["dcn_layers"],
    )

    # Estimate max steps
    if data_module.train_dataset is not None:
        steps_per_epoch = max(1, len(data_module.train_dataset) // config["batch_size"])
    else:
        steps_per_epoch = 100
    max_steps = steps_per_epoch * config["pretrain_epochs"]

    # Create Lightning module
    pretrain_config = {
        "learning_rate": config["pretrain_lr"],
        "weight_decay": config["weight_decay"],
        "warmup_steps": config["warmup_steps"],
        "max_steps": max_steps,
        "ssl_masking_prob": config["ssl_masking_prob"],
    }
    lightning_module = LDMPreTrainingModule(model, pretrain_config)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(results_dir / "checkpoints"),
            filename="pretrain-{epoch:02d}-{pretrain/val_loss:.4f}",
            monitor="pretrain/val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Try to add RichProgressBar (may not be available)
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["pretrain_epochs"],
        callbacks=callbacks,
        default_root_dir=str(results_dir),
        accelerator="auto",
        devices=1,
        precision="32",  # Use 32-bit for stability in pre-training
        enable_progress_bar=True,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    # Train
    logger.info(f"Starting pre-training for {config['pretrain_epochs']} epochs...")
    start_time = time.time()
    trainer.fit(lightning_module, data_module)
    pretrain_time = time.time() - start_time
    logger.info(f"Pre-training completed in {pretrain_time:.1f}s")

    # Save pre-trained model
    pretrain_path = results_dir / "ldm_pretrained.pt"
    model.save_checkpoint(str(pretrain_path), extra_info={"pretrain_time": pretrain_time})

    return model


def run_finetuning(
    config: dict,
    model: LargeDataModel,
    data_module: LDMDataModule,
    results_dir: Path,
) -> dict:
    """
    Phase 2: Supervised Fine-Tuning.

    Takes the pre-trained model and trains it for binary fraud classification
    using labeled data and the DCNv2 fusion head.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: SUPERVISED FINE-TUNING (Fraud Classification)")
    logger.info("=" * 70)

    # Estimate max steps
    if data_module.train_dataset is not None:
        steps_per_epoch = max(1, len(data_module.train_dataset) // config["batch_size"])
    else:
        steps_per_epoch = 100
    max_steps = steps_per_epoch * config["finetune_epochs"]

    # Create Lightning module
    finetune_config = {
        "learning_rate": config["finetune_lr"],
        "weight_decay": config["weight_decay"],
        "warmup_steps": config["warmup_steps"],
        "max_steps": max_steps,
        "pos_weight": config["pos_weight"],
        "metric_threshold": config["metric_threshold"],
    }
    lightning_module = LDMFineTuningModule(model, finetune_config)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(results_dir / "checkpoints"),
            filename="finetune-{epoch:02d}-{finetune/val_auc:.4f}",
            monitor="finetune/val_auc",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="finetune/val_auc",
            mode="max",
            patience=5,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["finetune_epochs"],
        callbacks=callbacks,
        default_root_dir=str(results_dir),
        accelerator="auto",
        devices=1,
        precision="32",
        enable_progress_bar=True,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    # Train
    logger.info(f"Starting fine-tuning for {config['finetune_epochs']} epochs...")
    start_time = time.time()
    trainer.fit(lightning_module, data_module)
    finetune_time = time.time() - start_time
    logger.info(f"Fine-tuning completed in {finetune_time:.1f}s")

    # Test evaluation
    logger.info("Running test evaluation...")
    test_results = trainer.test(lightning_module, data_module)

    # Save final model
    finetune_path = results_dir / "ldm_finetuned.pt"
    model.save_checkpoint(
        str(finetune_path),
        extra_info={"finetune_time": finetune_time, "test_results": test_results},
    )

    return test_results[0] if test_results else {}


def generate_report(
    config: dict,
    test_results: dict,
    results_dir: Path,
    pretrain_time: float,
    finetune_time: float,
):
    """
    Generates a comprehensive metrics report.

    Creates a markdown report with all training results,
    suitable for presentation to stakeholders.
    """
    logger.info("=" * 70)
    logger.info("GENERATING RESULTS REPORT")
    logger.info("=" * 70)

    report = f"""# LDM Training Results Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Config**: {config.get('_config_name', 'default')}

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | {config['hidden_size']} |
| Transformer Layers | {config['num_layers']} |
| Attention Heads | {config['num_heads']} |
| DCNv2 Cross Layers | {config['dcn_layers']} |
| Vocabulary Size | {config['vocab_size']} |
| Dropout | {config['dropout']} |

## Dataset

| Parameter | Value |
|-----------|-------|
| Number of Users | {config['num_users']} |
| Transactions/User | {config['min_transactions']}-{config['max_transactions']} |
| Fraud Rate | {config['fraud_rate']:.1%} |
| Max Sequence Length | {config['max_seq_len']} |

## Training Summary

| Phase | Epochs | Learning Rate | Duration |
|-------|--------|--------------|----------|
| Pre-training (LimiX) | {config['pretrain_epochs']} | {config['pretrain_lr']} | {pretrain_time:.1f}s |
| Fine-tuning (Classification) | {config['finetune_epochs']} | {config['finetune_lr']} | {finetune_time:.1f}s |

## Test Results (Final Evaluation)

| Metric | Value |
|--------|-------|
"""
    for key, value in test_results.items():
        if isinstance(value, float):
            report += f"| {key} | {value:.4f} |\n"
        else:
            report += f"| {key} | {value} |\n"

    report += f"""
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
"""

    # Save report
    report_path = results_dir / "metrics_report.md"
    report_path.write_text(report, encoding="utf-8")

    # Save config
    config_path = results_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Config saved to {config_path}")


def main():
    """Main entrypoint for the LDM Research Lab pipeline."""

    # =========================================================================
    # Argument Parsing
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="LDM Research Lab — Train and evaluate Large Data Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=list(CONFIGS.keys()),
        help="Configuration preset to use (default or small)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pre-training phase (useful for debugging fine-tuning)",
    )
    args = parser.parse_args()

    # =========================================================================
    # Setup
    # =========================================================================
    config = CONFIGS[args.config].copy()
    config["_config_name"] = args.config
    set_seed(config["seed"])
    results_dir = create_results_dir()

    logger.info("=" * 70)
    logger.info("    LDM RESEARCH LAB — Large Data Models")
    logger.info(f"    Config: {args.config}")
    logger.info(f"    Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"    Results: {results_dir}")
    logger.info("=" * 70)

    # =========================================================================
    # Data Pipeline
    # =========================================================================
    data_module = LDMDataModule(
        data_path=config["data_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_seq_len=config["max_seq_len"],
        num_users=config["num_users"],
        min_transactions=config["min_transactions"],
        max_transactions=config["max_transactions"],
        fraud_rate=config["fraud_rate"],
        seed=config["seed"],
    )
    data_module.prepare_data()
    data_module.setup()

    # Update vocab size from actual data
    config["vocab_size"] = int(data_module.vocab_size)
    config["tabular_feature_size"] = int(data_module.tabular_feature_size)

    # =========================================================================
    # Phase 1: Pre-Training
    # =========================================================================
    pretrain_time = 0.0
    if not args.eval_only and not args.skip_pretrain:
        start = time.time()
        model = run_pretraining(config, data_module, results_dir)
        pretrain_time = time.time() - start
    elif args.checkpoint:
        model = LargeDataModel.load_checkpoint(args.checkpoint)
    else:
        # Skip pretrain, create fresh model
        model = LargeDataModel(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            tabular_feature_size=config["tabular_feature_size"],
            dcn_layers=config["dcn_layers"],
        )

    # =========================================================================
    # Phase 2: Fine-Tuning
    # =========================================================================
    finetune_time = 0.0
    test_results = {}
    if not args.eval_only:
        start = time.time()
        test_results = run_finetuning(config, model, data_module, results_dir)
        finetune_time = time.time() - start

    # =========================================================================
    # Report Generation
    # =========================================================================
    generate_report(config, test_results, results_dir, pretrain_time, finetune_time)

    # =========================================================================
    # Final Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("    PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"    Pre-training time: {pretrain_time:.1f}s")
    logger.info(f"    Fine-tuning time:  {finetune_time:.1f}s")
    logger.info(f"    Total time:        {pretrain_time + finetune_time:.1f}s")
    logger.info("")
    for key, value in test_results.items():
        if isinstance(value, float):
            logger.info(f"    {key}: {value:.4f}")
    logger.info("")
    logger.info(f"    Results saved to: {results_dir}")
    logger.info(f"    Model checkpoint: {results_dir / 'ldm_finetuned.pt'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

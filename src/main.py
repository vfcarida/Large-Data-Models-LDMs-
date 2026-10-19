"""
LDM Research Lab — Main Pipeline Entrypoint.

============================================================================
USAGE
============================================================================
    # Run full pipeline (generate data -> pretrain -> finetune -> evaluate)
    python src/main.py

    # Run with small config (fast CPU verification)
    python src/main.py --config small

    # Run only evaluation on a trained model checkpoint
    python src/main.py --eval-only --checkpoint results/ldm_finetuned.pt

============================================================================
PIPELINE OVERVIEW
============================================================================
    1. GENERATE: Create synthetic financial transaction trajectories (KVT + Tabular)
    2. PRETRAIN: Collapse-free self-supervised modeling (LimiX + PRAGMA + CoLES)
    3. FINETUNE: Supervised fraud classification with parallel DCNv2 fusion
    4. EVALUATE: Compute discrimination (ROC-AUC, PR-AUC, Recall@FPR, F1),
                 calibration (ECE, Brier), Conformal Abstention, and FGSM shift
    5. REPORT:   Save checkpoints, JSON configuration, and Markdown audit report
============================================================================
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
import yaml

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        RichProgressBar,
    )
except ImportError:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        RichProgressBar,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_module import LDMDataModule
from src.evaluation.benchmarks import BenchmarkSuite
from src.models.ldm_model import LargeDataModel
from src.training.lightning_module import LDMFineTuningModule, LDMPreTrainingModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LDM")

DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {
        "_config_name": "default",
        "data_path": "data/synthetic_transactions.parquet",
        "num_users": 2500,
        "min_transactions": 30,
        "max_transactions": 300,
        "fraud_rate": 0.02,
        "batch_size": 64,
        "max_seq_len": 256,
        "num_workers": 0,
        "vocab_size": 20,
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
        "tabular_feature_size": 3,
        "dcn_layers": 3,
        "use_virtual_tokens": True,
        "num_virtual_tokens": 2,
        "pooling": "mean",
        "pretrain_epochs": 10,
        "pretrain_lr": 1e-3,
        "ssl_masking_prob": 0.20,
        "ssl_temperature": 0.15,
        "finetune_epochs": 15,
        "finetune_lr": 1e-3,
        "pos_weight": 8.0,
        "metric_threshold": 0.5,
        "conformal_alpha": 0.10,
        "enable_strategic_shift": True,
        "noise_std": 0.10,
        "adversarial_epsilon": 0.05,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "seed": 42,
    },
    "small": {
        "_config_name": "small",
        "data_path": "data/synthetic_small.parquet",
        "num_users": 200,
        "min_transactions": 12,
        "max_transactions": 50,
        "fraud_rate": 0.10,
        "batch_size": 16,
        "max_seq_len": 64,
        "num_workers": 0,
        "vocab_size": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "tabular_feature_size": 3,
        "dcn_layers": 2,
        "use_virtual_tokens": True,
        "num_virtual_tokens": 2,
        "pooling": "mean",
        "pretrain_epochs": 3,
        "pretrain_lr": 1e-3,
        "ssl_masking_prob": 0.15,
        "ssl_temperature": 0.15,
        "finetune_epochs": 8,
        "finetune_lr": 2e-3,
        "pos_weight": 4.0,
        "metric_threshold": 0.5,
        "conformal_alpha": 0.10,
        "enable_strategic_shift": True,
        "noise_std": 0.10,
        "adversarial_epsilon": 0.05,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "seed": 42,
    },
}


def load_config(preset_name: str) -> Dict[str, Any]:
    """Loads configuration from `configs/` YAML files with fallback to `DEFAULT_CONFIGS`."""
    config = dict(DEFAULT_CONFIGS.get(preset_name, DEFAULT_CONFIGS["default"]))
    yaml_filename = "config.yaml" if preset_name == "default" else f"config_{preset_name}.yaml"
    yaml_path = PROJECT_ROOT / "configs" / yaml_filename

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config.update(loaded)
    config["_config_name"] = preset_name
    return config


def set_seed(seed: int) -> None:
    """Sets random seeds for deterministic reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    logger.info(f"Random seed set to {seed}")


def create_results_dir() -> Path:
    """Ensures the `results/` directory exists."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_pretraining(
    config: Dict[str, Any],
    data_module: LDMDataModule,
    results_dir: Path,
) -> LargeDataModel:
    """
    Phase 1: Collapse-Free Self-Supervised Pre-Training (LimiX + PRAGMA + CoLES).
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: SELF-SUPERVISED PRE-TRAINING (LimiX + PRAGMA + CoLES)")
    logger.info("=" * 70)

    model = LargeDataModel(
        vocab_size=int(config["vocab_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dropout=float(config["dropout"]),
        tabular_feature_size=int(config["tabular_feature_size"]),
        dcn_layers=int(config["dcn_layers"]),
        use_virtual_tokens=bool(config.get("use_virtual_tokens", True)),
        num_virtual_tokens=int(config.get("num_virtual_tokens", 2)),
        pooling=config.get("pooling", "mean"),
    )

    if data_module.train_dataset is not None:
        steps_per_epoch = max(1, len(data_module.train_dataset) // int(config["batch_size"]))
    else:
        steps_per_epoch = 100
    max_steps = steps_per_epoch * int(config["pretrain_epochs"])

    pretrain_config = {
        "learning_rate": float(config["pretrain_lr"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_steps": int(config["warmup_steps"]),
        "max_steps": max_steps,
        "ssl_masking_prob": float(config["ssl_masking_prob"]),
        "ssl_temperature": float(config.get("ssl_temperature", 0.15)),
    }
    lightning_module = LDMPreTrainingModule(model, pretrain_config)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(results_dir / "checkpoints"),
            filename="pretrain-{epoch:02d}",
            monitor="pretrain/val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=int(config["pretrain_epochs"]),
        callbacks=callbacks,
        default_root_dir=str(results_dir),
        accelerator="auto",
        devices=1,
        precision="32",
        enable_progress_bar=False,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    logger.info(f"Starting pre-training for {config['pretrain_epochs']} epochs...")
    start_time = time.time()
    trainer.fit(lightning_module, data_module)
    pretrain_time = time.time() - start_time
    logger.info(f"Pre-training completed in {pretrain_time:.1f}s")

    pretrain_path = results_dir / "ldm_pretrained.pt"
    model.save_checkpoint(str(pretrain_path), extra_info={"pretrain_time": pretrain_time})

    return model


def run_finetuning(
    config: Dict[str, Any],
    model: LargeDataModel,
    data_module: LDMDataModule,
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Phase 2: Supervised Fine-Tuning & Governance Benchmark Evaluation.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: SUPERVISED FINE-TUNING (Fraud Classification)")
    logger.info("=" * 70)

    if data_module.train_dataset is not None:
        steps_per_epoch = max(1, len(data_module.train_dataset) // int(config["batch_size"]))
    else:
        steps_per_epoch = 100
    max_steps = steps_per_epoch * int(config["finetune_epochs"])

    finetune_config = {
        "learning_rate": float(config["finetune_lr"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_steps": int(config["warmup_steps"]),
        "max_steps": max_steps,
        "pos_weight": float(config["pos_weight"]),
        "metric_threshold": float(config["metric_threshold"]),
    }
    lightning_module = LDMFineTuningModule(model, finetune_config)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(results_dir / "checkpoints"),
            filename="finetune-{epoch:02d}",
            monitor="finetune/val_auc",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="finetune/val_auc",
            mode="max",
            patience=5,
            verbose=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=int(config["finetune_epochs"]),
        callbacks=callbacks,
        default_root_dir=str(results_dir),
        accelerator="auto",
        devices=1,
        precision="32",
        enable_progress_bar=False,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    logger.info(f"Starting fine-tuning for {config['finetune_epochs']} epochs...")
    start_time = time.time()
    trainer.fit(lightning_module, data_module)
    finetune_time = time.time() - start_time
    logger.info(f"Fine-tuning completed in {finetune_time:.1f}s")

    logger.info("Running test evaluation and governance benchmark suite...")
    test_results_list = trainer.test(lightning_module, data_module, verbose=False)
    combined_results: Dict[str, Any] = dict(test_results_list[0]) if test_results_list else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = BenchmarkSuite(model, config, device=device)
    suite_results = benchmark.run_all(
        data_module.test_dataloader(),
        val_dataloader=data_module.val_dataloader(),
    )
    for k, v in suite_results.items():
        combined_results[f"benchmark/{k}"] = v

    from src.evaluation.ablation_suite import AblationBenchmarkSuite

    ablation_runner = AblationBenchmarkSuite(model, device=device)
    ablation_metrics = ablation_runner.run_ablation_study(
        data_module.test_dataloader(),
        support_dataloader=data_module.train_dataloader(),
    )
    combined_results["_ablation_markdown"] = AblationBenchmarkSuite.format_markdown_table(
        ablation_metrics
    )

    finetune_path = results_dir / "ldm_finetuned.pt"
    model.save_checkpoint(
        str(finetune_path),
        extra_info={"finetune_time": finetune_time},
    )

    return combined_results


def generate_report(
    config: Dict[str, Any],
    test_results: Dict[str, Any],
    results_dir: Path,
    pretrain_time: float,
    finetune_time: float,
) -> None:
    """
    Generates an audit-ready Markdown metrics report and JSON configuration snapshot.
    """
    logger.info("=" * 70)
    logger.info("GENERATING RESULTS REPORT")
    logger.info("=" * 70)

    ablation_md = test_results.pop("_ablation_markdown", None)

    report = f"""# LDM Training & Governance Evaluation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Config**: {config.get('_config_name', 'default')}

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | {config['hidden_size']} |
| Transformer Layers | {config['num_layers']} |
| Attention Heads | {config['num_heads']} |
| Numerical Embedding | Periodic Numerical Embedding (PLR, 16 frequencies) |
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
| Pre-training (LimiX + PRAGMA + CoLES) | {config['pretrain_epochs']} | {config['pretrain_lr']} | {pretrain_time:.1f}s |
| Fine-tuning (DCNv2 Classification) | {config['finetune_epochs']} | {config['finetune_lr']} | {finetune_time:.1f}s |

## Test & Governance Evaluation Results

| Metric | Value |
|--------|-------|
"""
    for key, value in test_results.items():
        if isinstance(value, float):
            report += f"| {key} | {value:.4f} |\n"
        else:
            report += f"| {key} | {value} |\n"

    if ablation_md:
        report += f"\n## Architectural Ablation Study\n\n{ablation_md}\n"

    report += """
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
"""

    report_path = results_dir / "metrics_report.md"
    report_path.write_text(report, encoding="utf-8")

    config_path = results_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Config saved to {config_path}")


def main() -> None:
    """Main entrypoint for the LDM Research Lab pipeline."""
    parser = argparse.ArgumentParser(
        description="LDM Research Lab — Train and evaluate Large Data Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=list(DEFAULT_CONFIGS.keys()),
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
        help="Skip pre-training phase (useful for ablation comparisons)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    results_dir = create_results_dir()

    logger.info("=" * 70)
    logger.info("    LDM RESEARCH LAB — Large Data Models")
    logger.info(f"    Config: {args.config}")
    logger.info(f"    Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"    Results: {results_dir}")
    logger.info("=" * 70)

    data_module = LDMDataModule(
        data_path=str(PROJECT_ROOT / config["data_path"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        max_seq_len=int(config["max_seq_len"]),
        num_users=int(config["num_users"]),
        min_transactions=int(config["min_transactions"]),
        max_transactions=int(config["max_transactions"]),
        fraud_rate=float(config["fraud_rate"]),
        seed=int(config["seed"]),
    )
    data_module.prepare_data()
    data_module.setup()

    config["vocab_size"] = int(data_module.vocab_size)
    config["tabular_feature_size"] = int(data_module.tabular_feature_size)

    pretrain_time = 0.0
    if not args.eval_only and not args.skip_pretrain:
        start = time.time()
        model = run_pretraining(config, data_module, results_dir)
        pretrain_time = time.time() - start
    elif args.checkpoint:
        model = LargeDataModel.load_checkpoint(args.checkpoint)
    else:
        model = LargeDataModel(
            vocab_size=int(config["vocab_size"]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
            tabular_feature_size=int(config["tabular_feature_size"]),
            dcn_layers=int(config["dcn_layers"]),
            use_virtual_tokens=bool(config.get("use_virtual_tokens", True)),
            num_virtual_tokens=int(config.get("num_virtual_tokens", 2)),
            pooling=config.get("pooling", "mean"),
        )

    finetune_time = 0.0
    test_results: Dict[str, Any] = {}
    if not args.eval_only:
        start = time.time()
        test_results = run_finetuning(config, model, data_module, results_dir)
        finetune_time = time.time() - start
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        benchmark = BenchmarkSuite(model, config, device=device)
        test_results = benchmark.run_all(
            data_module.test_dataloader(),
            val_dataloader=data_module.val_dataloader(),
        )

    generate_report(config, test_results, results_dir, pretrain_time, finetune_time)

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

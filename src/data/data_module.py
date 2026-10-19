"""
PyTorch Lightning DataModule for Large Data Models (LDMs).

============================================================================
PURPOSE
============================================================================
Manages data generation, Parquet caching, stratified user-level train/val/test
partitioning, zero-leakage z-score feature normalization, and PyTorch
DataLoader instantiation.

Design invariants:
  1. Entity-Level Partitioning: All transactions belonging to a given
     `user_id` are assigned to the same split (`train`, `val`, or `test`),
     preventing temporal or identity leakage across splits.
  2. Stratified Label Preservation: User splits are stratified by `is_fraud`
     so validation and test partitions always contain both positive and
     negative classes even under severe class imbalance (1%-5%).
  3. Zero-Leakage Scaling: `NormalizationStats` is fitted strictly on the
     training partition and applied to validation and test partitions.
============================================================================
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl

from src.data.synthetic_generator import SyntheticTransactionGenerator
from src.data.mmtt_dataset import (
    FinancialEventDataset,
    NormalizationStats,
    TABULAR_COLUMNS,
    dynamic_sequence_packing_collate_fn,
)

logger = logging.getLogger(__name__)


class LDMDataModule(pl.LightningDataModule):
    """
    LightningDataModule orchestrating KVT sequences and static tabular features.

    Args:
        data_path: Path to Parquet/CSV transaction file (generated if missing).
        batch_size: Mini-batch size for DataLoaders.
        num_workers: Number of subprocess workers for DataLoader.
        max_seq_len: Maximum sequence length per user.
        num_users: Number of synthetic users to generate if `data_path` is absent.
        min_transactions: Minimum transactions per synthetic user.
        max_transactions: Maximum transactions per synthetic user.
        fraud_rate: Target user fraud rate.
        seed: Random seed for deterministic generation and splitting.
        force_regenerate: If True, regenerates synthetic data even if `data_path` exists.
    """

    def __init__(
        self,
        data_path: str = "data/synthetic_transactions.parquet",
        batch_size: int = 32,
        num_workers: int = 0,
        max_seq_len: int = 256,
        num_users: int = 1000,
        min_transactions: int = 20,
        max_transactions: int = 200,
        fraud_rate: float = 0.02,
        seed: int = 42,
        force_regenerate: bool = False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.num_users = num_users
        self.min_transactions = min_transactions
        self.max_transactions = max_transactions
        self.fraud_rate = fraud_rate
        self.seed = seed
        self.force_regenerate = force_regenerate

        self.vocab_size: int = 20
        self.tabular_feature_size: int = len(TABULAR_COLUMNS)
        self.stats: Optional[NormalizationStats] = None

        self.train_dataset: Optional[FinancialEventDataset] = None
        self.val_dataset: Optional[FinancialEventDataset] = None
        self.test_dataset: Optional[FinancialEventDataset] = None

    def prepare_data(self) -> None:
        """Generates synthetic transaction data if not already present on disk."""
        need_gen = self.force_regenerate or not self.data_path.exists()
        if not need_gen and self.data_path.exists():
            try:
                existing_df = (
                    pd.read_parquet(self.data_path)
                    if self.data_path.suffix.lower() == ".parquet"
                    else pd.read_csv(self.data_path)
                )
                if existing_df["user_id"].nunique() != self.num_users:
                    need_gen = True
            except Exception:
                need_gen = True

        if need_gen:
            logger.info(
                f"Generating synthetic dataset ({self.num_users} users, "
                f"fraud_rate={self.fraud_rate:.1%}) at {self.data_path}..."
            )
            generator = SyntheticTransactionGenerator(
                num_users=self.num_users,
                min_transactions=self.min_transactions,
                max_transactions=self.max_transactions,
                fraud_rate=self.fraud_rate,
                seed=self.seed,
            )
            generator.generate(save_path=str(self.data_path))

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads data, performs stratified user-level splitting, and fits normalization on train.
        """
        if self.train_dataset is not None and stage not in ("fit", None):
            return

        if not self.data_path.exists():
            self.prepare_data()

        df = (
            pd.read_parquet(self.data_path)
            if self.data_path.suffix.lower() == ".parquet"
            else pd.read_csv(self.data_path)
        )

        self.vocab_size = int(df["mcc_token"].max()) + 1
        available_tab = [c for c in TABULAR_COLUMNS if c in df.columns]
        self.tabular_feature_size = max(1, len(available_tab))

        # Extract user-level labels for stratified entity partitioning
        user_summary = df.groupby("user_id", sort=True)["is_fraud"].max().reset_index()
        users = user_summary["user_id"].to_numpy()
        labels = user_summary["is_fraud"].to_numpy()

        can_stratify = np.min(np.bincount(labels)) >= 3 if len(np.unique(labels)) > 1 else False

        train_users, temp_users, _, temp_labels = train_test_split(
            users,
            labels,
            test_size=0.30,
            random_state=self.seed,
            stratify=labels if can_stratify else None,
        )

        can_stratify_temp = (
            np.min(np.bincount(temp_labels)) >= 2 if len(np.unique(temp_labels)) > 1 else False
        )
        val_users, test_users = train_test_split(
            temp_users,
            test_size=0.50,
            random_state=self.seed,
            stratify=temp_labels if can_stratify_temp else None,
        )

        train_df = df[df["user_id"].isin(set(train_users))].copy()
        val_df = df[df["user_id"].isin(set(val_users))].copy()
        test_df = df[df["user_id"].isin(set(test_users))].copy()

        # Fit normalization strictly on training split
        self.stats = NormalizationStats.fit(train_df, available_tab)

        self.train_dataset = FinancialEventDataset(
            train_df,
            max_seq_len=self.max_seq_len,
            stats=self.stats,
            tabular_cols=available_tab,
        )
        self.val_dataset = FinancialEventDataset(
            val_df,
            max_seq_len=self.max_seq_len,
            stats=self.stats,
            tabular_cols=available_tab,
        )
        self.test_dataset = FinancialEventDataset(
            test_df,
            max_seq_len=self.max_seq_len,
            stats=self.stats,
            tabular_cols=available_tab,
        )

        logger.info(
            f"LDMDataModule setup complete | Train: {len(self.train_dataset)} users, "
            f"Val: {len(self.val_dataset)} users, Test: {len(self.test_dataset)} users | "
            f"Vocab size: {self.vocab_size}, Tabular features: {self.tabular_feature_size}"
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup() before train_dataloader()"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dynamic_sequence_packing_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup() before val_dataloader()"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dynamic_sequence_packing_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Call setup() before test_dataloader()"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dynamic_sequence_packing_collate_fn,
        )

"""
Multi-Modal-Temporal-Tabular (MMTT) Financial Event Dataset & Sequence Collator.

============================================================================
PURPOSE
============================================================================
Converts raw transactional DataFrames into Key-Value-Time (KVT) PyTorch
tensors paired with static credit bureau features and boolean attention
padding masks.

Key reliability guarantees:
  1. Zero-Leakage Feature Scaling: `NormalizationStats` computes column-wise
     z-score statistics strictly on the training split and applies those
     exact parameters to validation and test splits.
  2. Padding Isolation: `dynamic_sequence_packing_collate_fn` generates a
     strict boolean `key_padding_mask` (`True` = padded position) alongside
     cumulative sequence lengths (`cu_seqlens`), ensuring padded elements
     never contaminate multi-head attention or sequence pooling.

============================================================================
REFERENCES
============================================================================
  - Ostroukhov et al. (2026): "PRAGMA: A Self-Supervised Foundation Model
    for Banking Events" (KVT Tokenization, arXiv:2604.08649)
  - Dou et al. (2025): "TransactionGPT: Foundation Models for Payment
    Trajectories" (MMTT Representation, arXiv:2511.08939)
============================================================================
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TABULAR_COLUMNS: List[str] = ["bureau_score", "income", "age"]


@dataclass(frozen=True)
class NormalizationStats:
    """
    Immutable container for training-set z-score normalization statistics.

    Attributes:
        amount_mean: Mean of log1p-transformed transaction amounts.
        amount_std: Standard deviation of log1p-transformed transaction amounts.
        tabular_mean: Per-column means for static tabular features.
        tabular_std: Per-column standard deviations for static tabular features.
    """

    amount_mean: float
    amount_std: float
    tabular_mean: np.ndarray
    tabular_std: np.ndarray

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        tabular_cols: Sequence[str] = TABULAR_COLUMNS,
    ) -> "NormalizationStats":
        """
        Fits normalization parameters on a training DataFrame.

        Args:
            df: Training transaction DataFrame.
            tabular_cols: Names of static user-level numerical columns.

        Returns:
            Fitted `NormalizationStats` instance.
        """
        log_amounts = np.log1p(np.clip(df["amount"].to_numpy(dtype=np.float64), 0.0, None))
        amount_mean = float(np.mean(log_amounts))
        amount_std = float(max(np.std(log_amounts), 1e-6))

        available_cols = [c for c in tabular_cols if c in df.columns]
        if available_cols:
            user_df = df.groupby("user_id", sort=False)[available_cols].first()
            tab_vals = user_df.to_numpy(dtype=np.float64)
            tab_mean = np.mean(tab_vals, axis=0).astype(np.float32)
            tab_std = np.maximum(np.std(tab_vals, axis=0), 1e-6).astype(np.float32)
        else:
            tab_mean = np.zeros(len(tabular_cols), dtype=np.float32)
            tab_std = np.ones(len(tabular_cols), dtype=np.float32)

        return cls(
            amount_mean=amount_mean,
            amount_std=amount_std,
            tabular_mean=tab_mean,
            tabular_std=tab_std,
        )


class FinancialEventDataset(Dataset):
    """
    PyTorch Dataset grouping user transaction histories into KVT sequences.

    Each item represents one user's recent transaction history (truncated to
    `max_seq_len` most recent events) along with standardized static features.

    Args:
        df: Transaction DataFrame containing `user_id`, `mcc_token`, `amount`,
            `time_delta`, `is_fraud`, and optional static tabular columns.
        max_seq_len: Maximum number of transactions retained per user.
        stats: Pre-fitted `NormalizationStats` from the training split. If
            `None`, statistics are fitted on `df` directly.
        tabular_cols: Static tabular feature column names.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_seq_len: int = 256,
        stats: Optional[NormalizationStats] = None,
        tabular_cols: Sequence[str] = TABULAR_COLUMNS,
    ):
        super().__init__()
        if df.empty:
            raise ValueError("FinancialEventDataset received an empty DataFrame.")

        self.max_seq_len = max_seq_len
        self.tabular_cols = [c for c in tabular_cols if c in df.columns]
        self.stats = stats if stats is not None else NormalizationStats.fit(df, self.tabular_cols)

        self.samples: List[Dict[str, torch.Tensor]] = []
        self._build_user_sequences(df)

    def _build_user_sequences(self, df: pd.DataFrame) -> None:
        """Groups transactions by `user_id` and converts them to KVT tensors."""
        grouped = df.groupby("user_id", sort=True)

        for _, group in grouped:
            # Retain the most recent `max_seq_len` transactions (where recent fraud bursts occur)
            if len(group) > self.max_seq_len:
                group = group.iloc[-self.max_seq_len :]

            keys = group["mcc_token"].to_numpy(dtype=np.int64)
            raw_amounts = np.clip(group["amount"].to_numpy(dtype=np.float64), 0.0, None)
            norm_amounts = (
                (np.log1p(raw_amounts) - self.stats.amount_mean) / self.stats.amount_std
            ).astype(np.float32)

            times = np.clip(group["time_delta"].to_numpy(dtype=np.int64), 0, 9999)

            if self.tabular_cols:
                raw_tab = group[self.tabular_cols].iloc[0].to_numpy(dtype=np.float32)
                norm_tab = (raw_tab - self.stats.tabular_mean) / self.stats.tabular_std
            else:
                norm_tab = np.zeros(3, dtype=np.float32)

            label = int(group["is_fraud"].max()) if "is_fraud" in group.columns else 0
            seq_len = len(keys)

            self.samples.append(
                {
                    "keys": torch.from_numpy(keys.copy()).long(),
                    "values": torch.from_numpy(norm_amounts.copy()).float(),
                    "times": torch.from_numpy(times.copy()).long(),
                    "tabular": torch.from_numpy(norm_tab.copy()).float(),
                    "label": torch.tensor([label], dtype=torch.long),
                    "seq_len": torch.tensor([seq_len], dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def dynamic_sequence_packing_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collates variable-length KVT sequences into a padded batch with strict attention masks.

    Generates:
      - `packed_keys`: `[B, max_len]` long tensor (padded with 0)
      - `packed_values`: `[B, max_len]` float tensor (padded with 0.0)
      - `packed_times`: `[B, max_len]` long tensor (padded with 0)
      - `key_padding_mask`: `[B, max_len]` boolean tensor (`False` = valid token,
        `True` = padded position to be ignored by attention and pooling)
      - `tabular`: `[B, tabular_dim]` float tensor
      - `labels`: `[B, 1]` long tensor
      - `seq_lens`: `[B]` long tensor
      - `cu_seqlens`: `[B + 1]` cumulative sequence lengths for Flash/Packed kernels
      - `max_seqlen`: integer maximum sequence length in the batch

    Args:
        batch: List of sample dictionaries from `FinancialEventDataset`.

    Returns:
        Collated dictionary ready for `MMTTTransformerEncoder` and `LargeDataModel`.
    """
    batch_size = len(batch)
    seq_lens = [int(item["keys"].shape[0]) for item in batch]
    max_len = max(seq_lens) if seq_lens else 1

    packed_keys = torch.zeros((batch_size, max_len), dtype=torch.long)
    packed_values = torch.zeros((batch_size, max_len), dtype=torch.float)
    packed_times = torch.zeros((batch_size, max_len), dtype=torch.long)
    # True indicates padding position in PyTorch's MultiheadAttention / TransformerEncoder
    key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    tabular_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []

    for i, item in enumerate(batch):
        length = seq_lens[i]
        packed_keys[i, :length] = item["keys"][:length]
        packed_values[i, :length] = item["values"][:length]
        packed_times[i, :length] = item["times"][:length]
        key_padding_mask[i, :length] = False

        tabular_list.append(item["tabular"].view(-1))
        label_list.append(item["label"].view(1))

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens_tensor, dim=0)

    return {
        "packed_keys": packed_keys,
        "packed_values": packed_values,
        "packed_times": packed_times,
        "key_padding_mask": key_padding_mask,
        "tabular": torch.stack(tabular_list, dim=0),
        "labels": torch.stack(label_list, dim=0),
        "seq_lens": seq_lens_tensor.long(),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_len,
    }

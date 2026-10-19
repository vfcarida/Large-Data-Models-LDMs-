"""
Synthetic Financial Transaction Prior-Data Generator.

============================================================================
PURPOSE
============================================================================
Generates realistic, multi-modal financial transaction trajectories paired
with static credit bureau attributes for pre-training and evaluating Large
Data Models (LDMs).

Inspired by synthetic prior-data generation in TabPFN v2 (Hollmann et al.,
Nature 2025) and banking event trajectory modeling in TransactionGPT (Dou
et al., Visa Research, 2025) and PRAGMA (Ostroukhov et al., Revolut, 2026),
this generator models:
  1. Static User Profiles: Credit bureau score, annual income, and age.
  2. Key-Value-Time (KVT) Sequences:
     - Key (`mcc_token`): 20 Merchant Category Codes distributed across
       everyday retail, travel/lifestyle, and high-risk financial channels.
     - Value (`amount`): Heavy-tailed log-normal transaction amounts
       conditioned on user income and merchant category.
     - Time (`time_delta`): Quantized inter-transaction time intervals
       capturing normal circadian rhythms vs. rapid automated bursts.
  3. Multi-Pattern Fraud Archetypes (`is_fraud`):
     - Pattern A (Velocity Structuring / Smurfing): Rapid consecutive
       transfers in high-risk MCCs (14-19) with near-zero time deltas.
     - Pattern B (Account Takeover / Probe-and-Sweep): Small digital verification
       probe followed by large anomalous purchases in electronics/crypto/wire.
     - Pattern C (Bust-Out Credit Abuse): Compromised bureau profile paired
       with sustained high-ticket cash advance and luxury liquidations.

============================================================================
REFERENCES
============================================================================
  - Hollmann et al. (2025): "Accurate predictions on small data with a
    tabular foundation model" (TabPFN v2, Nature 637:319-326)
  - Dou et al. (2025): "TransactionGPT: Foundation Models for Payment
    Trajectories" (Visa Research, arXiv:2511.08939)
  - Ostroukhov et al. (2026): "PRAGMA: A Self-Supervised Foundation Model
    for Banking Events" (Revolut/NVIDIA, arXiv:2604.08649)
  - Padhi et al. (2021): "Tabular Transformers for Modeling Multivariate
    Time Series" (IBM TabFormer, IEEE ICASSP, arXiv:2011.01843)
============================================================================
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical 20-category Merchant Category Code (MCC) vocabulary
# Indices 0..13: Normal everyday & lifestyle categories
# Indices 14..19: Elevated-risk / liquidation categories
MCC_VOCABULARY: Dict[int, str] = {
    0: "grocery_stores",
    1: "restaurants_dining",
    2: "gas_service_stations",
    3: "utilities_telecom",
    4: "pharmacies_healthcare",
    5: "public_transit_rideshare",
    6: "streaming_subscriptions",
    7: "department_stores",
    8: "apparel_footwear",
    9: "home_improvement",
    10: "airlines_lodging",
    11: "entertainment_events",
    12: "insurance_services",
    13: "education_books",
    14: "electronics_high_ticket",
    15: "digital_goods_gaming",
    16: "jewelry_luxury_watches",
    17: "gift_cards_prepaid",
    18: "wire_transfer_money_order",
    19: "crypto_fx_exchange",
}


class SyntheticTransactionGenerator:
    """
    Generates realistic multi-modal financial transaction histories.

    Produces a user-sorted transaction DataFrame containing both sequential
    KVT columns (`mcc_token`, `amount`, `time_delta`) and static user-level
    tabular features (`bureau_score`, `income`, `age`), along with binary
    fraud labels (`is_fraud`).

    Args:
        num_users: Total number of distinct users to simulate.
        min_transactions: Minimum sequence length per user.
        max_transactions: Maximum sequence length per user.
        fraud_rate: Target proportion of fraudulent users (in (0, 1)).
        vocab_size: Number of discrete MCC tokens (default: 20).
        seed: Deterministic random seed.
    """

    def __init__(
        self,
        num_users: int = 1000,
        min_transactions: int = 20,
        max_transactions: int = 200,
        fraud_rate: float = 0.02,
        vocab_size: int = 20,
        seed: int = 42,
    ):
        if num_users < 2:
            raise ValueError(f"num_users must be >= 2, got {num_users}")
        if min_transactions < 2 or max_transactions < min_transactions:
            raise ValueError(
                f"Invalid transaction bounds: min={min_transactions}, max={max_transactions}"
            )
        if not (0.0 < fraud_rate < 1.0):
            raise ValueError(f"fraud_rate must be in (0, 1), got {fraud_rate}")

        self.num_users = num_users
        self.min_transactions = min_transactions
        self.max_transactions = max_transactions
        self.fraud_rate = fraud_rate
        self.vocab_size = vocab_size
        self.seed = seed

    def generate(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generates the synthetic transaction DataFrame and optionally writes to Parquet.

        Args:
            save_path: Optional path to save the generated DataFrame as Parquet/CSV.

        Returns:
            DataFrame with columns:
                - `user_id` (int64)
                - `txn_idx` (int64)
                - `mcc_token` (int64 in [0, vocab_size - 1])
                - `amount` (float32, positive transaction value)
                - `time_delta` (int64 in [0, 9999], quantized time delta bin)
                - `bureau_score` (float32 in [300, 850])
                - `income` (float32, annual income)
                - `age` (float32 in [18, 80])
                - `is_fraud` (int64, 0 or 1 user-level fraud label)
        """
        rng = np.random.default_rng(self.seed)

        # Ensure a deterministic, well-conditioned number of fraudulent users
        # (at least 1 positive user for tiny tests, and at least 3 when num_users >= 30
        # so train/val/test stratified splits each receive positive instances)
        target_fraud_users = int(round(self.num_users * self.fraud_rate))
        if self.num_users >= 30:
            num_fraud_users = max(6, min(self.num_users - 6, target_fraud_users))
        else:
            num_fraud_users = max(1, min(self.num_users - 1, target_fraud_users))

        fraud_user_indices = set(
            rng.choice(self.num_users, size=num_fraud_users, replace=False).tolist()
        )

        # Normal MCC distribution (weighted toward everyday categories 0..13,
        # but with 15% legitimate activity in high-ticket categories 14..19)
        normal_probs = np.zeros(self.vocab_size, dtype=np.float64)
        low_risk_cut = max(1, int(self.vocab_size * 0.7))
        normal_probs[:low_risk_cut] = 1.0 / np.arange(1, low_risk_cut + 1) ** 0.65
        normal_probs[low_risk_cut:] = 0.12
        normal_probs /= normal_probs.sum()

        # Elevated-risk MCC distribution (higher weight on 14..19, but blended with normal MCCs)
        fraud_probs = np.zeros(self.vocab_size, dtype=np.float64)
        fraud_probs[:low_risk_cut] = 0.08
        fraud_probs[low_risk_cut:] = np.linspace(0.45, 0.95, self.vocab_size - low_risk_cut)
        fraud_probs /= fraud_probs.sum()

        records: List[Dict[str, object]] = []

        for user_id in range(self.num_users):
            is_fraud = 1 if user_id in fraud_user_indices else 0
            seq_len = int(rng.integers(self.min_transactions, self.max_transactions + 1))

            # Static profile generation with realistic distributional overlap
            if is_fraud:
                bureau_score = float(np.clip(rng.normal(655.0, 80.0), 300.0, 845.0))
                income = float(np.clip(rng.lognormal(mean=10.92, sigma=0.52), 18000.0, 380000.0))
                age = float(np.clip(rng.normal(39.0, 12.5), 18.0, 78.0))
            else:
                bureau_score = float(np.clip(rng.normal(675.0, 80.0), 320.0, 850.0))
                income = float(np.clip(rng.lognormal(mean=11.00, sigma=0.52), 20000.0, 450000.0))
                age = float(np.clip(rng.normal(41.0, 12.5), 18.0, 80.0))

            income_scale = (income / 60000.0) ** 0.35

            # Generate baseline legitimate sequence
            mccs = rng.choice(self.vocab_size, size=seq_len, p=normal_probs).astype(np.int64)
            base_log_mean = 3.4 + 0.08 * (mccs % 10)
            amounts = (
                rng.lognormal(mean=base_log_mean, sigma=0.65, size=seq_len) * income_scale
            ).astype(np.float64)
            # Normal inter-transaction time deltas: bins 5..450
            time_deltas = rng.integers(5, 450, size=seq_len, endpoint=True).astype(np.int64)
            time_deltas[0] = 0

            # 25% of legitimate users exhibit a holiday/travel high-velocity shopping burst
            # that partially overlaps with fraud velocity signatures
            if not is_fraud and rng.random() < 0.25:
                spree_len = max(3, int(seq_len * 0.25))
                s_idx = int(rng.integers(1, max(2, seq_len - spree_len + 1)))
                mccs[s_idx : s_idx + spree_len] = rng.choice(
                    self.vocab_size, size=spree_len, p=fraud_probs
                )
                amounts[s_idx : s_idx + spree_len] = rng.uniform(
                    160.0, 620.0, size=spree_len
                ) * income_scale
                time_deltas[s_idx : s_idx + spree_len] = rng.integers(1, 14, size=spree_len)

            if is_fraud:
                # Inject a fraud episode over 20-30% of the trajectory
                burst_len = max(3, int(seq_len * 0.25))
                start_idx = int(rng.integers(1, max(2, seq_len - burst_len + 1)))
                end_idx = min(seq_len, start_idx + burst_len)
                ep_len = end_idx - start_idx
                archetype = user_id % 3

                if archetype == 0:
                    # Archetype 0: Velocity Structuring / Smurfing
                    mccs[start_idx:end_idx] = rng.choice(
                        self.vocab_size, size=ep_len, p=fraud_probs
                    )
                    amounts[start_idx:end_idx] = rng.uniform(
                        200.0, 720.0, size=ep_len
                    ) * max(0.8, income_scale)
                    time_deltas[start_idx:end_idx] = rng.integers(
                        0, 9, size=ep_len, endpoint=True
                    )
                elif archetype == 1:
                    # Archetype 1: Account Takeover (Small probe followed by elevated sweep)
                    mccs[start_idx] = min(self.vocab_size - 1, 15)
                    amounts[start_idx] = float(rng.uniform(2.0, 12.0))
                    time_deltas[start_idx] = 2
                    if start_idx + 1 < end_idx:
                        rem = end_idx - (start_idx + 1)
                        mccs[start_idx + 1 : end_idx] = rng.choice(
                            self.vocab_size,
                            size=rem,
                            p=fraud_probs,
                        )
                        amounts[start_idx + 1 : end_idx] = rng.uniform(
                            260.0, 980.0, size=rem
                        )
                        time_deltas[start_idx + 1 : end_idx] = rng.integers(
                            1, 14, size=rem, endpoint=True
                        )
                else:
                    # Archetype 2: Stealth Camouflaged Escalation (subtle mimicry of normal behavior)
                    mccs[start_idx:end_idx] = rng.choice(
                        self.vocab_size, size=ep_len, p=0.5 * normal_probs + 0.5 * fraud_probs
                    )
                    escalation = np.linspace(1.1, 2.2, ep_len)
                    amounts[start_idx:end_idx] = (
                        rng.uniform(95.0, 380.0, size=ep_len) * escalation
                    )
                    time_deltas[start_idx:end_idx] = rng.integers(
                        4, 35, size=ep_len, endpoint=True
                    )

            time_deltas = np.clip(time_deltas, 0, 9999)
            amounts = np.clip(amounts, 0.50, 100000.0)

            for idx in range(seq_len):
                records.append(
                    {
                        "user_id": int(user_id),
                        "txn_idx": int(idx),
                        "mcc_token": int(mccs[idx]),
                        "amount": float(amounts[idx]),
                        "time_delta": int(time_deltas[idx]),
                        "bureau_score": float(bureau_score),
                        "income": float(income),
                        "age": float(age),
                        "is_fraud": int(is_fraud),
                    }
                )

        df = pd.DataFrame.from_records(records)

        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.suffix.lower() == ".parquet":
                df.to_parquet(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)
            logger.info(
                f"Generated {len(df):,} transactions across {self.num_users:,} users "
                f"({num_fraud_users} fraud) -> saved to {out_path}"
            )

        return df

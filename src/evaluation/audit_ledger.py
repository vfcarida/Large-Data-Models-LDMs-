"""
Cryptographic Tamper-Evident Audit Ledger (`HMAC-SHA256` Hash-Chaining) & Statistical Output Watermarking.

============================================================================
PURPOSE & REGULATORY ANCHORS
============================================================================
High-risk financial AI systems (real-time payment authorization, credit
underwriting, AML alerting) are subject to mandatory immutable record-keeping
and provenance verification under:
  1. EU AI Act (Regulation EU 2024/1689) Article 12 (Record-Keeping):
     - Requires automatic, tamper-evident logging of every automated decision,
       input hash, risk probability, conformal abstention interval, and adverse
       action reason code throughout the system lifecycle.
  2. EU AI Act Article 50 & NIST AI 100-1 (Output Provenance / Watermarking):
     - Requires machine-verifiable provenance markers on model-generated scores
       to detect unauthorized score tampering or shadow-model spoofing.
  3. Federal Reserve SR 11-7 (Model Risk Management):
     - Demands reproducible, cryptographically verifiable audit trails for
       regulatory examination.
============================================================================
"""

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class AuditRecord:
    """
    Immutable, hash-chained record of a single automated LDM decision.
    """

    sequence_index: int
    transaction_id: str
    entity_id: str
    risk_probability: float
    decision: str
    conformal_lower: float
    conformal_upper: float
    adverse_reasons: Tuple[str, ...]
    model_version: str
    prev_hash: str
    record_hash: str

    def canonical_payload(self) -> str:
        """
        Returns deterministic canonical JSON string (excluding `record_hash`)
        used to compute the `HMAC-SHA256` signature.
        """
        payload = {
            "sequence_index": self.sequence_index,
            "transaction_id": self.transaction_id,
            "entity_id": self.entity_id,
            "risk_probability": round(self.risk_probability, 8),
            "decision": self.decision,
            "conformal_lower": round(self.conformal_lower, 8),
            "conformal_upper": round(self.conformal_upper, 8),
            "adverse_reasons": list(self.adverse_reasons),
            "model_version": self.model_version,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "transaction_id": self.transaction_id,
            "entity_id": self.entity_id,
            "risk_probability": round(self.risk_probability, 8),
            "decision": self.decision,
            "conformal_lower": round(self.conformal_lower, 8),
            "conformal_upper": round(self.conformal_upper, 8),
            "adverse_reasons": list(self.adverse_reasons),
            "model_version": self.model_version,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }


class TamperEvidentAuditLedger:
    """
    Append-only `HMAC-SHA256` hash-chained ledger for high-risk financial AI
    decisions (`EU AI Act Art. 12` / `SR 11-7`).

    Any retrospective modification of a risk score, decision, adverse action
    reason code, or record ordering invalidates the cryptographic chain and is
    pinpointed by `verify_chain_integrity()`.
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        secret_key: str = "ldm-enterprise-audit-hmac-key",
        model_version: str = "0.7.0",
    ):
        self._secret_bytes = secret_key.encode("utf-8")
        self.model_version = model_version
        self._records: List[AuditRecord] = []

    def _compute_hmac(self, canonical_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            canonical_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def record_decision(
        self,
        transaction_id: str,
        entity_id: str,
        risk_probability: float,
        decision: str,
        conformal_interval: Tuple[float, float] = (0.05, 0.95),
        adverse_reasons: Optional[List[str]] = None,
    ) -> AuditRecord:
        """
        Appends a new cryptographically signed decision record to the hash chain.
        """
        seq_idx = len(self._records)
        prev_hash = (
            self.GENESIS_HASH
            if seq_idx == 0
            else self._records[-1].record_hash
        )
        reasons_tuple = tuple(adverse_reasons or [])

        draft = AuditRecord(
            sequence_index=seq_idx,
            transaction_id=str(transaction_id),
            entity_id=str(entity_id),
            risk_probability=round(float(risk_probability), 8),
            decision=str(decision),
            conformal_lower=round(float(conformal_interval[0]), 8),
            conformal_upper=round(float(conformal_interval[1]), 8),
            adverse_reasons=reasons_tuple,
            model_version=self.model_version,
            prev_hash=prev_hash,
            record_hash="",
        )
        sig = self._compute_hmac(draft.canonical_payload())
        signed_record = AuditRecord(
            sequence_index=draft.sequence_index,
            transaction_id=draft.transaction_id,
            entity_id=draft.entity_id,
            risk_probability=draft.risk_probability,
            decision=draft.decision,
            conformal_lower=draft.conformal_lower,
            conformal_upper=draft.conformal_upper,
            adverse_reasons=draft.adverse_reasons,
            model_version=draft.model_version,
            prev_hash=draft.prev_hash,
            record_hash=sig,
        )
        self._records.append(signed_record)
        return signed_record

    def verify_chain_integrity(self) -> Tuple[bool, Optional[int]]:
        """
        Verifies the entire hash chain from `GENESIS_HASH` to the latest record.

        Returns:
            Tuple `(is_valid, first_tampered_index)`:
              - `(True, None)` if all records and links are authentic.
              - `(False, idx)` indicating the exact 0-based index where tampering
                or link breakage occurred.
        """
        expected_prev = self.GENESIS_HASH
        for idx, rec in enumerate(self._records):
            if rec.sequence_index != idx:
                return False, idx
            if rec.prev_hash != expected_prev:
                return False, idx
            expected_sig = self._compute_hmac(rec.canonical_payload())
            if not hmac.compare_digest(rec.record_hash, expected_sig):
                return False, idx
            expected_prev = rec.record_hash
        return True, None

    @property
    def records(self) -> List[AuditRecord]:
        return list(self._records)


class StatisticalLogitWatermarker:
    r"""
    embeds and detects a secret-keyed cryptographic micro-watermark in continuous
    LDM risk logits (`EU AI Act Art. 50` provenance verification).

    For each sample $i$ with transaction/entity identifier $c_i$, a keyed PRF
    (`HMAC-SHA256`) deterministically derives a Rademacher sign $s_i \in \{-1, +1\}$
    and perturbs the logit by $s_i \cdot \delta_w$ ($\delta_w = 10^{-4}$), shifting
    probabilities by at most $2.5 \times 10^{-5}$ while enabling high-power
    $z$-score hypothesis testing across a batch of $N$ predictions.
    """

    def __init__(
        self,
        secret_key: str = "ldm-watermark-secret-key",
        delta_w: float = 1e-4,
        z_threshold: float = 3.0,
    ):
        self._secret_bytes = secret_key.encode("utf-8")
        self.delta_w = float(delta_w)
        self.z_threshold = float(z_threshold)

    def _derive_signs(self, context_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Deterministically maps integer `context_ids` `[B]` to $\{-1.0, +1.0\}$
        using `HMAC-SHA256`.
        """
        flat_ids = context_ids.detach().cpu().view(-1).tolist()
        signs: List[float] = []
        for cid in flat_ids:
            digest = hmac.new(
                self._secret_bytes,
                f"ctx:{int(cid)}".encode("utf-8"),
                hashlib.sha256,
            ).digest()
            bit = digest[0] & 1
            signs.append(1.0 if bit == 1 else -1.0)
        return torch.tensor(signs, dtype=torch.float32, device=context_ids.device)

    def embed_watermark(
        self,
        logits: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Embeds the keyed $\pm \delta_w$ watermark into `logits` `[B]` or `[B, 1]`.
        """
        signs = self._derive_signs(context_ids).to(dtype=logits.dtype)
        if logits.dim() == 2:
            signs = signs.view(-1, 1)
        return logits + self.delta_w * signs

    def detect_watermark(
        self,
        candidate_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Computes the aligned correlation $z$-score between residual perturbations
        and the secret Rademacher signs.

        Returns:
            Dictionary with `is_watermarked`, `z_score`, and `max_prob_shift`.
        """
        cand = candidate_logits.detach().float().view(-1)
        ref = reference_logits.detach().float().view(-1)
        signs = self._derive_signs(context_ids).view(-1)

        residuals = cand - ref
        aligned = torch.sign(residuals) * signs
        n = max(int(aligned.numel()), 1)
        # Under H0 (random/unwatermarked signs), sum(aligned) ~ N(0, n)
        z_score = float(aligned.sum().item()) / math.sqrt(float(n))
        max_prob_shift = float(
            (torch.sigmoid(cand) - torch.sigmoid(ref)).abs().max().item()
        )

        return {
            "is_watermarked": bool(z_score >= self.z_threshold),
            "z_score": round(z_score, 4),
            "max_prob_shift": round(max_prob_shift, 8),
        }

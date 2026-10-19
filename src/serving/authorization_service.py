"""
Real-Time Payment Authorization Service with Split-Conformal Governance & Adverse Action Explanations.

============================================================================
PURPOSE
============================================================================
Provides an enterprise-ready, low-latency payment authorization engine that
combines:
  1. Compiled TorchScript (`LDMInferenceWrapper`) sub-10ms forward scoring.
  2. Split-Conformal Risk Abstention (`ConformalRiskAbstainer`) routing
     decisions into `"APPROVE"`, `"DECLINE"`, or `"HUMAN_REVIEW"`.
  3. Regulatory Adverse Action Notice generation (`LDMExplainabilityEngine`)
     under ECOA / SR 11-7 / GDPR Article 22 when a transaction is declined
     or routed for manual review.
============================================================================
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import time

import torch

from src.evaluation.explainability import LDMExplainabilityEngine
from src.evaluation.metrics import ConformalRiskAbstainer
from src.models.ldm_model import LargeDataModel
from src.serving.exporter import LDMInferenceWrapper, export_to_torchscript


@dataclass(frozen=True)
class AuthorizationDecision:
    """Structured real-time payment authorization response payload."""

    decision: str
    decision_code: int
    fraud_probability: float
    conformal_lower_threshold: float
    conformal_upper_threshold: float
    latency_ms: float
    adverse_action_reasons: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dictionary representation."""
        return asdict(self)


class RealTimeAuthorizationService:
    """
    Low-Latency Streaming Payment Authorization Service.

    Args:
        model: Trained `LargeDataModel` instance.
        conformal_alpha: Target error tolerance for `ConformalRiskAbstainer`.
        use_torchscript: If True, compiles the model via `export_to_torchscript`
            for sub-10ms p99 inference latency.
        seq_len: Default sequence length used to trace the TorchScript graph.
    """

    DECISION_MAP = {
        0: "APPROVE",
        1: "DECLINE",
        -1: "HUMAN_REVIEW",
    }

    def __init__(
        self,
        model: LargeDataModel,
        conformal_alpha: float = 0.10,
        use_torchscript: bool = True,
        seq_len: int = 32,
    ):
        self.model = model
        self.model.eval()
        self.wrapper = LDMInferenceWrapper(self.model)
        self.abstainer = ConformalRiskAbstainer(alpha=conformal_alpha)
        self.explainer = LDMExplainabilityEngine(self.model)
        self.use_torchscript = use_torchscript

        self.traced_module: Optional[torch.jit.ScriptModule] = None
        if use_torchscript:
            self.traced_module, _ = export_to_torchscript(
                self.model, batch_size=1, seq_len=seq_len
            )

    def calibrate_governance(
        self,
        calibration_logits: torch.Tensor,
        calibration_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calibrates the split-conformal abstention thresholds on a held-out
        validation cohort.

        Returns:
            Dictionary with `"lower_threshold"` and `"upper_threshold"`.
        """
        self.abstainer.calibrate(calibration_logits, calibration_targets)
        return {
            "lower_threshold": float(self.abstainer.lower_threshold),
            "upper_threshold": float(self.abstainer.upper_threshold),
        }

    def authorize(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        explain_on_adverse: bool = True,
    ) -> AuthorizationDecision:
        """
        Scores a single entity trajectory (`batch_size=1`) and issues an
        audit-ready `AuthorizationDecision`.

        Args:
            packed_keys: LongTensor `[1, S]` of MCC tokens.
            packed_values: FloatTensor `[1, S]` of normalized transaction amounts.
            packed_times: LongTensor `[1, S]` of temporal delta bins.
            tabular_features: FloatTensor `[1, T]` of normalized static features.
            key_padding_mask: Optional BoolTensor `[1, S]` (`True` = padded).
            explain_on_adverse: If True, attaches Integrated Gradients adverse
                action reasons whenever the decision is `DECLINE` or `HUMAN_REVIEW`.

        Returns:
            `AuthorizationDecision` instance.
        """
        keys_1 = packed_keys[:1]
        vals_1 = packed_values[:1]
        times_1 = packed_times[:1]
        tab_1 = tabular_features[:1]
        if key_padding_mask is None:
            pad_1 = torch.zeros_like(keys_1, dtype=torch.bool)
        else:
            pad_1 = key_padding_mask[:1]

        t0 = time.perf_counter()
        with torch.no_grad():
            if self.traced_module is not None:
                logits, probs = self.traced_module(keys_1, vals_1, times_1, tab_1, pad_1)
            else:
                logits, probs = self.wrapper(keys_1, vals_1, times_1, tab_1, pad_1)

            decision_tensor = self.abstainer.predict_with_abstention(logits)
            decision_code = int(decision_tensor.view(-1)[0].item())
            prob_val = float(probs.view(-1)[0].item())

        latency_ms = (time.perf_counter() - t0) * 1000.0
        decision_label = self.DECISION_MAP.get(decision_code, "HUMAN_REVIEW")

        reasons_list: List[Dict[str, Any]] = []
        if explain_on_adverse and decision_code in (1, -1):
            extracted = self.explainer.extract_adverse_action_reasons(
                packed_keys=keys_1,
                packed_values=vals_1,
                packed_times=times_1,
                tabular_features=tab_1,
                key_padding_mask=pad_1,
                top_k_events=3,
                top_k_tabular=2,
                num_steps=8,
            )[0]
            reasons_list = [asdict(r) for r in extracted]

        return AuthorizationDecision(
            decision=decision_label,
            decision_code=decision_code,
            fraud_probability=prob_val,
            conformal_lower_threshold=float(self.abstainer.lower_threshold),
            conformal_upper_threshold=float(self.abstainer.upper_threshold),
            latency_ms=latency_ms,
            adverse_action_reasons=reasons_list,
        )

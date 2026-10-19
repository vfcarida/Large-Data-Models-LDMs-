"""
Regulatory Explainability & Counterfactual Algorithmic Recourse Engine.

============================================================================
PURPOSE
============================================================================
Provides axiomatic multi-modal feature attribution and sparse counterfactual
recourse generation for `LargeDataModel` (`LDM`), aligned with regulatory
mandates including US Federal Reserve SR 11-7, Equal Credit Opportunity Act
(ECOA) Adverse Action Notices, EU GDPR Article 22, and Gartner AI TRiSM.

Capabilities:
  1. Mask-Aware Integrated Gradients (`compute_integrated_gradients`):
     Computes path-integrated attributions (Sundararajan et al., ICML 2017)
     across sequential transaction amounts (`packed_values`) and static
     credit bureau features (`tabular_features`), strictly zeroing padded
     positions (`key_padding_mask`).
  2. Adverse Action Notice Extraction (`extract_adverse_action_reasons`):
     Maps top attributed sequential events (via `MCC_VOCABULARY`) and static
     tabular features into structured, human-readable regulatory reason codes.
  3. Sparse Counterfactual Algorithmic Recourse (`generate_counterfactual_recourse`):
     Solves an L1/L2-regularized projected gradient optimization problem
     (Wachter et al., Harvard JOLT 2017) to identify the minimal valid
     perturbation to continuous transaction/bureau features required to flip
     an unfavorable risk decision (`DECLINE` / `HUMAN_REVIEW` -> `APPROVE`).

============================================================================
REFERENCES
============================================================================
  - Sundararajan, M., Taly, A., & Yan, Q. (2017): "Axiomatic Attribution for
    Deep Networks" (ICML 2017, arXiv:1703.01365)
  - Wachter, S., Mittelstadt, B., & Russell, C. (2017): "Counterfactual
    Explanations without Opening the Black Box: Automated Decisions and the
    GDPR" (Harvard Journal of Law & Technology, 31(2), 841-887)
============================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn.functional as F

from src.data.synthetic_generator import MCC_VOCABULARY
from src.models.ldm_model import LargeDataModel


DEFAULT_TABULAR_NAMES: Sequence[str] = ("bureau_score", "income", "age")


@dataclass(frozen=True)
class AdverseActionReason:
    """Single regulator-ready adverse action reason code."""

    modality: str
    feature_name: str
    event_index: Optional[int]
    raw_attribution: float
    relative_share: float
    direction: str


@dataclass(frozen=True)
class CounterfactualRecourseResult:
    """Result of sparse counterfactual trajectory optimization."""

    initial_probability: float
    counterfactual_probability: float
    target_probability: float
    success: bool
    num_steps_taken: int
    counterfactual_values: torch.Tensor
    counterfactual_tabular: torch.Tensor
    delta_values: torch.Tensor
    delta_tabular: torch.Tensor
    tabular_changes: Dict[str, float]


class LDMExplainabilityEngine:
    """
    Integrated Gradients Attribution & Counterfactual Recourse Engine for LDM.

    Args:
        model: Trained `LargeDataModel` instance.
        tabular_feature_names: Ordered names of static tabular features.
    """

    def __init__(
        self,
        model: LargeDataModel,
        tabular_feature_names: Sequence[str] = DEFAULT_TABULAR_NAMES,
    ):
        self.model = model
        self.tabular_feature_names = list(tabular_feature_names)

    def compute_integrated_gradients(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        num_steps: int = 16,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Riemann path-integrated gradients with respect to continuous
        transaction values and static tabular features from a zero baseline.

        Args:
            packed_keys: LongTensor `[B, S]` of MCC tokens.
            packed_values: FloatTensor `[B, S]` of normalized transaction amounts.
            packed_times: LongTensor `[B, S]` of temporal delta bins.
            tabular_features: FloatTensor `[B, T]` of normalized static features.
            key_padding_mask: Optional BoolTensor `[B, S]` (`True` = padded).
            num_steps: Number of Riemann interpolation steps (default: 16).

        Returns:
            Dictionary containing:
              - `"value_attributions"`: Signed IG attributions `[B, S]`
              - `"tabular_attributions"`: Signed IG attributions `[B, T]`
              - `"event_importance_share"`: Normalized non-negative shares `[B, S]` summing to 1
              - `"tabular_importance_share"`: Normalized non-negative shares `[B, T]` summing to 1
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        was_training = self.model.training
        self.model.eval()

        baseline_values = torch.zeros_like(packed_values)
        baseline_tabular = torch.zeros_like(tabular_features)

        diff_values = packed_values - baseline_values
        diff_tabular = tabular_features - baseline_tabular

        total_grad_values = torch.zeros_like(packed_values)
        total_grad_tabular = torch.zeros_like(tabular_features)

        for step in range(1, num_steps + 1):
            alpha = float(step) / float(num_steps)
            interp_values = (baseline_values + alpha * diff_values).detach().requires_grad_(True)
            interp_tabular = (baseline_tabular + alpha * diff_tabular).detach().requires_grad_(True)

            with torch.enable_grad():
                logits = self.model(
                    packed_keys=packed_keys,
                    packed_values=interp_values,
                    packed_times=packed_times,
                    tabular_features=interp_tabular,
                    key_padding_mask=key_padding_mask,
                    update_memory=False,
                )
                scalar_out = logits.sum()
                grad_val, grad_tab = torch.autograd.grad(
                    outputs=scalar_out,
                    inputs=(interp_values, interp_tabular),
                    retain_graph=False,
                    create_graph=False,
                )

            total_grad_values = total_grad_values + grad_val.detach()
            total_grad_tabular = total_grad_tabular + grad_tab.detach()

        ig_values = diff_values * (total_grad_values / float(num_steps))
        ig_tabular = diff_tabular * (total_grad_tabular / float(num_steps))

        if key_padding_mask is not None:
            ig_values = ig_values.masked_fill(key_padding_mask, 0.0)

        abs_val = ig_values.abs()
        val_denom = abs_val.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        event_share = abs_val / val_denom

        abs_tab = ig_tabular.abs()
        tab_denom = abs_tab.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        tab_share = abs_tab / tab_denom

        if was_training:
            self.model.train()

        return {
            "value_attributions": ig_values,
            "tabular_attributions": ig_tabular,
            "event_importance_share": event_share,
            "tabular_importance_share": tab_share,
        }

    def extract_adverse_action_reasons(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        top_k_events: int = 3,
        top_k_tabular: int = 2,
        num_steps: int = 12,
    ) -> List[List[AdverseActionReason]]:
        """
        Extracts structured Adverse Action Reasons for each user in the batch.

        Args:
            packed_keys: LongTensor `[B, S]`.
            packed_values: FloatTensor `[B, S]`.
            packed_times: LongTensor `[B, S]`.
            tabular_features: FloatTensor `[B, T]`.
            key_padding_mask: Optional BoolTensor `[B, S]`.
            top_k_events: Number of top sequential events to report per user.
            top_k_tabular: Number of top static bureau features to report per user.
            num_steps: Integrated Gradients Riemann steps.

        Returns:
            List (length `B`) of ranked `AdverseActionReason` items.
        """
        attrs = self.compute_integrated_gradients(
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            tabular_features=tabular_features,
            key_padding_mask=key_padding_mask,
            num_steps=num_steps,
        )

        val_attr = attrs["value_attributions"].detach().cpu()
        tab_attr = attrs["tabular_attributions"].detach().cpu()
        ev_share = attrs["event_importance_share"].detach().cpu()
        tb_share = attrs["tabular_importance_share"].detach().cpu()
        keys_cpu = packed_keys.detach().cpu()

        batch_size, seq_len = val_attr.shape
        num_tab = tab_attr.shape[1]
        batch_reasons: List[List[AdverseActionReason]] = []

        for b in range(batch_size):
            user_reasons: List[AdverseActionReason] = []

            # Top sequential events by absolute attribution magnitude
            actual_k_ev = min(top_k_events, seq_len)
            ev_abs = val_attr[b].abs()
            if key_padding_mask is not None:
                ev_abs = ev_abs.masked_fill(key_padding_mask[b].cpu(), -1.0)
            _, top_ev_idx = torch.topk(ev_abs, k=actual_k_ev)

            for idx_tensor in top_ev_idx:
                idx = int(idx_tensor.item())
                if key_padding_mask is not None and bool(key_padding_mask[b, idx].item()):
                    continue
                mcc_id = int(keys_cpu[b, idx].item())
                mcc_name = MCC_VOCABULARY.get(mcc_id, f"mcc_{mcc_id}")
                raw_score = float(val_attr[b, idx].item())
                share = float(ev_share[b, idx].item())
                direction = "risk_increasing" if raw_score >= 0.0 else "risk_mitigating"
                user_reasons.append(
                    AdverseActionReason(
                        modality="sequential_transaction",
                        feature_name=f"event[{idx}]:{mcc_name}",
                        event_index=idx,
                        raw_attribution=raw_score,
                        relative_share=share,
                        direction=direction,
                    )
                )

            # Top static tabular features
            actual_k_tb = min(top_k_tabular, num_tab)
            _, top_tb_idx = torch.topk(tab_attr[b].abs(), k=actual_k_tb)
            for t_tensor in top_tb_idx:
                t_idx = int(t_tensor.item())
                fname = (
                    self.tabular_feature_names[t_idx]
                    if t_idx < len(self.tabular_feature_names)
                    else f"tabular_{t_idx}"
                )
                raw_score = float(tab_attr[b, t_idx].item())
                share = float(tb_share[b, t_idx].item())
                direction = "risk_increasing" if raw_score >= 0.0 else "risk_mitigating"
                user_reasons.append(
                    AdverseActionReason(
                        modality="static_bureau",
                        feature_name=fname,
                        event_index=None,
                        raw_attribution=raw_score,
                        relative_share=share,
                        direction=direction,
                    )
                )

            batch_reasons.append(user_reasons)

        return batch_reasons

    def generate_counterfactual_recourse(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        target_probability: float = 0.25,
        max_steps: int = 40,
        step_size: float = 0.15,
        l1_weight: float = 0.02,
        l2_weight: float = 0.01,
    ) -> CounterfactualRecourseResult:
        """
        Finds a sparse counterfactual perturbation `(delta_values, delta_tabular)`
        that reduces the predicted fraud probability below `target_probability`.

        Operates on a single entity (`batch_size == 1`) or the first entity of a batch.

        Args:
            packed_keys: LongTensor `[1, S]`.
            packed_values: FloatTensor `[1, S]`.
            packed_times: LongTensor `[1, S]`.
            tabular_features: FloatTensor `[1, T]`.
            key_padding_mask: Optional BoolTensor `[1, S]`.
            target_probability: Desired maximum fraud probability (`APPROVE` threshold).
            max_steps: Maximum proximal gradient steps.
            step_size: Optimization learning rate.
            l1_weight: L1 sparsity regularization coefficient.
            l2_weight: L2 proximity regularization coefficient.

        Returns:
            `CounterfactualRecourseResult` with counterfactual tensors and deltas.
        """
        was_training = self.model.training
        self.model.eval()

        keys_1 = packed_keys[:1].detach()
        orig_values = packed_values[:1].detach().clone()
        times_1 = packed_times[:1].detach()
        orig_tabular = tabular_features[:1].detach().clone()
        pad_1 = key_padding_mask[:1].detach() if key_padding_mask is not None else None

        with torch.no_grad():
            init_logit = self.model(
                keys_1,
                orig_values,
                times_1,
                tabular_features=orig_tabular,
                key_padding_mask=pad_1,
                update_memory=False,
            )
            init_prob = float(torch.sigmoid(init_logit).item())

        delta_val = torch.zeros_like(orig_values, requires_grad=True)
        delta_tab = torch.zeros_like(orig_tabular, requires_grad=True)

        optimizer = torch.optim.Adam([delta_val, delta_tab], lr=step_size)
        target_zero = torch.zeros((1, 1), dtype=orig_values.dtype, device=orig_values.device)

        steps_taken = 0
        curr_prob = init_prob

        for step in range(1, max_steps + 1):
            if curr_prob <= target_probability:
                break

            steps_taken = step
            optimizer.zero_grad()

            if pad_1 is not None:
                masked_delta_val = delta_val.masked_fill(pad_1, 0.0)
            else:
                masked_delta_val = delta_val

            cf_val = orig_values + masked_delta_val
            cf_tab = orig_tabular + delta_tab

            logits = self.model(
                keys_1,
                cf_val,
                times_1,
                tabular_features=cf_tab,
                key_padding_mask=pad_1,
                update_memory=False,
            )
            curr_prob = float(torch.sigmoid(logits.detach()).item())
            if curr_prob <= target_probability:
                break

            bce_loss = F.binary_cross_entropy_with_logits(logits, target_zero)
            sparsity_loss = l1_weight * (
                masked_delta_val.abs().mean() + delta_tab.abs().mean()
            )
            proximity_loss = l2_weight * (
                masked_delta_val.pow(2).mean() + delta_tab.pow(2).mean()
            )
            total_loss = bce_loss + sparsity_loss + proximity_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                if pad_1 is not None:
                    delta_val.masked_fill_(pad_1, 0.0)

        with torch.no_grad():
            final_delta_val = delta_val.detach()
            if pad_1 is not None:
                final_delta_val = final_delta_val.masked_fill(pad_1, 0.0)
            final_delta_tab = delta_tab.detach()
            final_cf_val = orig_values + final_delta_val
            final_cf_tab = orig_tabular + final_delta_tab

            final_logit = self.model(
                keys_1,
                final_cf_val,
                times_1,
                tabular_features=final_cf_tab,
                key_padding_mask=pad_1,
                update_memory=False,
            )
            final_prob = float(torch.sigmoid(final_logit).item())

        tab_changes: Dict[str, float] = {}
        for idx in range(final_delta_tab.shape[1]):
            name = (
                self.tabular_feature_names[idx]
                if idx < len(self.tabular_feature_names)
                else f"tabular_{idx}"
            )
            tab_changes[name] = float(final_delta_tab[0, idx].item())

        if was_training:
            self.model.train()

        return CounterfactualRecourseResult(
            initial_probability=init_prob,
            counterfactual_probability=final_prob,
            target_probability=target_probability,
            success=bool(final_prob <= target_probability or final_prob < init_prob),
            num_steps_taken=steps_taken,
            counterfactual_values=final_cf_val,
            counterfactual_tabular=final_cf_tab,
            delta_values=final_delta_val,
            delta_tabular=final_delta_tab,
            tabular_changes=tab_changes,
        )

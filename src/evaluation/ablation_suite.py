"""
Automated Scientific Ablation & Pareto Comparison Suite.

============================================================================
PURPOSE
============================================================================
Quantifies the incremental discrimination and calibration contribution of
each subsystem within `LargeDataModel` (`LDM`) over an identical test cohort:

  1. `tabular_dcnv2_only`: Static credit bureau features + DCNv2 cross
     network only (zeroed sequential transaction representation).
  2. `mmtt_backbone_only`: Pre-trained MMTT Transformer + Periodic Numerical
     Embeddings (PLR) + DCNv2 (without Relational GNN or TabR retrieval).
  3. `mmtt_plus_relational_gnn`: MMTT + Bipartite User-Merchant Graph Neural
     Network (`RelBench`) + DCNv2.
  4. `full_ldm_hybrid`: Full architecture combining MMTT + Relational GNN +
     TabR kNN Memory Retrieval + DCNv2.
  5. `in_context_pfn_cold_start`: Zero-gradient single-pass posterior
     inference via `InContextPFNHead` (`TabPFN v2`) conditioned on a support
     cohort.
============================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from src.evaluation.metrics import (
    calculate_auc,
    calculate_brier_score,
    calculate_ece,
    calculate_pr_auc,
)
from src.models.ldm_model import LargeDataModel


@dataclass(frozen=True)
class AblationVariantMetrics:
    """Discrimination and calibration metrics for a single architectural variant."""

    variant_name: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    ece: float


class AblationBenchmarkSuite:
    """
    Executes multi-pathway architectural ablations on a `LargeDataModel`.

    Args:
        model: `LargeDataModel` instance.
        device: Target evaluation device (`"cpu"` or `"cuda"`).
    """

    def __init__(self, model: LargeDataModel, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def run_ablation_study(
        self,
        test_dataloader: Any,
        support_dataloader: Optional[Any] = None,
    ) -> List[AblationVariantMetrics]:
        """
        Evaluates all architectural variants across `test_dataloader`.

        Args:
            test_dataloader: Test DataLoader yielding packed KVT + tabular batches.
            support_dataloader: Optional support DataLoader for `InContextPFNHead`.
                If None, uses the first batch of `test_dataloader` as support.

        Returns:
            List of `AblationVariantMetrics` across the evaluated variants.
        """
        was_training = self.model.training
        self.model.eval()
        self.model.to(self.device)

        logits_tabular_only: List[torch.Tensor] = []
        logits_mmtt_only: List[torch.Tensor] = []
        logits_mmtt_gnn: List[torch.Tensor] = []
        logits_full: List[torch.Tensor] = []
        logits_pfn: List[torch.Tensor] = []
        all_labels_list: List[torch.Tensor] = []

        # Prepare a representative multi-batch support cohort for InContextPFNHead
        sup_source = support_dataloader if support_dataloader is not None else test_dataloader
        sup_batches = []
        for idx, b in enumerate(sup_source):
            sup_batches.append(b)
            if idx >= 2:
                break
        max_sup_len = max(int(b["packed_keys"].shape[1]) for b in sup_batches)

        def _pad_2d(t: torch.Tensor, target_len: int, pad_val: Any = 0) -> torch.Tensor:
            diff = target_len - int(t.shape[1])
            if diff <= 0:
                return t
            return torch.nn.functional.pad(t, (0, diff), value=pad_val)

        sup_keys = torch.cat(
            [_pad_2d(b["packed_keys"], max_sup_len, 0) for b in sup_batches], dim=0
        ).to(self.device)
        sup_vals = torch.cat(
            [_pad_2d(b["packed_values"], max_sup_len, 0.0) for b in sup_batches], dim=0
        ).to(self.device)
        sup_times = torch.cat(
            [_pad_2d(b["packed_times"], max_sup_len, 0) for b in sup_batches], dim=0
        ).to(self.device)
        sup_tab = torch.cat([b["tabular"] for b in sup_batches], dim=0).to(self.device)
        sup_labels = torch.cat([b["labels"] for b in sup_batches], dim=0).to(self.device)
        if sup_batches[0].get("key_padding_mask") is not None:
            sup_pad = torch.cat(
                [_pad_2d(b["key_padding_mask"], max_sup_len, True) for b in sup_batches],
                dim=0,
            ).to(self.device)
        else:
            sup_pad = None

        for batch in test_dataloader:
            keys = batch["packed_keys"].to(self.device)
            vals = batch["packed_values"].to(self.device)
            times = batch["packed_times"].to(self.device)
            tab = batch["tabular"].to(self.device)
            labels = batch["labels"].to(self.device)
            pad = batch.get("key_padding_mask")
            if pad is not None:
                pad = pad.to(self.device)

            # 1. Pooled MMTT sequence representation
            pooled_mmtt = self.model.get_pooled_embedding(
                keys, vals, times, key_padding_mask=pad
            )

            # Variant 1: Static Tabular + DCNv2 only (zeroed sequence representation)
            zero_seq = torch.zeros_like(pooled_mmtt)
            out_tab_only = self.model.fusion_head(zero_seq, tab)

            # Variant 2: MMTT Backbone + DCNv2 only
            out_mmtt_only = self.model.fusion_head(pooled_mmtt, tab)

            # Variant 3: MMTT + Relational GNN + DCNv2
            if self.model.relational_gnn is not None:
                gnn_emb = self.model.relational_gnn(
                    user_embeddings=pooled_mmtt,
                    event_keys=keys,
                    event_values=vals,
                    key_padding_mask=pad,
                )
            else:
                gnn_emb = pooled_mmtt
            out_mmtt_gnn = self.model.fusion_head(gnn_emb, tab)

            # Variant 4: Full LDM (MMTT + Relational GNN + TabR Retrieval + DCNv2)
            out_full = self.model(
                keys,
                vals,
                times,
                tabular_features=tab,
                key_padding_mask=pad,
                update_memory=False,
            )

            # Variant 5: In-Context PFN Head (TabPFN v2 cold-start posterior)
            out_pfn = self.model.forward_in_context(
                query_keys=keys,
                query_values=vals,
                query_times=times,
                query_tabular=tab,
                support_keys=sup_keys,
                support_values=sup_vals,
                support_times=sup_times,
                support_tabular=sup_tab,
                support_labels=sup_labels,
                query_padding_mask=pad,
                support_padding_mask=sup_pad,
            )

            logits_tabular_only.append(out_tab_only.cpu())
            logits_mmtt_only.append(out_mmtt_only.cpu())
            logits_mmtt_gnn.append(out_mmtt_gnn.cpu())
            logits_full.append(out_full.cpu())
            logits_pfn.append(out_pfn.cpu())
            all_labels_list.append(labels.cpu())

        if was_training:
            self.model.train()

        targets = torch.cat(all_labels_list, dim=0)

        variant_map = [
            ("1. Static Tabular + DCNv2 Only", torch.cat(logits_tabular_only, dim=0)),
            ("2. MMTT + PLR + DCNv2", torch.cat(logits_mmtt_only, dim=0)),
            ("3. MMTT + RelBench Bipartite GNN + DCNv2", torch.cat(logits_mmtt_gnn, dim=0)),
            ("4. Full LDM (MMTT + GNN + TabR + DCNv2)", torch.cat(logits_full, dim=0)),
            ("5. In-Context PFN Head (Cold-Start ICL)", torch.cat(logits_pfn, dim=0)),
        ]

        results: List[AblationVariantMetrics] = []
        for name, l_tensor in variant_map:
            results.append(
                AblationVariantMetrics(
                    variant_name=name,
                    roc_auc=float(calculate_auc(l_tensor, targets)),
                    pr_auc=float(calculate_pr_auc(l_tensor, targets)),
                    brier_score=float(calculate_brier_score(l_tensor, targets)),
                    ece=float(calculate_ece(l_tensor, targets)),
                )
            )

        return results

    @staticmethod
    def format_markdown_table(metrics_list: List[AblationVariantMetrics]) -> str:
        """Formats ablation study results as a publication-grade Markdown table."""
        lines = [
            "| Architectural Variant | ROC-AUC | PR-AUC | Brier Score | ECE |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for m in metrics_list:
            lines.append(
                f"| {m.variant_name} | {m.roc_auc:.4f} | {m.pr_auc:.4f} | "
                f"{m.brier_score:.4f} | {m.ece:.4f} |"
            )
        return "\n".join(lines)

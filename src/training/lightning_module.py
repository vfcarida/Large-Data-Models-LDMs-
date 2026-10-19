"""
PyTorch Lightning Training Modules for Large Data Models (LDMs).

============================================================================
PURPOSE
============================================================================
Encapsulates the training, validation, and testing workflows for both phases
of the LDM lifecycle:

1. LDMPreTrainingModule (Self-Supervised Pre-Training):
   - Combines Masked Joint-Distribution Modeling (LimiX / PRAGMA) across
     categorical keys, continuous amounts, temporal deltas, and predictor-head
     latent targets with Contrastive Event Sequence Learning (CoLES).
   - Eliminates representation collapse by requiring both per-token multi-modal
     feature recovery and batch-wise user contrastive separation.

2. LDMFineTuningModule (Supervised Risk Classification):
   - Uses the pre-trained MMTT Transformer encoder + parallel DCNv2 fusion head.
   - Optimizes class-weighted Binary Cross-Entropy with differential learning
     rates and logs ROC-AUC, PR-AUC, F1, Precision, Recall, and Brier Score.

============================================================================
REFERENCES
============================================================================
  - LimiX (2025/2026): Masked joint-distribution pre-training (arXiv:2509.03505)
  - PRAGMA (2026): Multi-modal banking event SSL (arXiv:2604.08649)
  - Babaev et al. (2022): CoLES contrastive event sequences (arXiv:2002.08232)
============================================================================
"""

from typing import Any, Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl

from src.models.ldm_model import LargeDataModel
from src.training.objectives import (
    ContextConditionalMaskedLoss,
    ContrastiveEventSequenceLoss,
    NextTokenPredictionLoss,
)
from src.training.lr_scheduler import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class LDMPreTrainingModule(pl.LightningModule):
    """
    Lightning Module for Collapse-Free LDM Self-Supervised Pre-Training.

    Jointly optimizes:
      1. Multi-modal KVT feature reconstruction at masked positions (`L_multimodal`)
      2. Predictor-head latent consistency at masked positions (`L_latent`)
      3. Contrastive Event Sequence alignment across two stochastic views (`L_coles`)

    Total SSL Loss:
        L_SSL = L_multimodal + 0.5 * L_latent + 0.25 * L_coles

    Args:
        model: `LargeDataModel` instance.
        config: Dictionary with training hyperparameters.
    """

    def __init__(
        self,
        model: LargeDataModel,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = model

        self.masked_loss = ContextConditionalMaskedLoss(
            masking_prob=config.get("ssl_masking_prob", 0.15)
        )
        self.contrastive_loss = ContrastiveEventSequenceLoss(
            temperature=config.get("ssl_temperature", 0.15)
        )

        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 100)
        self.max_steps = config.get("max_steps", 10000)

    def _safe_log(self, name: str, value: Any, **kwargs) -> None:
        """Logs metric to Lightning Trainer when attached, avoiding standalone warnings."""
        if getattr(self, "_trainer", None) is not None:
            self.log(name, value, **kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encodes transaction sequences in bidirectional mode."""
        return self.model.encode(
            batch["packed_keys"],
            batch["packed_values"],
            batch["packed_times"],
            key_padding_mask=batch.get("key_padding_mask"),
            is_causal=False,
        )

    def _corrupt_inputs(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        times: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies multi-modal masking to keys, values, and times at masked positions."""
        masked_keys = keys.clone()
        noisy_values = values.clone()
        masked_times = times.clone()

        if noisy_values.dim() == 1:
            flat_mask = input_mask.squeeze(0) if input_mask.dim() == 2 else input_mask
            flat_mask = flat_mask[: noisy_values.shape[0]]
            masked_keys[flat_mask] = 0
            noisy_values[flat_mask] = 0.0
            masked_times[flat_mask] = 0
        else:
            masked_keys[input_mask] = 0
            noisy_values[input_mask] = 0.0
            masked_times[input_mask] = 0

        return masked_keys, noisy_values, masked_times

    def _compute_ssl_step_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes joint multi-modal reconstruction + latent + CoLES contrastive loss."""
        keys = batch["packed_keys"]
        values = batch["packed_values"]
        times = batch["packed_times"]
        key_padding_mask = batch.get("key_padding_mask")

        # 1. Target clean representations (stop-gradient teacher view)
        with torch.no_grad():
            original_embeddings = self.model.encode(
                keys,
                values,
                times,
                key_padding_mask=key_padding_mask,
                is_causal=False,
            )

        if keys.dim() == 2:
            B, S = keys.shape
        else:
            B, S = 1, keys.shape[0]

        # 2. Generate two independent heterogeneous masks (for reconstruction and CoLES)
        input_mask_1 = self.masked_loss.heterogeneous_mask_schedule(B, S, keys.device)
        input_mask_2 = self.masked_loss.heterogeneous_mask_schedule(B, S, keys.device)

        if key_padding_mask is not None and input_mask_1.dim() == 2:
            input_mask_1 = input_mask_1 & (~key_padding_mask)
            input_mask_2 = input_mask_2 & (~key_padding_mask)

        masked_keys_1, noisy_values_1, masked_times_1 = self._corrupt_inputs(
            keys, values, times, input_mask_1
        )
        masked_keys_2, noisy_values_2, masked_times_2 = self._corrupt_inputs(
            keys, values, times, input_mask_2
        )

        # 3. Encode corrupted view 1 and predict masked modalities
        reconstructed_1 = self.model.encode(
            masked_keys_1,
            noisy_values_1,
            masked_times_1,
            key_padding_mask=key_padding_mask,
            is_causal=False,
        )
        preds_1 = self.model.predict_masked_modalities(reconstructed_1)

        num_virtual = reconstructed_1.shape[1] - S
        if num_virtual > 0:
            if input_mask_1.dim() == 2:
                v_pad = torch.zeros((B, num_virtual), dtype=torch.bool, device=keys.device)
                loss_mask = torch.cat([v_pad, input_mask_1], dim=1)
            else:
                v_pad = torch.zeros((num_virtual,), dtype=torch.bool, device=keys.device)
                loss_mask = torch.cat([v_pad, input_mask_1], dim=0)
        else:
            loss_mask = input_mask_1

        if key_padding_mask is not None:
            if num_virtual > 0:
                v_pad = torch.zeros((B, num_virtual), dtype=torch.bool, device=keys.device)
                full_padding_mask = torch.cat([v_pad, key_padding_mask], dim=1)
            else:
                full_padding_mask = key_padding_mask
            loss_mask = loss_mask & (~full_padding_mask)

        # Asymmetric predictor latent consistency loss (BYOL/JEPA style)
        latent_loss = self.masked_loss(
            preds_1["latent_pred"], original_embeddings.detach(), loss_mask
        )

        # Multi-modal KVT feature reconstruction loss (LimiX / PRAGMA style)
        token_mask = input_mask_1 if keys.dim() == 2 else input_mask_1.squeeze(0)
        multimodal_loss, mm_metrics = self.masked_loss.compute_multimodal_loss(
            preds_1,
            keys if keys.dim() == 2 else keys.unsqueeze(0),
            values if values.dim() == 2 else values.unsqueeze(0),
            times if times.dim() == 2 else times.unsqueeze(0),
            token_mask,
        )

        # 4. Contrastive Event Sequence Loss (CoLES) across pooled views 1 and 2
        z1 = self.model.get_pooled_embedding(
            masked_keys_1,
            noisy_values_1,
            masked_times_1,
            key_padding_mask=key_padding_mask,
        )
        z2 = self.model.get_pooled_embedding(
            masked_keys_2,
            noisy_values_2,
            masked_times_2,
            key_padding_mask=key_padding_mask,
        )
        coles_loss = self.contrastive_loss(z1, z2)

        total_loss = multimodal_loss + 0.5 * latent_loss + 0.25 * coles_loss
        metrics = {
            "multimodal_loss": float(multimodal_loss.detach().item()),
            "latent_loss": float(latent_loss.detach().item()),
            "coles_loss": float(coles_loss.detach().item()),
            **mm_metrics,
        }
        return total_loss, metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Executes a single self-supervised pre-training step."""
        loss, metrics = self._compute_ssl_step_loss(batch)
        self._safe_log("pretrain/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._safe_log("pretrain/coles_loss", metrics["coles_loss"], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Executes a single self-supervised validation step."""
        loss, _ = self._compute_ssl_step_loss(batch)
        self._safe_log("pretrain/val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures AdamW optimizer with linear warmup and cosine decay."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class LDMFineTuningModule(pl.LightningModule):
    """
    Lightning Module for LDM Supervised Fine-Tuning.

    Trains the pre-trained MMTT Transformer encoder and parallel DCNv2 fusion
    head for imbalanced binary risk classification using differential learning
    rates and class-weighted Binary Cross-Entropy loss.

    Args:
        model: `LargeDataModel` instance (typically pre-trained).
        config: Dictionary with fine-tuning hyperparameters.
    """

    def __init__(
        self,
        model: LargeDataModel,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = model

        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 50)
        self.max_steps = config.get("max_steps", 5000)
        self.threshold = config.get("metric_threshold", 0.5)

        pos_weight = float(config.get("pos_weight", 10.0))
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _safe_log(self, name: str, value: Any, **kwargs) -> None:
        """Logs metric to Lightning Trainer when attached, avoiding standalone warnings."""
        if getattr(self, "_trainer", None) is not None:
            self.log(name, value, **kwargs)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        update_memory: bool = False,
    ) -> torch.Tensor:
        """Forward pass for binary classification."""
        return self.model(
            batch["packed_keys"],
            batch["packed_values"],
            batch["packed_times"],
            tabular_features=batch.get("tabular"),
            key_padding_mask=batch.get("key_padding_mask"),
            labels=batch.get("labels"),
            update_memory=update_memory,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Executes a single supervised fine-tuning step."""
        logits = self.forward(batch, update_memory=True)
        labels = batch["labels"].float().view(-1, 1)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight_tensor.to(logits.device),
        )

        # Episodic training of the In-Context PFN head on detached representations
        batch_size = int(labels.shape[0])
        if batch_size >= 4 and batch.get("tabular") is not None:
            half = batch_size // 2
            pad = batch.get("key_padding_mask")
            pfn_logits = self.model.forward_in_context(
                query_keys=batch["packed_keys"][half:],
                query_values=batch["packed_values"][half:],
                query_times=batch["packed_times"][half:],
                query_tabular=batch["tabular"][half:],
                support_keys=batch["packed_keys"][:half],
                support_values=batch["packed_values"][:half],
                support_times=batch["packed_times"][:half],
                support_tabular=batch["tabular"][:half],
                support_labels=labels[:half],
                query_padding_mask=pad[half:] if pad is not None else None,
                support_padding_mask=pad[:half] if pad is not None else None,
                detach_backbone=True,
            )
            pfn_loss = F.binary_cross_entropy_with_logits(
                pfn_logits,
                labels[half:],
                pos_weight=self.pos_weight_tensor.to(logits.device),
            )
            loss = loss + pfn_loss

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()
            accuracy = (preds == labels).float().mean()

        self._safe_log("finetune/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._safe_log("finetune/train_acc", accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Collects validation predictions for epoch-level risk evaluation."""
        logits = self.forward(batch)
        labels = batch["labels"].float().view(-1, 1)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight_tensor.to(logits.device),
        )

        self._safe_log("finetune/val_loss", loss, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append(
            {
                "logits": logits.detach().cpu(),
                "labels": labels.detach().cpu(),
            }
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Computes aggregate validation metrics across the full validation set."""
        if not self.validation_step_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

        from src.evaluation.metrics import (
            calculate_auc,
            calculate_pr_auc,
            calculate_f1_flexible,
            calculate_brier_score,
            calculate_ece,
        )

        auc = calculate_auc(all_logits, all_labels)
        pr_auc = calculate_pr_auc(all_logits, all_labels)
        f1 = calculate_f1_flexible(all_logits, all_labels, self.threshold)
        brier = calculate_brier_score(all_logits, all_labels)
        ece = calculate_ece(all_logits, all_labels)

        probs = torch.sigmoid(all_logits).view(-1)
        preds = (probs >= self.threshold).float()
        targets = all_labels.view(-1).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()

        self._safe_log("finetune/val_auc", auc, prog_bar=True)
        self._safe_log("finetune/val_pr_auc", pr_auc, prog_bar=True)
        self._safe_log("finetune/val_f1", f1, prog_bar=True)
        self._safe_log("finetune/val_precision", precision)
        self._safe_log("finetune/val_recall", recall)
        self._safe_log("finetune/val_brier", brier)
        self._safe_log("finetune/val_ece", ece)

        self.validation_step_outputs.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Collects test predictions for final epoch-level evaluation."""
        logits = self.forward(batch)
        labels = batch["labels"].float().view(-1, 1)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight_tensor.to(logits.device),
        )

        self._safe_log("test/loss", loss)

        self.test_step_outputs.append(
            {
                "logits": logits.detach().cpu(),
                "labels": labels.detach().cpu(),
            }
        )
        return loss

    def on_test_epoch_end(self) -> None:
        """Computes final test discrimination and calibration metrics."""
        if not self.test_step_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        from src.evaluation.metrics import (
            calculate_auc,
            calculate_pr_auc,
            calculate_f1_flexible,
            calculate_recall_at_fpr,
            calculate_brier_score,
            calculate_ece,
        )

        auc = calculate_auc(all_logits, all_labels)
        pr_auc = calculate_pr_auc(all_logits, all_labels)
        f1 = calculate_f1_flexible(all_logits, all_labels, self.threshold)
        recall_at_5pct_fpr = calculate_recall_at_fpr(all_logits, all_labels, max_fpr=0.05)
        brier = calculate_brier_score(all_logits, all_labels)
        ece = calculate_ece(all_logits, all_labels)

        self._safe_log("test/auc", auc)
        self._safe_log("test/pr_auc", pr_auc)
        self._safe_log("test/f1", f1)
        self._safe_log("test/recall_at_5pct_fpr", recall_at_5pct_fpr)
        self._safe_log("test/brier_score", brier)
        self._safe_log("test/ece", ece)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configures AdamW with differential learning rates.

        The pre-trained encoder uses a 0.25x learning rate multiplier relative
        to the DCNv2 fusion head to preserve pre-trained structure while
        adapting to the supervised risk task.
        """
        encoder_params = list(self.model.encoder.parameters())
        head_params = list(self.model.fusion_head.parameters())
        if self.model.relational_gnn is not None:
            head_params.extend(list(self.model.relational_gnn.parameters()))
        if self.model.tabr_retrieval is not None:
            head_params.extend(list(self.model.tabr_retrieval.parameters()))
        if self.model.pfn_head is not None:
            head_params.extend(list(self.model.pfn_head.parameters()))

        param_groups = [
            {"params": encoder_params, "lr": self.learning_rate * 0.25},
            {"params": head_params, "lr": self.learning_rate},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

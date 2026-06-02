"""
PyTorch Lightning Training Modules for LDM.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
These modules encapsulate the COMPLETE training logic for both phases:

1. LDMPreTrainingModule:
   - Self-supervised masked prediction (LimiX objective)
   - No labels required — learns from data structure
   - Produces a pre-trained encoder checkpoint

2. LDMFineTuningModule:
   - Supervised binary classification (fraud detection)
   - Uses pre-trained encoder + DCNv2 fusion head
   - Produces the final model with real metrics

Why PyTorch Lightning?
  - Eliminates boilerplate (training loops, device placement, logging)
  - Built-in distributed training (DDP) support
  - Automatic mixed precision (fp16/bf16)
  - Integrated experiment tracking (WandB, TensorBoard)
  - Reproducibility through seed management

============================================================================
REFERENCES
============================================================================
  - PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
  - WandB + Lightning: https://docs.wandb.ai/guides/integrations/lightning
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl

from src.models.ldm_model import LargeDataModel
from src.training.objectives import NextTokenPredictionLoss, ContextConditionalMaskedLoss
from src.training.lr_scheduler import get_cosine_schedule_with_warmup

import logging

logger = logging.getLogger(__name__)


class LDMPreTrainingModule(pl.LightningModule):
    """
    Lightning Module for LDM Self-Supervised Pre-Training.

    Pre-training Phase:
    - Objective: Masked Joint-Distribution Modeling (LimiX)
    - The model learns to reconstruct randomly masked transaction
      features from the visible context
    - No labels are needed — the data itself provides supervision
    - The result is a powerful encoder that understands transaction patterns

    How it works:
    1. Take a batch of transaction sequences
    2. Randomly mask some tokens (heterogeneous schedule)
    3. Feed the masked sequences through the Transformer
    4. Predict the original values at masked positions
    5. Compute MSE loss between predictions and ground truth

    Args:
        model: LargeDataModel instance (or will be created from config).
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

        # Masked objective (LimiX)
        self.masked_loss = ContextConditionalMaskedLoss(
            masking_prob=config.get("ssl_masking_prob", 0.15)
        )

        # Training config
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 100)
        self.max_steps = config.get("max_steps", 10000)

    def forward(self, batch):
        """Forward pass — encode transaction sequences."""
        return self.model.encode(
            batch["packed_keys"],
            batch["packed_values"],
            batch["packed_times"],
            is_causal=False,  # Bidirectional for masked modeling
        )

    def training_step(self, batch, batch_idx):
        """
        Single training step for pre-training.

        Process:
        1. Get the original embeddings (encode without masking)
        2. Generate a heterogeneous mask
        3. Apply mask to the input (replace masked tokens with noise)
        4. Encode the masked input
        5. Compute reconstruction loss at masked positions
        """
        keys = batch["packed_keys"]
        values = batch["packed_values"]
        times = batch["packed_times"]

        # Step 1: Get original (unmasked) embeddings as targets
        with torch.no_grad():
            original_embeddings = self.model.encode(
                keys, values, times, is_causal=False
            )

        # Step 2: Generate mask based on INPUT sequence length
        if keys.dim() == 2:
            B, S = keys.shape
        else:
            B, S = 1, keys.shape[0]

        input_mask = self.masked_loss.heterogeneous_mask_schedule(B, S, keys.device)

        # Step 3: Create masked input by adding noise to masked positions
        noisy_values = values.clone()
        if noisy_values.dim() == 1:
            flat_mask = input_mask.squeeze(0) if input_mask.dim() == 2 else input_mask
            if flat_mask.shape[0] > noisy_values.shape[0]:
                flat_mask = flat_mask[:noisy_values.shape[0]]
            elif flat_mask.shape[0] < noisy_values.shape[0]:
                flat_mask = F.pad(flat_mask, (0, noisy_values.shape[0] - flat_mask.shape[0]))
            noisy_values[flat_mask] = 0.0  # Zero out masked values
        else:
            noisy_values[input_mask] = 0.0

        # Step 4: Encode the masked input
        reconstructed = self.model.encode(
            keys, noisy_values, times, is_causal=False
        )

        # The mask for loss computation needs to match the output embedding shape!
        # If virtual tokens were added, we pad the mask with False at the beginning
        num_virtual = reconstructed.shape[1] - S
        if num_virtual > 0:
            if input_mask.dim() == 2:
                padding = torch.zeros((B, num_virtual), dtype=torch.bool, device=keys.device)
                loss_mask = torch.cat([padding, input_mask], dim=1)
            else:
                padding = torch.zeros((num_virtual,), dtype=torch.bool, device=keys.device)
                loss_mask = torch.cat([padding, input_mask], dim=0)
        else:
            loss_mask = input_mask

        # Step 5: Compute loss only at masked positions
        loss = self.masked_loss(reconstructed, original_embeddings, loss_mask)

        # Logging
        self.log("pretrain/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step — same as training but without gradient."""
        keys = batch["packed_keys"]
        values = batch["packed_values"]
        times = batch["packed_times"]

        original = self.model.encode(keys, values, times, is_causal=False)
        
        if keys.dim() == 2:
            B, S = keys.shape
        else:
            B, S = 1, keys.shape[0]
            
        input_mask = self.masked_loss.heterogeneous_mask_schedule(B, S, keys.device)

        noisy_values = values.clone()
        if noisy_values.dim() == 1:
            flat_mask = input_mask.squeeze(0) if input_mask.dim() == 2 else input_mask
            if flat_mask.shape[0] > noisy_values.shape[0]:
                flat_mask = flat_mask[:noisy_values.shape[0]]
            noisy_values[flat_mask[:noisy_values.shape[0]]] = 0.0
        else:
            noisy_values[input_mask] = 0.0

        reconstructed = self.model.encode(keys, noisy_values, times, is_causal=False)
        
        num_virtual = reconstructed.shape[1] - S
        if num_virtual > 0:
            if input_mask.dim() == 2:
                padding = torch.zeros((B, num_virtual), dtype=torch.bool, device=keys.device)
                loss_mask = torch.cat([padding, input_mask], dim=1)
            else:
                padding = torch.zeros((num_virtual,), dtype=torch.bool, device=keys.device)
                loss_mask = torch.cat([padding, input_mask], dim=0)
        else:
            loss_mask = input_mask
            
        loss = self.masked_loss(reconstructed, original, loss_mask)

        self.log("pretrain/val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Uses AdamW (Adam with decoupled weight decay) — the standard
        optimizer for Transformers. Weight decay acts as L2 regularization
        but is properly decoupled from the gradient update.

        The learning rate schedule uses warmup + cosine decay.
        """
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

    Fine-tuning Phase:
    - Objective: Binary classification (e.g., fraud detection)
    - Uses pre-trained Transformer encoder + DCNv2 fusion head
    - Combines transaction embeddings with tabular features
    - Handles severe class imbalance via weighted loss

    The fine-tuning process:
    1. Load pre-trained encoder weights
    2. Optionally freeze encoder (train only fusion head)
    3. Train on labeled data with class-weighted BCE loss
    4. Track AUC, F1, Precision, Recall

    Args:
        model: LargeDataModel instance (ideally pre-trained).
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

        # Training config
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 50)
        self.max_steps = config.get("max_steps", 5000)
        self.threshold = config.get("metric_threshold", 0.5)

        # =====================================================================
        # Class-Weighted Binary Cross-Entropy Loss
        # =====================================================================
        # Financial fraud is extremely rare (typically 0.1-2% of transactions).
        # Without class weighting, the model would learn to predict "not fraud"
        # for everything and achieve 98%+ accuracy — while catching ZERO fraud.
        #
        # pos_weight tells the loss to weight positive (fraud) examples more
        # heavily. A weight of 10 means each fraud example counts as 10
        # non-fraud examples during loss computation.
        # =====================================================================
        pos_weight = config.get("pos_weight", 10.0)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # Metric tracking
        self.validation_step_outputs = []

    def forward(self, batch):
        """Forward pass for classification."""
        return self.model(
            batch["packed_keys"],
            batch["packed_values"],
            batch["packed_times"],
            tabular_features=batch.get("tabular"),
        )

    def training_step(self, batch, batch_idx):
        """
        Single training step for fine-tuning.

        Computes the binary classification loss and logs metrics.
        """
        logits = self.forward(batch)
        labels = batch["labels"].float().view(-1, 1)

        # Move pos_weight to correct device
        self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)

        loss = self.loss_fn(logits, labels)

        # Compute training metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()
            accuracy = (preds == labels).float().mean()

        self.log("finetune/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("finetune/train_acc", accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step — collects predictions for epoch-level metrics.
        """
        logits = self.forward(batch)
        labels = batch["labels"].float().view(-1, 1)

        self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)
        loss = self.loss_fn(logits, labels)

        self.log("finetune/val_loss", loss, on_epoch=True, prog_bar=True)

        # Store for epoch-end metric computation
        self.validation_step_outputs.append({
            "logits": logits.detach().cpu(),
            "labels": labels.detach().cpu(),
        })

        return loss

    def on_validation_epoch_end(self):
        """
        Computes aggregate metrics over the full validation set.

        Individual batch metrics can be misleading for imbalanced data.
        Epoch-level metrics give a more accurate picture of model performance.
        """
        if not self.validation_step_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

        # AUC computation
        from src.evaluation.metrics import calculate_auc, calculate_f1_flexible
        auc = calculate_auc(all_logits, all_labels)
        f1 = calculate_f1_flexible(all_logits, all_labels, self.threshold)

        # Precision and Recall
        probs = torch.sigmoid(all_logits).view(-1)
        preds = (probs >= self.threshold).float()
        targets = all_labels.view(-1).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()

        self.log("finetune/val_auc", auc, prog_bar=True)
        self.log("finetune/val_f1", f1, prog_bar=True)
        self.log("finetune/val_precision", precision)
        self.log("finetune/val_recall", recall)

        # Clear stored outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step — same logic as validation for final evaluation."""
        logits = self.forward(batch)
        labels = batch["labels"].float().view(-1, 1)

        self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)
        loss = self.loss_fn(logits, labels)

        self.log("test/loss", loss)

        # Store for test_epoch_end
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
        self.test_step_outputs.append({
            "logits": logits.detach().cpu(),
            "labels": labels.detach().cpu(),
        })
        return loss

    def on_test_epoch_end(self):
        """Computes final test metrics."""
        if not hasattr(self, "test_step_outputs") or not self.test_step_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        from src.evaluation.metrics import calculate_auc, calculate_f1_flexible
        auc = calculate_auc(all_logits, all_labels)
        f1 = calculate_f1_flexible(all_logits, all_labels, self.threshold)

        self.log("test/auc", auc)
        self.log("test/f1", f1)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configures optimizer with differential learning rates.

        The encoder uses a LOWER learning rate (1/10) than the fusion head
        to preserve pre-trained representations while still allowing
        task-specific adaptation.
        """
        # Differential learning rates
        encoder_params = list(self.model.encoder.parameters())
        head_params = list(self.model.fusion_head.parameters())

        param_groups = [
            {"params": encoder_params, "lr": self.learning_rate * 0.1},
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

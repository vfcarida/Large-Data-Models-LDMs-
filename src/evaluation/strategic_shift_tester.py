"""
Out-of-Distribution (OOD) and Strategic Manipulation Tester.

============================================================================
PURPOSE
============================================================================
Evaluates the resilience of Large Data Models (LDMs) under both natural
distribution drift and intentional adversarial evasion (strategic shift):

1. Random Shift (Natural Drift):
   - Injects Gaussian perturbations into continuous transaction amounts to
     simulate macroeconomic drift (inflation, seasonal spending shifts).

2. FGSM White-Box Adversarial Shift (Strategic Evasion):
   - Uses the Fast Gradient Sign Method (Goodfellow et al., ICLR 2015) to
     compute worst-case input perturbations along the gradient of the binary
     cross-entropy loss with respect to continuous transaction values.
   - Simulates a strategic adversary structuring transaction amounts to
     evade fraud detection thresholds.

3. Parameter-Efficient LoRA Adaptation:
   - Injects Low-Rank Adaptation (Hu et al., ICLR 2022) adapters into
     attention projections to allow rapid emergency updates (~1% trainable
     parameters) without catastrophic forgetting.

============================================================================
REFERENCES
============================================================================
  - Goodfellow et al. (2015): "Explaining and Harnessing Adversarial Examples"
  - Hu et al. (2022): "LoRA: Low-Rank Adaptation of Large Language Models"
============================================================================
"""

from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrategicShiftTester:
    """
    Quantifies model robustness against natural drift and strategic evasion.

    Args:
        noise_std: Standard deviation of random Gaussian drift (default: `0.1`).
        adversarial_epsilon: FGSM perturbation magnitude (default: `0.05`).
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        adversarial_epsilon: float = 0.05,
    ):
        self.noise_std = noise_std
        self.epsilon = adversarial_epsilon

    def apply_random_shift(
        self,
        continuous_features: torch.Tensor,
    ) -> torch.Tensor:
        """Applies Gaussian noise `N(0, noise_std^2)` to continuous features."""
        noise = torch.randn_like(continuous_features) * self.noise_std
        return (continuous_features + noise).detach()

    def apply_gradient_adversarial_shift(
        self,
        model: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Applies FGSM white-box adversarial perturbation to continuous transaction values:
            x_adv = x + epsilon * sign(grad_x L(model(x), y))
        """
        packed_keys, packed_values, packed_times = inputs

        perturbed_input = packed_values.clone().detach().requires_grad_(True)

        logits = model(
            packed_keys,
            perturbed_input,
            packed_times,
            tabular_features=tabular,
            key_padding_mask=key_padding_mask,
        )
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), targets.float().view(-1)
        )

        model.zero_grad()
        loss.backward()

        data_grad = perturbed_input.grad.data
        perturbed_values = perturbed_input + self.epsilon * data_grad.sign()

        if key_padding_mask is not None:
            perturbed_values = perturbed_values.masked_fill(key_padding_mask, 0.0)

        return (packed_keys, perturbed_values.detach(), packed_times)

    def evaluate_robustness(
        self,
        model: nn.Module,
        clean_inputs: Tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Computes ROC-AUC under clean, random-drift, and FGSM adversarial conditions.
        """
        from src.evaluation.metrics import calculate_auc

        model.eval()
        keys, values, times = clean_inputs

        with torch.no_grad():
            clean_logits = model(
                keys,
                values,
                times,
                tabular_features=tabular,
                key_padding_mask=key_padding_mask,
            )
        clean_auc = calculate_auc(clean_logits, targets)

        shifted_values = self.apply_random_shift(values)
        if key_padding_mask is not None:
            shifted_values = shifted_values.masked_fill(key_padding_mask, 0.0)
        with torch.no_grad():
            shifted_logits = model(
                keys,
                shifted_values,
                times,
                tabular_features=tabular,
                key_padding_mask=key_padding_mask,
            )
        shifted_auc = calculate_auc(shifted_logits, targets)

        adversarial_inputs = self.apply_gradient_adversarial_shift(
            model,
            (keys, values, times),
            targets,
            tabular=tabular,
            key_padding_mask=key_padding_mask,
        )
        with torch.no_grad():
            adv_logits = model(
                adversarial_inputs[0],
                adversarial_inputs[1],
                adversarial_inputs[2],
                tabular_features=tabular,
                key_padding_mask=key_padding_mask,
            )
        adv_auc = calculate_auc(adv_logits, targets)

        results = {
            "clean_auc": float(clean_auc),
            "random_shift_auc": float(shifted_auc),
            "fgsm_auc": float(adv_auc),
            "random_degradation": float(clean_auc - shifted_auc),
            "fgsm_degradation": float(clean_auc - adv_auc),
        }

        logger.info(f"Robustness evaluation results: {results}")
        return results


def inject_lora_adaptation(
    transformer_model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """
    Injects LoRA (Low-Rank Adaptation) into Transformer attention projections.
    """
    if not PEFT_AVAILABLE:
        logger.warning(
            "'peft' package not found. Returning base model. "
            "Install via: pip install peft"
        )
        return transformer_model

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        target_modules=["in_proj_weight", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(transformer_model, config)

    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"LoRA injected: {trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )

    return peft_model

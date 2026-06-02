"""
Out-of-Distribution (OOD) and Strategic Manipulation Tester.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
A critical challenge in financial AI is that malicious actors ACTIVELY
try to evade detection. Unlike natural distribution shift (e.g., COVID
changing spending patterns), strategic manipulation involves INTENTIONAL
behavior changes designed to fool the model.

Examples of Strategic Manipulation:
  - A fraudster splits large wire transfers into many small ones
    (structuring/smurfing) to stay below detection thresholds
  - A defaulting borrower temporarily inflates their income metrics
  - An account takeover starts with small legitimate-looking purchases
    before making large fraudulent ones

This module tests the model's ROBUSTNESS against such attacks using:

1. Random Shift: Adds Gaussian noise to continuous features.
   Simulates natural distribution drift (e.g., inflation, seasonal changes).

2. FGSM Attack: Uses the model's own gradients to find the OPTIMAL
   perturbation that maximizes evasion probability. Simulates a
   sophisticated adversary with knowledge of the model.

3. LoRA Adaptation: Parameter-efficient fine-tuning that allows rapid
   model updates to counter new attack patterns without full retraining.

============================================================================
KEY CONCEPTS
============================================================================
FGSM (Fast Gradient Sign Method):
  - Computes the gradient of the loss with respect to INPUT features
  - Perturbs inputs in the direction that MAXIMIZES the loss
  - This is the WORST CASE scenario for the model
  - If the model is robust to FGSM, it's robust to simple attacks

LoRA (Low-Rank Adaptation):
  - Injects small trainable matrices into frozen model weights
  - Allows rapid adaptation with ~1% of original parameters
  - Perfect for "emergency patches" when new attack patterns emerge

============================================================================
REFERENCES
============================================================================
  - Goodfellow et al. (2015): "Explaining and Harnessing Adversarial Examples"
  - Hu et al. (2022): "LoRA: Low-Rank Adaptation of Large Language Models"
  - LDM² (2024): Dynamic reinforcement learning under uncertainty
============================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import logging

# PeFT library for LoRA injection
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrategicShiftTester:
    """
    Quantifies model robustness against strategic manipulation.

    Tests two types of distribution shift:

    1. RANDOM SHIFT (simulates natural drift):
       - Adds Gaussian noise to continuous features
       - Models scenarios like: inflation, seasonal changes, new products

    2. ADVERSARIAL SHIFT (simulates intentional evasion):
       - Uses FGSM to find worst-case perturbations
       - Models scenarios like: fraud structuring, identity manipulation

    The tester reports the DEGRADATION in model performance:
    degradation = AUC_clean - AUC_shifted

    A robust model should have small degradation (< 5%).

    Args:
        noise_std: Standard deviation of random noise (default: 0.1).
        adversarial_epsilon: FGSM perturbation magnitude (default: 0.05).
            Larger epsilon = stronger attack but less realistic.

    Example:
        >>> tester = StrategicShiftTester(noise_std=0.1, adversarial_epsilon=0.05)
        >>> shifted_values = tester.apply_random_shift(original_values)
        >>> adversarial_inputs = tester.apply_gradient_adversarial_shift(
        ...     model, (keys, values, times), targets
        ... )
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
        """
        Applies Gaussian noise to continuous features.

        This simulates NATURAL distribution shift — gradual changes in
        the data distribution that occur over time (e.g., inflation
        increasing average transaction amounts by 10%).

        Mathematical formulation:
            x_shifted = x_original + ε, where ε ~ N(0, σ²)

        The .detach() call ensures the noisy tensor is disconnected
        from the computation graph, preventing memory leaks in
        repeated evaluation loops.

        Args:
            continuous_features: Original continuous values [B, S] or [N].

        Returns:
            Shifted features with same shape and dtype.
        """
        noise = torch.randn_like(continuous_features) * self.noise_std
        return (continuous_features + noise).detach()

    def apply_gradient_adversarial_shift(
        self,
        model: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Applies FGSM adversarial perturbation to continuous features.

        FGSM (Fast Gradient Sign Method):
        1. Forward pass through the model
        2. Compute the loss with respect to the true labels
        3. Backpropagate to get gradients WITH RESPECT TO THE INPUT
           (not the model weights — this is the key difference)
        4. Perturb the input in the SIGN direction of the gradient
        5. This maximizes the loss, creating the worst-case input

        Mathematical formulation:
            x_adversarial = x + ε · sign(∇_x L(x, y))

        This is a WHITE-BOX attack: the attacker has full knowledge of
        the model. If the model is robust to this, it's likely robust
        to simpler black-box attacks.

        Args:
            model: The model to attack (must be in eval mode).
            inputs: Tuple of (packed_keys, packed_values, packed_times).
            targets: Ground truth labels.

        Returns:
            Tuple with perturbed continuous values (keys and times unchanged).
        """
        packed_keys, packed_values, packed_times = inputs

        # Clone and enable gradient tracking on the continuous values
        # We need gradients w.r.t. the INPUT, not the model parameters
        packed_values = packed_values.clone().detach().requires_grad_(True)

        # Forward pass
        logits = model(packed_keys, packed_values, packed_times)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), targets.float().view(-1)
        )

        # Backward pass to get input gradients
        model.zero_grad()
        loss.backward()

        # FGSM perturbation: move in the direction that maximizes loss
        data_grad = packed_values.grad.data
        perturbed_values = packed_values + self.epsilon * data_grad.sign()

        # Detach to prevent memory leaks
        return (packed_keys, perturbed_values.detach(), packed_times)

    def evaluate_robustness(
        self,
        model: nn.Module,
        clean_inputs: Tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive robustness evaluation.

        Computes metrics under three conditions:
        1. Clean (no perturbation) — baseline performance
        2. Random shift — natural drift resilience
        3. FGSM attack — adversarial robustness

        Args:
            model: Model to evaluate.
            clean_inputs: (keys, values, times) tuple.
            targets: Ground truth labels.
            tabular: Optional tabular features.

        Returns:
            Dictionary with AUC under each condition and degradation.
        """
        from src.evaluation.metrics import calculate_auc

        model.eval()
        keys, values, times = clean_inputs

        # 1. Clean evaluation
        with torch.no_grad():
            clean_logits = model(keys, values, times, tabular_features=tabular)
        clean_auc = calculate_auc(clean_logits, targets)

        # 2. Random shift evaluation
        shifted_values = self.apply_random_shift(values)
        with torch.no_grad():
            shifted_logits = model(keys, shifted_values, times, tabular_features=tabular)
        shifted_auc = calculate_auc(shifted_logits, targets)

        # 3. FGSM evaluation (requires gradients)
        adversarial_inputs = self.apply_gradient_adversarial_shift(
            model, (keys, values, times), targets
        )
        with torch.no_grad():
            adv_logits = model(
                adversarial_inputs[0], adversarial_inputs[1], adversarial_inputs[2],
                tabular_features=tabular,
            )
        adv_auc = calculate_auc(adv_logits, targets)

        results = {
            "clean_auc": clean_auc,
            "random_shift_auc": shifted_auc,
            "fgsm_auc": adv_auc,
            "random_degradation": clean_auc - shifted_auc,
            "fgsm_degradation": clean_auc - adv_auc,
        }

        logger.info(f"Robustness results: {results}")
        return results


def inject_lora_adaptation(
    transformer_model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """
    Injects LoRA (Low-Rank Adaptation) into the Transformer's attention layers.

    LoRA adds small trainable matrices alongside the frozen original weights:
        W_new = W_frozen + (α/r) · B · A

    Where:
        W_frozen: Original weight matrix (frozen, not updated)
        A: Low-rank matrix (r × d), initialized randomly
        B: Low-rank matrix (d × r), initialized to zero
        r: Rank (typically 4-16, much smaller than d)
        α: Scaling factor

    Benefits:
    - Only ~1% of parameters are trainable
    - No catastrophic forgetting of pre-trained knowledge
    - Can be merged back into the original weights for inference
    - Multiple LoRA adapters can be swapped for different tasks

    Use case in LDMs:
    When a new type of fraud is detected post-deployment, LoRA allows
    rapid model updates (hours instead of days) without full retraining.

    Args:
        transformer_model: The model to inject LoRA into.
        rank: Rank of the low-rank matrices (lower = fewer params).
        alpha: Scaling factor (higher = stronger adaptation signal).

    Returns:
        Model with LoRA layers injected.
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

    # Freeze everything except LoRA parameters
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # Log parameter counts
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"LoRA injected: {trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )

    return peft_model

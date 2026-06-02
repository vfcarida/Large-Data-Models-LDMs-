"""
Learning Rate Scheduler — Warmup + Cosine Annealing.

============================================================================
PURPOSE (PROPÓSITO)
============================================================================
The learning rate schedule is one of the most critical hyperparameters
for training Transformers. Using a constant learning rate leads to:
  - Unstable training in early epochs (gradients are noisy)
  - Suboptimal convergence in later epochs (LR too high for fine-tuning)

The WARMUP + COSINE ANNEALING schedule is the modern standard:

Phase 1 — WARMUP (linear):
  LR increases linearly from 0 to max_lr over the first N steps.
  This gives the model time to build reasonable gradients before
  making large parameter updates.

Phase 2 — COSINE DECAY:
  LR decreases following a cosine curve from max_lr to min_lr.
  The smooth decrease allows the model to progressively "fine-tune"
  its weights with smaller and smaller updates.

Visual representation:
  LR ▲
     │     ╱‾‾‾╲
     │    ╱      ╲
     │   ╱        ╲
     │  ╱          ╲
     │ ╱            ╲___
     │╱
     └──────────────────► Steps
     warmup   cosine decay

============================================================================
REFERENCES
============================================================================
  - Loshchilov & Hutter (2017): "SGDR: Stochastic Gradient Descent
    with Warm Restarts" (introduced cosine annealing)
  - Vaswani et al. (2017): Used warmup in the original Transformer
  - Chinchilla (2022): Demonstrated optimal LR schedules for LLMs
============================================================================
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        num_warmup_steps: Number of steps for the linear warmup phase.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of the peak LR.
                      E.g., 0.1 means the LR decays to 10% of max.
        num_cycles: Number of cosine cycles. 0.5 = half cosine (standard).

    Returns:
        LambdaLR scheduler that can be stepped after each optimizer step.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=100, num_training_steps=1000
        ... )
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()
    """

    def lr_lambda(current_step: int) -> float:
        """
        Computes the learning rate multiplier for the current step.

        Returns a value in [min_lr_ratio, 1.0] that is multiplied with
        the base learning rate set in the optimizer.
        """
        # Phase 1: Linear Warmup
        if current_step < num_warmup_steps:
            # Linear increase from 0 to 1.0
            return float(current_step) / float(max(1, num_warmup_steps))

        # Phase 2: Cosine Decay
        # Map current step to progress [0, 1] within the decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # Cosine decay from 1.0 to min_lr_ratio
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        # Scale to [min_lr_ratio, 1.0]
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value)

    return LambdaLR(optimizer, lr_lambda)

"""
Out-of-Distribution (OOD) and Strategic Manipulation Tester.

Incorporates foundations of Dynamic Memory ($LDM^2$) and Kernel In-Context Learning (KernelICL)
by simulating adversarial agents that actively perturb transactional patterns 
(Strategic Shift) to evade the model. Also includes PeFT (LoRA) infrastructure 
to allow emergency model adaptation without catastrophic retraining.
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class StrategicShiftTester:
    """
    Quantifies OOD performance degradation caused by Strategic Manipulation.
    
    Simulates malicious users altering their behaviors (e.g., suddenly dropping 
    expenditure volumes, fragmenting wire transfers) to evade detection logic 
    trained on static distributions (like Adult or Bank Marketing datasets).
    """
    def __init__(self, noise_std: float = 0.1, adversarial_epsilon: float = 0.05):
        self.noise_std = noise_std
        self.epsilon = adversarial_epsilon

    def apply_random_shift(self, continuous_features: torch.Tensor) -> torch.Tensor:
        """
        Distorts original features by applying a noise tensor (Random Data Shift).
        """
        noise = torch.randn_like(continuous_features) * self.noise_std
        # Detach ensures disconnected tensors do not cause memory leaks in repeated loops
        return (continuous_features + noise).detach()

    def apply_gradient_adversarial_shift(self, 
                                         model: nn.Module, 
                                         inputs: Tuple[torch.Tensor, ...], 
                                         targets: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Applies a simplified Fast Gradient Sign Method (FGSM) to emulate tactical 
        adversarial attacks on continuous tabular variables.
        
        Requires the Continuous Value Tensor to have `requires_grad=True`.
        """
        packed_keys, packed_values, packed_times = inputs
        
        # Isolate tensors and ensure graph release
        packed_values = packed_values.clone().detach().requires_grad_(True)
        
        # Forward test pass
        logits = model(packed_keys, packed_values, packed_times)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), targets.float().view(-1))
        
        model.zero_grad()
        loss.backward()
        
        # Gradient-guided adversarial perturbation
        data_grad = packed_values.grad.data
        perturbed_values = packed_values + self.epsilon * data_grad.sign()
        
        # Return the input tuple with perturbed continuous values and cleansed autograd memory
        return (packed_keys, perturbed_values.detach(), packed_times)

def inject_lora_adaptation(transformer_model: nn.Module, rank: int = 8, alpha: int = 16) -> nn.Module:
    """
    Parameter-Efficient Fine-Tuning (PeFT) via LoRA injections.
    
    Injects low-rank matrices into the Self-Attention of the Transformer Encoder.
    This acts as a defense mechanism against post-deployment distribution shifts,
    allowing rapid, targeted updates for new analytical surfaces while keeping 
    the multi-billion parameter structural backbone frozen.
    """
    if not PEFT_AVAILABLE:
        print("[WARNING] 'peft' package not found. Returning base model. Install via: pip install peft")
        return transformer_model

    # Generic LoRA configuration adaptable for dense linear matrices of Custom Transformers
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        # In PyTorch nn.TransformerEncoderLayer, projections are `in_proj_weight` and `out_proj`. 
        # For HuggingFace, it would be target_modules=['q_proj', 'v_proj']
        target_modules=["in_proj_weight", "out_proj"], 
        lora_dropout=0.05,
        bias="none",
    )
    
    peft_model = get_peft_model(transformer_model, config)
    
    # Freeze the rest of the structures (e.g., Embeddings, FeedForwards, and DCNv2) 
    # to enforce surgical adaptation
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            
    return peft_model

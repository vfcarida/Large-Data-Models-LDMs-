r"""
Parameter-Efficient Low-Rank Adaptation (`LoRA` / `DoRA`) & Blockwise 4-Bit NormalFloat (`NF4`) Quantization.

============================================================================
RESEARCH & ARCHITECTURAL ANCHORS
============================================================================
Multi-tenant payment processors and regional banking silos need to adapt a
shared Foundation Large Data Model (`LargeDataModel`) to local fraud patterns
without duplicating full FP32 weights per tenant.

Synthesizes three frontier parameter-efficiency & compression paradigms:
  1. Low-Rank Adaptation (`LoRA`, Hu et al., ICLR 2022, arXiv:2106.09685):
     - Freezes pre-trained weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
       and injects trainable rank-$r$ matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$,
       $B \in \mathbb{R}^{d_{\text{out}} \times r}$ with $B=0$ at initialization.
  2. Weight-Decomposed Low-Rank Adaptation (`DoRA`, Liu et al., ICML 2024 Oral,
     "DoRA: Weight-Decomposed Low-Rank Adaptation", arXiv:2402.09353):
     - Decomposes adapted weights into a learnable row-wise magnitude vector
       $m \in \mathbb{R}^{d_{\text{out}} \times 1}$ and normalized directional matrix:
       $$W_{\text{DoRA}} = m \cdot \frac{W_0 + \frac{\alpha}{r} B A}{\|W_0 + \frac{\alpha}{r} B A\|_{\text{row}}}$$
     - Supports exact zero-latency folding (`merge_weights`) into a standard
       `nn.Linear` layer after fine-tuning.
  3. 4-Bit NormalFloat Quantization (`NF4`, Dettmers et al., NeurIPS 2023,
     "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314):
     - Quantizes blockwise-normalized weights ($B_{\text{block}} = 64$) onto the
       16-level information-theoretically optimal standard normal quantiles,
       packing two 4-bit indices per `uint8` byte for ~`3.5x` weight compression.
============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# Exact 16-level 4-bit NormalFloat (NF4) quantile codebook in [-1.0, 1.0]
# (Dettmers et al., NeurIPS 2023 QLoRA)
NF4_CODEBOOK = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


class LoRALinear(nn.Module):
    r"""
    Low-Rank (`LoRA`) and Weight-Decomposed (`DoRA`) adapter wrapper around `nn.Linear`.

    At initialization ($B = 0$, $m = \|W_0\|_{\text{row}}$), the forward pass is
    mathematically identical to the base `nn.Linear` layer. After fine-tuning,
    `merge_weights()` folds the adapter into a single `nn.Linear` with zero
    runtime latency overhead.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        use_dora: bool = True,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.use_dora = use_dora

        # Freeze base weights
        self.register_buffer("base_weight", base_linear.weight.detach().clone())
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        device = self.base_weight.device
        dtype = self.base_weight.dtype

        self.lora_A = nn.Parameter(
            torch.randn(self.rank, self.in_features, device=device, dtype=dtype) * 0.02
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        )

        if self.use_dora:
            init_mag = self.base_weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            self.dora_magnitude = nn.Parameter(init_mag.clone())
        else:
            self.register_parameter("dora_magnitude", None)

    @property
    def weight(self) -> torch.Tensor:
        r"""
        Computes the effective adapted weight matrix $W_{\text{eff}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$.
        """
        delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        combined = self.base_weight + delta_w
        if self.use_dora and self.dora_magnitude is not None:
            row_norm = combined.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            return self.dora_magnitude * (combined / row_norm)
        return combined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def merge_weights(self) -> nn.Linear:
        """
        Folds the learned `LoRA`/`DoRA` parameters into a standard `nn.Linear`
        module for zero-overhead inference.
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.base_weight.device,
            dtype=self.base_weight.dtype,
        )
        merged.weight.copy_(self.weight.detach())
        if self.bias is not None and merged.bias is not None:
            merged.bias.copy_(self.bias.detach())
        return merged


class BlockwiseNF4Linear(nn.Module):
    """
    4-Bit NormalFloat (`NF4`) quantized linear layer with blockwise absmax scaling
    and packed `uint8` storage (two 4-bit indices per byte).
    """

    def __init__(self, linear: nn.Linear, block_size: int = 64):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.block_size = max(8, int(block_size))

        w = linear.weight.detach().float()
        device = w.device
        codebook = NF4_CODEBOOK.to(device)
        self.register_buffer("codebook", codebook)

        numel = w.numel()
        pad_len = (self.block_size - (numel % self.block_size)) % self.block_size
        if pad_len > 0:
            w_flat = F.pad(w.reshape(-1), (0, pad_len), value=0.0)
        else:
            w_flat = w.reshape(-1)

        # Ensure even length for 2-per-byte nibble packing
        if w_flat.numel() % 2 != 0:
            w_flat = F.pad(w_flat, (0, 1), value=0.0)

        self.padded_numel = int(w_flat.numel())
        blocks = w_flat.view(-1, self.block_size)
        absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        normalized = (blocks / absmax).clamp(-1.0, 1.0)

        # Nearest-neighbor lookup in 16-level NF4 codebook
        diffs = (normalized.unsqueeze(-1) - codebook.view(1, 1, 16)).abs()
        indices = diffs.argmin(dim=-1).to(torch.uint8).reshape(-1)  # [padded_numel] in 0..15

        # Pack two 4-bit indices into a single uint8 byte: (high_nibble << 4) | low_nibble
        high_nibble = indices[0::2]
        low_nibble = indices[1::2]
        packed_bytes = ((high_nibble << 4) | (low_nibble & 0x0F)).to(torch.uint8)

        self.register_buffer("packed_nf4", packed_bytes)
        self.register_buffer("block_absmax", absmax.squeeze(-1).to(torch.float32))

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        high = (self.packed_nf4 >> 4) & 0x0F
        low = self.packed_nf4 & 0x0F
        unpacked_idx = torch.stack([high, low], dim=-1).reshape(-1).long()
        vals = self.codebook[unpacked_idx]  # [padded_numel]
        blocks = vals.view(-1, self.block_size) * self.block_absmax.unsqueeze(-1)
        orig_numel = self.out_features * self.in_features
        return blocks.reshape(-1)[:orig_numel].view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def inject_lora_adapters(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    use_dora: bool = True,
    freeze_non_lora: bool = True,
) -> Dict[str, Any]:
    """
    Replaces `nn.Linear` layers in `model` (excluding internal MultiheadAttention
    projection C++ fastpaths) with `LoRALinear` (`DoRA`) adapters and optionally
    freezes all non-adapter parameters.

    Returns:
        Summary dictionary with `total_params`, `trainable_params`, and `trainable_pct`.
    """
    if freeze_non_lora:
        for p in model.parameters():
            p.requires_grad = False

    def _replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                continue
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    LoRALinear(child, rank=rank, alpha=alpha, use_dora=use_dora),
                )
            else:
                _replace(child)

    _replace(model)
    total_p = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for name, b in model.named_buffers() if "base_weight" in name
    )
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total_p),
        "trainable_params": int(trainable_p),
        "trainable_pct": round(100.0 * trainable_p / max(total_p, 1), 2),
    }


def merge_lora_adapters(model: nn.Module) -> int:
    """
    Merges all `LoRALinear` modules in `model` back into standard `nn.Linear`
    layers in-place. Returns the number of merged layers.
    """
    merged_count = 0

    def _merge(module: nn.Module) -> None:
        nonlocal merged_count
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge_weights())
                merged_count += 1
            else:
                _merge(child)

    _merge(model)
    return merged_count


def quantize_ldm_nf4(
    model: nn.Module,
    block_size: int = 64,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Clones `model` and replaces `nn.Linear` layers with 4-bit NormalFloat
    (`BlockwiseNF4Linear`) modules.

    Returns:
        Tuple `(nf4_model, report_dict)` with `fp32_mb`, `nf4_mb`, and `compression_ratio`.
    """
    model.eval()

    def _bytes(m: nn.Module) -> int:
        p_b = sum(p.numel() * p.element_size() for p in m.parameters())
        b_b = sum(b.numel() * b.element_size() for b in m.buffers())
        return p_b + b_b

    fp32_bytes = _bytes(model)
    nf4_model = copy.deepcopy(model)

    fp32_linear_bytes = 0
    nf4_linear_bytes = 0

    def _quant(module: nn.Module) -> None:
        nonlocal fp32_linear_bytes, nf4_linear_bytes
        for name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                continue
            if isinstance(child, nn.Linear):
                fp32_linear_bytes += child.weight.numel() * child.weight.element_size()
                q_layer = BlockwiseNF4Linear(child, block_size=block_size)
                nf4_linear_bytes += (
                    q_layer.packed_nf4.numel() * q_layer.packed_nf4.element_size()
                    + q_layer.block_absmax.numel() * q_layer.block_absmax.element_size()
                )
                setattr(module, name, q_layer)
            else:
                _quant(child)

    _quant(nf4_model)
    nf4_model.eval()
    nf4_bytes = _bytes(nf4_model)

    fp32_mb = fp32_bytes / (1024.0 * 1024.0)
    nf4_mb = nf4_bytes / (1024.0 * 1024.0)
    ratio = fp32_mb / max(nf4_mb, 1e-8)
    linear_ratio = float(fp32_linear_bytes) / max(float(nf4_linear_bytes), 1.0)

    return nf4_model, {
        "fp32_mb": round(fp32_mb, 4),
        "nf4_mb": round(nf4_mb, 4),
        "compression_ratio": round(ratio, 3),
        "linear_compression_ratio": round(linear_ratio, 3),
    }

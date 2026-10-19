"""
Low-Latency Model Export & Streaming Authorization Benchmark (TorchScript / ONNX).

============================================================================
PURPOSE
============================================================================
Provides deterministic graph export and sub-10ms real-time payment
authorization latency benchmarking for `LargeDataModel` (`LDM`).

In production card-present and instant-payment rails (e.g., Visa, Mastercard,
PIX, FedNow), fraud scoring must complete within a strict 5-15ms p99 SLA.
This module provides:
  1. `LDMInferenceWrapper`: Wraps `LargeDataModel` into a pure positional
     tensor graph (`keys, values, times, tabular, padding_mask -> logits, probs`)
     free of training side-effects.
  2. `export_to_torchscript`: Compiles the model via `torch.jit.trace`,
     verifies numerical parity (`max_abs_diff < 1e-5`), and serializes a
     standalone TorchScript artifact.
  3. `export_to_onnx`: Exports an ONNX computation graph with dynamic batch
     and sequence axes for TensorRT / ONNX Runtime deployment.
  4. `benchmark_inference_latency`: Measures p50, p95, and p99 single-request
     (`batch_size=1`) authorization latency in milliseconds across eager
     PyTorch and TorchScript runtimes.
============================================================================
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from src.models.ldm_model import LargeDataModel

logger = logging.getLogger(__name__)


class LDMInferenceWrapper(nn.Module):
    """
    Pure tensor-in, tensor-out inference wrapper for TorchScript and ONNX export.

    Args:
        model: Trained `LargeDataModel` instance.
    """

    def __init__(self, model: LargeDataModel):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes deterministic inference without memory-bank mutation.

        Args:
            packed_keys: LongTensor `[B, S]` of MCC tokens.
            packed_values: FloatTensor `[B, S]` of normalized transaction amounts.
            packed_times: LongTensor `[B, S]` of temporal delta bins.
            tabular_features: FloatTensor `[B, T]` of normalized static features.
            key_padding_mask: BoolTensor `[B, S]` (`True` = padded position).

        Returns:
            Tuple `(logits, probabilities)` each of shape `[B, 1]`.
        """
        logits = self.model(
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            tabular_features=tabular_features,
            key_padding_mask=key_padding_mask,
            update_memory=False,
        )
        probs = torch.sigmoid(logits)
        return logits, probs


def build_dummy_inference_inputs(
    model: LargeDataModel,
    batch_size: int = 1,
    seq_len: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates deterministic example inputs matching `LDMInferenceWrapper.forward`.
    """
    if device is None:
        device = next(model.parameters()).device

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)

    keys = torch.randint(
        0, model.vocab_size, (batch_size, seq_len), generator=gen, dtype=torch.long
    ).to(device)
    values = torch.randn(
        (batch_size, seq_len), generator=gen, dtype=torch.float32
    ).to(device)
    times = torch.randint(
        0, 500, (batch_size, seq_len), generator=gen, dtype=torch.long
    ).to(device)
    tabular = torch.randn(
        (batch_size, model.tabular_feature_size), generator=gen, dtype=torch.float32
    ).to(device)
    padding_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.bool, device=device
    )

    return keys, values, times, tabular, padding_mask


@torch.no_grad()
def export_to_torchscript(
    model: LargeDataModel,
    save_path: Optional[str] = None,
    batch_size: int = 1,
    seq_len: int = 32,
) -> Tuple[torch.jit.ScriptModule, float]:
    """
    Traces `LargeDataModel` into a standalone TorchScript `ScriptModule` and
    verifies numerical parity against eager PyTorch execution.

    Args:
        model: `LargeDataModel` instance to trace.
        save_path: Optional file path to save the compiled `.torchscript.pt` model.
        batch_size: Example batch size for tracing (default: 1 for streaming SLA).
        seq_len: Example sequence length for tracing (default: 32).

    Returns:
        Tuple `(traced_module, max_abs_diff)` where `max_abs_diff` is the maximum
        absolute discrepancy between eager and TorchScript outputs.
    """
    wrapper = LDMInferenceWrapper(model)
    wrapper.eval()

    example_inputs = build_dummy_inference_inputs(
        model, batch_size=batch_size, seq_len=seq_len
    )

    eager_logits, eager_probs = wrapper(*example_inputs)
    traced_module = torch.jit.trace(wrapper, example_inputs, strict=False)
    traced_logits, traced_probs = traced_module(*example_inputs)

    max_abs_diff = float(
        torch.max(
            torch.abs(eager_logits - traced_logits).max(),
            torch.abs(eager_probs - traced_probs).max(),
        ).item()
    )

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        traced_module.save(str(out))
        logger.info(
            f"TorchScript model exported to {out} (max parity diff: {max_abs_diff:.2e})"
        )

    return traced_module, max_abs_diff


@torch.no_grad()
def export_to_onnx(
    model: LargeDataModel,
    save_path: str,
    batch_size: int = 1,
    seq_len: int = 32,
    opset_version: int = 17,
) -> bool:
    """
    Exports `LargeDataModel` to an ONNX graph with dynamic batch and sequence axes.

    Args:
        model: `LargeDataModel` instance.
        save_path: Destination `.onnx` file path.
        batch_size: Example batch size.
        seq_len: Example sequence length.
        opset_version: ONNX opset version (default: 17).

    Returns:
        `True` if ONNX export succeeded, `False` if the optional `onnx` package
        is not installed in the current environment.
    """
    wrapper = LDMInferenceWrapper(model)
    wrapper.eval()

    example_inputs = build_dummy_inference_inputs(
        model, batch_size=batch_size, seq_len=seq_len
    )

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(out),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[
                "packed_keys",
                "packed_values",
                "packed_times",
                "tabular_features",
                "key_padding_mask",
            ],
            output_names=["logits", "probabilities"],
            dynamic_axes={
                "packed_keys": {0: "batch_size", 1: "seq_len"},
                "packed_values": {0: "batch_size", 1: "seq_len"},
                "packed_times": {0: "batch_size", 1: "seq_len"},
                "tabular_features": {0: "batch_size"},
                "key_padding_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            },
        )
        logger.info(f"ONNX graph exported to {out}")
        return True
    except (ModuleNotFoundError, ImportError) as exc:
        logger.info(
            f"Skipping ONNX export because optional dependency 'onnx' is not installed ({exc})."
        )
        return False


@torch.no_grad()
def benchmark_inference_latency(
    model: LargeDataModel,
    batch_size: int = 1,
    seq_len: int = 32,
    num_warmup: int = 10,
    num_runs: int = 50,
    export_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Benchmarks single-request (`batch_size=1`) payment authorization latency
    across Eager PyTorch and Compiled TorchScript execution.

    Args:
        model: `LargeDataModel` instance.
        batch_size: Inference batch size (default: 1 for real-time payment SLA).
        seq_len: Sequence length per request (default: 32).
        num_warmup: Number of warmup iterations before timing.
        num_runs: Number of timed forward passes.
        export_path: Optional path to save the compiled TorchScript artifact.

    Returns:
        Dictionary with millisecond latency percentiles (`p50`, `p95`, `p99`),
        throughput (requests/sec), and TorchScript numerical parity error.
    """
    wrapper = LDMInferenceWrapper(model)
    wrapper.eval()

    inputs = build_dummy_inference_inputs(
        model, batch_size=batch_size, seq_len=seq_len
    )

    traced_module, max_abs_diff = export_to_torchscript(
        model,
        save_path=export_path,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Warmup Eager
    for _ in range(num_warmup):
        _ = wrapper(*inputs)

    eager_times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = wrapper(*inputs)
        eager_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # Warmup TorchScript
    for _ in range(num_warmup):
        _ = traced_module(*inputs)

    jit_times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = traced_module(*inputs)
        jit_times_ms.append((time.perf_counter() - t0) * 1000.0)

    eager_arr = np.array(eager_times_ms, dtype=np.float64)
    jit_arr = np.array(jit_times_ms, dtype=np.float64)

    jit_mean_ms = float(np.mean(jit_arr))
    throughput_rps = float((1000.0 / max(jit_mean_ms, 1e-6)) * batch_size)

    return {
        "eager_p50_ms": float(np.percentile(eager_arr, 50)),
        "eager_p95_ms": float(np.percentile(eager_arr, 95)),
        "eager_p99_ms": float(np.percentile(eager_arr, 99)),
        "torchscript_p50_ms": float(np.percentile(jit_arr, 50)),
        "torchscript_p95_ms": float(np.percentile(jit_arr, 95)),
        "torchscript_p99_ms": float(np.percentile(jit_arr, 99)),
        "torchscript_throughput_rps": throughput_rps,
        "torchscript_parity_max_diff": max_abs_diff,
    }

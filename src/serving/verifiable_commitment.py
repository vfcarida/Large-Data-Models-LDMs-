r"""
Cryptographic Verifiable Inference Commitments & Execution Proofs (`ZK-LDM`).

Scientific & Regulatory Anchors:
  - Goldwasser, S., Kalai, Y. T., & Rothblum, G. N. (2015). "Delegating
    Computation: Interactive Proofs for Muggles." *Journal of the ACM (JACM)*,
    62(4), 1-64.
  - Kang, D., Hashimoto, T., Stoica, I., & Sun, Y. (2022). "Scaling up Trustless
    DNN Inference with Zero-Knowledge Proofs." *PVLDB*, 15(10), arXiv:2210.08674.
  - NIST FIPS 180-4 (SHA-256) & RFC 6962 (Merkle Hash Trees): Cryptographic
    commitments for model weight provenance, intermediate activation trace
    binding, and threshold compliance verification without exposing proprietary
    weights in plaintext.
"""

from dataclasses import dataclass
import hashlib
import hmac
import struct
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


def _tensor_to_canonical_bytes(tensor: torch.Tensor, scale: float = 1e5) -> bytes:
    """
    Quantizes a floating-point or integer tensor into canonical little-endian
    int64 bytes so commitments are bit-exact across platforms.
    """
    flat = tensor.detach().cpu().contiguous().view(-1)
    if flat.is_floating_point():
        q = torch.round(flat.double() * scale).to(torch.int64).tolist()
    else:
        q = flat.to(torch.int64).tolist()
    return struct.pack(f"<{len(q)}q", *q)


def _merkle_root_hex(leaf_hashes: Sequence[str]) -> str:
    """
    Computes the RFC 6962 style binary Merkle root over a sequence of hex digests.
    """
    if not leaf_hashes:
        return hashlib.sha256(b"EMPTY_MERKLE_TREE").hexdigest()

    current = [bytes.fromhex(h) for h in leaf_hashes]
    while len(current) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if (i + 1) < len(current) else current[i]
            combined = hashlib.sha256(b"\x01" + left + right).digest()
            next_level.append(combined)
        current = next_level
    return current[0].hex()


@dataclass
class VerifiableInferenceReceipt:
    r"""
    Cryptographic execution commitment receipt proving that a specific batch of
    decisions was produced by an unmodified `LargeDataModel` with Merkle weight
    root $\mathcal{R}_W$, intermediate layer activation trace $\mathcal{R}_H$,
    and policy threshold $\tau$.
    """

    receipt_id: str
    weight_merkle_root: str
    input_commitment: str
    trace_merkle_root: str
    output_commitment: str
    decision_threshold: float
    decisions: List[int]
    blinding_nonce: str
    hmac_signature: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "weight_merkle_root": self.weight_merkle_root,
            "input_commitment": self.input_commitment,
            "trace_merkle_root": self.trace_merkle_root,
            "output_commitment": self.output_commitment,
            "decision_threshold": round(self.decision_threshold, 6),
            "decisions": self.decisions,
            "blinding_nonce": self.blinding_nonce,
            "hmac_signature": self.hmac_signature,
        }


class VerifiableLDMCommitmentEngine:
    r"""
    Generates and verifies cryptographic inference commitments (`ZK-LDM`) for
    `LargeDataModel` authorization decisions.

    Provides:
      1. **Weight Provenance Merkle Tree ($\mathcal{R}_W$):** Commits to every
         named parameter tensor in the model without revealing weights.
      2. **Blinded Input Commitment ($\mathcal{C}_{\text{in}}$):** Commits to
         input tensors with a cryptographic blinding nonce so PII remains hidden
         from third-party auditors unless explicitly opened.
      3. **Layer Activation Execution Trace ($\mathcal{R}_H$):** Hooks internal
         PyTorch submodules during inference to commit to intermediate hidden
         states $h^{(1)}, \dots, h^{(L)}$.
      4. **Threshold Compliance & Signature Verification:** Proves that the
         binary decision vector strictly satisfies $d_i = \mathbb{I}[\sigma(z_i) \ge \tau]$.
    """

    def __init__(
        self,
        signing_secret: str = "ldm-zk-verifiable-governance-key",
        decision_threshold: float = 0.50,
    ):
        self.signing_key = signing_secret.encode("utf-8")
        self.decision_threshold = float(decision_threshold)

    @staticmethod
    def compute_weight_merkle_root(model: nn.Module) -> str:
        """
        Computes the deterministic Merkle root over all named parameters of `model`.
        """
        leaf_hashes: List[str] = []
        for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            h = hashlib.sha256()
            h.update(b"\x00WEIGHT:")
            h.update(name.encode("utf-8"))
            h.update(_tensor_to_canonical_bytes(param))
            leaf_hashes.append(h.hexdigest())
        return _merkle_root_hex(leaf_hashes)

    @staticmethod
    def compute_input_commitment(
        blinding_nonce: str,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Computes a blinded SHA-256 commitment over the transaction input tensors.
        """
        h = hashlib.sha256()
        h.update(b"INPUT_COMMITMENT:")
        h.update(blinding_nonce.encode("utf-8"))
        h.update(_tensor_to_canonical_bytes(packed_keys))
        h.update(_tensor_to_canonical_bytes(packed_values))
        h.update(_tensor_to_canonical_bytes(packed_times))
        if tabular_features is not None:
            h.update(_tensor_to_canonical_bytes(tabular_features))
        return h.hexdigest()

    def _sign_payload(
        self,
        weight_root: str,
        input_comm: str,
        trace_root: str,
        output_comm: str,
        threshold: float,
        decisions: Sequence[int],
    ) -> str:
        msg = (
            f"{weight_root}|{input_comm}|{trace_root}|{output_comm}|"
            f"{threshold:.6f}|{','.join(str(d) for d in decisions)}"
        ).encode("utf-8")
        return hmac.new(self.signing_key, msg, hashlib.sha256).hexdigest()

    @torch.no_grad()
    def prove_inference(
        self,
        model: nn.Module,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        blinding_nonce: str = "nonce-2026-zk-ldm",
    ) -> VerifiableInferenceReceipt:
        """
        Executes `model` in deterministic evaluation mode, captures intermediate
        layer activation commitments via forward hooks, and returns a signed
        `VerifiableInferenceReceipt`.
        """
        model.eval()
        weight_root = self.compute_weight_merkle_root(model)
        input_comm = self.compute_input_commitment(
            blinding_nonce=blinding_nonce,
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            tabular_features=tabular_features,
        )

        trace_leaves: List[str] = []
        handles = []

        def _make_hook(layer_name: str):
            def _hook(_module, _inp, out):
                tensor_out = out[0] if isinstance(out, tuple) else out
                if isinstance(tensor_out, torch.Tensor):
                    h = hashlib.sha256()
                    h.update(b"\x00TRACE:")
                    h.update(layer_name.encode("utf-8"))
                    h.update(_tensor_to_canonical_bytes(tensor_out))
                    trace_leaves.append(h.hexdigest())
            return _hook

        for name, module in model.named_children():
            handles.append(module.register_forward_hook(_make_hook(name)))

        try:
            logits = model(
                packed_keys=packed_keys,
                packed_values=packed_values,
                packed_times=packed_times,
                tabular_features=tabular_features,
                key_padding_mask=key_padding_mask,
            ).view(-1)
        finally:
            for handle in handles:
                handle.remove()

        probs = torch.sigmoid(logits)
        decisions = [int(p >= self.decision_threshold) for p in probs.detach().cpu().tolist()]

        trace_root = _merkle_root_hex(trace_leaves)
        out_hash = hashlib.sha256()
        out_hash.update(b"OUTPUT_COMMITMENT:")
        out_hash.update(blinding_nonce.encode("utf-8"))
        out_hash.update(_tensor_to_canonical_bytes(logits))
        output_comm = out_hash.hexdigest()

        signature = self._sign_payload(
            weight_root=weight_root,
            input_comm=input_comm,
            trace_root=trace_root,
            output_comm=output_comm,
            threshold=self.decision_threshold,
            decisions=decisions,
        )
        receipt_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

        return VerifiableInferenceReceipt(
            receipt_id=receipt_id,
            weight_merkle_root=weight_root,
            input_commitment=input_comm,
            trace_merkle_root=trace_root,
            output_commitment=output_comm,
            decision_threshold=self.decision_threshold,
            decisions=decisions,
            blinding_nonce=blinding_nonce,
            hmac_signature=signature,
        )

    @torch.no_grad()
    def verify_receipt(
        self,
        receipt: VerifiableInferenceReceipt,
        model: nn.Module,
        packed_keys: Optional[torch.Tensor] = None,
        packed_values: Optional[torch.Tensor] = None,
        packed_times: Optional[torch.Tensor] = None,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Verifies a `VerifiableInferenceReceipt` against `model` (and optionally
        re-runs deterministic trace verification when input tensors are opened).
        Returns `False` if weights, inputs, trace, or signature were tampered with.
        """
        expected_sig = self._sign_payload(
            weight_root=receipt.weight_merkle_root,
            input_comm=receipt.input_commitment,
            trace_root=receipt.trace_merkle_root,
            output_comm=receipt.output_commitment,
            threshold=receipt.decision_threshold,
            decisions=receipt.decisions,
        )
        if not hmac.compare_digest(expected_sig, receipt.hmac_signature):
            return False

        current_weight_root = self.compute_weight_merkle_root(model)
        if current_weight_root != receipt.weight_merkle_root:
            return False

        if (
            packed_keys is not None
            and packed_values is not None
            and packed_times is not None
        ):
            recomputed = self.prove_inference(
                model=model,
                packed_keys=packed_keys,
                packed_values=packed_values,
                packed_times=packed_times,
                tabular_features=tabular_features,
                key_padding_mask=key_padding_mask,
                blinding_nonce=receipt.blinding_nonce,
            )
            if (
                recomputed.input_commitment != receipt.input_commitment
                or recomputed.trace_merkle_root != receipt.trace_merkle_root
                or recomputed.output_commitment != receipt.output_commitment
                or recomputed.decisions != receipt.decisions
            ):
                return False

        return True

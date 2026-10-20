r"""
Certified Partially Monotonic Neural Risk Head & SR 11-7 / Basel III Governance (`MonotonicLDM`).

============================================================================
RESEARCH & REGULATORY ANCHORS
============================================================================
Banking credit underwriting and regulatory risk committees (Federal Reserve
SR 11-7, Basel III Internal Ratings-Based / IRB models, ECOA) require strict,
provable partial monotonicity with respect to risk-ordered covariates:
  - Increasing verified income or credit bureau score ($s_j = -1$) must NEVER
    increase predicted default/fraud risk anywhere in the input space.
  - Increasing debt-to-income ratio, utilization, or velocity burst count
    ($s_j = +1$) must NEVER decrease predicted risk.

Synthesizes two foundational monotonic neural network architectures:
  1. Deep Lattice & Partial Monotonic Networks (You et al., NeurIPS 2017,
     "Deep Lattice Networks and Partial Monotonic Functions", arXiv:1709.06680)
  2. Constrained Monotonic Neural Networks (Runje & Shankaranarayana, ICML 2023,
     "Constrained Monotonic Neural Networks", arXiv:2205.11775):
     - Decomposes the input vector $x \in \mathbb{R}^D$ into an unconstrained
       subspace ($s_j = 0$) and a monotonic subspace ($s_j \in \{-1, +1\}$).
     - Parameterizes the first-layer monotonic weights as
       $W_1[:, j] = \text{softplus}(\tilde{W}_1[:, j]) \cdot s_j$ and all
       subsequent hidden-to-output weights on the monotonic pathway as strictly
       non-negative $W_k^+ = \text{softplus}(\tilde{W}_k)$ with monotone
       non-decreasing activations (`tanh` / `ReLU`), guaranteeing a `0.0000`
       monotonicity violation rate across $\mathbb{R}^D$.
============================================================================
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MonotonicityAuditReport:
    """
    Regulatory certification report produced by `MonotonicityCertificationAuditor`.
    """

    num_constrained_features: int
    total_counterfactual_pairs: int
    num_violations: int
    violation_rate: float
    max_violation_magnitude: float
    is_certified_monotonic: bool

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "num_constrained_features": self.num_constrained_features,
            "total_counterfactual_pairs": self.total_counterfactual_pairs,
            "num_violations": self.num_violations,
            "violation_rate": round(self.violation_rate, 8),
            "max_violation_magnitude": round(self.max_violation_magnitude, 8),
            "is_certified_monotonic": self.is_certified_monotonic,
        }


class MonotonicConstrainedHead(nn.Module):
    r"""
    Partially monotonic neural risk head over joint `[h_seq || x_tab]` features.

    Guarantees by construction that for every feature $j$ with sign
    $s_j \in \{-1, +1\}$, the partial derivative satisfies
    $s_j \cdot \frac{\partial f(x)}{\partial x_j} \ge 0$ everywhere in $\mathbb{R}^D$.

    Args:
        input_dim: Total input dimension (`hidden_size + tabular_feature_size`).
        monotonicity_signs: Sequence of integers in `{-1, 0, +1}` of length
            `input_dim`. If shorter than `input_dim`, aligned to the trailing
            `tabular_feature_size` dimensions (leaving leading sequence dimensions
            unconstrained `0`).
        hidden_dim: Hidden width of the monotonic & unconstrained pathways (default: `32`).
    """

    def __init__(
        self,
        input_dim: int,
        monotonicity_signs: Optional[Sequence[int]] = None,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if monotonicity_signs is None:
            # Default: last 2 tabular features are monotonic (+1, -1)
            signs = [0] * max(0, input_dim - 2)
            if input_dim >= 2:
                signs.extend([1, -1])
            elif input_dim == 1:
                signs.append(1)
        else:
            raw_signs = [int(max(-1, min(1, s))) for s in monotonicity_signs]
            if len(raw_signs) < input_dim:
                signs = [0] * (input_dim - len(raw_signs)) + raw_signs
            else:
                signs = raw_signs[:input_dim]

        sign_tensor = torch.tensor(signs, dtype=torch.float32)
        unconstrained_mask = (sign_tensor == 0.0).float()

        self.register_buffer("sign_vec", sign_tensor)
        self.register_buffer("unconstrained_mask", unconstrained_mask)

        # Unconstrained pathway (strictly masked to s_j == 0 coordinates)
        self.free_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Monotonic pathway: W1 has sign s_j, W2 >= 0, W_out >= 0
        self.mono_w1_raw = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.mono_b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.mono_w2_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.mono_b2 = nn.Parameter(torch.zeros(hidden_dim))
        self.mono_w_out_raw = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.mono_b_out = nn.Parameter(torch.zeros(1))

        # Direct linear skip on monotonic features with guaranteed sign s_j
        self.mono_skip_raw = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes partially monotonic risk logits `[B, 1]`.
        """
        # 1. Unconstrained branch sees ONLY coordinates where sign_vec == 0
        x_free = x * self.unconstrained_mask.unsqueeze(0)
        free_logit = self.free_mlp(x_free)

        # 2. Monotonic branch enforces exact sign s_j on constrained coordinates
        w1_pos = F.softplus(self.mono_w1_raw)
        w1_signed = w1_pos * self.sign_vec.unsqueeze(0)
        h1 = torch.tanh(F.linear(x, w1_signed, self.mono_b1))

        w2_pos = F.softplus(self.mono_w2_raw)
        h2 = F.relu(F.linear(h1, w2_pos, self.mono_b2))

        w_out_pos = F.softplus(self.mono_w_out_raw)
        mono_nl = F.linear(h2, w_out_pos, self.mono_b_out)

        w_skip_signed = F.softplus(self.mono_skip_raw) * self.sign_vec.unsqueeze(0)
        mono_skip = F.linear(x, w_skip_signed)

        return free_logit + mono_nl + mono_skip


class MonotonicityCertificationAuditor:
    """
    Empirically and mathematically audits a `MonotonicConstrainedHead` (or any
    callable risk model) across counterfactual grid perturbations along all
    constrained features (`SR 11-7` / `Basel III` compliance verification).
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    @torch.no_grad()
    def certify(
        self,
        head: MonotonicConstrainedHead,
        reference_inputs: torch.Tensor,
        delta_grid: Sequence[float] = (0.1, 0.5, 1.0, 2.5),
    ) -> MonotonicityAuditReport:
        r"""
        Perturbs each constrained coordinate $j$ ($s_j \in \{-1, +1\}$) by
        $+\Delta > 0$ across all samples in `reference_inputs` and checks that:
          - $f(x + \Delta e_j) - f(x) \ge -\text{tol}$ when $s_j = +1$, and
          - $f(x + \Delta e_j) - f(x) \le +\text{tol}$ when $s_j = -1$.
        """
        head.eval()
        base_logits = head(reference_inputs).view(-1)
        signs = head.sign_vec.detach().cpu().tolist()

        constrained_indices = [j for j, s in enumerate(signs) if abs(s) > 0.5]
        total_pairs = 0
        violations = 0
        max_violation = 0.0

        for j in constrained_indices:
            s_j = float(signs[j])
            for delta in delta_grid:
                if delta <= 0:
                    continue
                x_pert = reference_inputs.clone()
                x_pert[:, j] = x_pert[:, j] + float(delta)
                pert_logits = head(x_pert).view(-1)

                # Signed change: s_j * (f(x + delta) - f(x)) must be >= 0
                signed_diff = s_j * (pert_logits - base_logits)
                viol_mask = signed_diff < -self.tolerance
                num_v = int(viol_mask.sum().item())
                violations += num_v
                total_pairs += int(signed_diff.numel())

                if num_v > 0:
                    worst = float((-signed_diff[viol_mask]).max().item())
                    max_violation = max(max_violation, worst)

        viol_rate = float(violations) / float(max(total_pairs, 1))
        return MonotonicityAuditReport(
            num_constrained_features=len(constrained_indices),
            total_counterfactual_pairs=total_pairs,
            num_violations=violations,
            violation_rate=viol_rate,
            max_violation_magnitude=max_violation,
            is_certified_monotonic=(violations == 0),
        )

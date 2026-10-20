r"""
Federated Cross-Institution Training with Rényi Differential Privacy (`DP-FedAvg`).

Scientific & Regulatory Anchors:
  - McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
    "Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)."
    *AISTATS 2017*, arXiv:1602.05629.
  - Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K.,
    & Zhang, L. (2016). "Deep Learning with Differential Privacy (DP-SGD)."
    *ACM CCS 2016*, arXiv:1607.00133.
  - Mironov, I. (2017). "Rényi Differential Privacy (RDP)." *IEEE 30th Computer
    Security Foundations Symposium (CSF 2017)*, arXiv:1702.07476.

Enables inter-bank consortiums and payment networks to collaboratively train or
adapt `LargeDataModel` weights (`LoRA` / `DoRA` adapters or full backbones)
without sharing raw cardholder PII across institutional boundaries, while
maintaining formal $(\epsilon, \delta)$-Differential Privacy guarantees under
GDPR, LGPD, and GLBA.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


DEFAULT_RDP_ORDERS: Tuple[float, ...] = (
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
)


@dataclass
class FederatedRoundReport:
    r"""
    Telemetry and formal $(\epsilon, \delta)$-DP accounting report for a single
    `DP-FedAvg` communication round.
    """

    round_index: int
    num_clients: int
    mean_unclipped_l2_norm: float
    mean_clipped_l2_norm: float
    clip_norm_c: float
    noise_multiplier: float
    cumulative_epsilon: float
    target_delta: float
    optimal_rdp_order: float
    aggregated_update_l2_norm: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "round_index": self.round_index,
            "num_clients": self.num_clients,
            "mean_unclipped_l2_norm": round(self.mean_unclipped_l2_norm, 6),
            "mean_clipped_l2_norm": round(self.mean_clipped_l2_norm, 6),
            "clip_norm_c": round(self.clip_norm_c, 6),
            "noise_multiplier": round(self.noise_multiplier, 6),
            "cumulative_epsilon": round(self.cumulative_epsilon, 6),
            "target_delta": self.target_delta,
            "optimal_rdp_order": self.optimal_rdp_order,
            "aggregated_update_l2_norm": round(self.aggregated_update_l2_norm, 6),
        }


class RdpPrivacyAccountant:
    r"""
    Analytical Rényi Differential Privacy (`RDP`) accountant (Mironov, *IEEE CSF 2017*)
    with tight conversion to $(\epsilon, \delta)$-Differential Privacy.

    For the Gaussian mechanism with noise multiplier $\sigma_{\text{dp}}$ and
    client sampling rate $q \in (0, 1]$, accumulates Rényi divergence across a
    spectrum of orders $\alpha > 1$ and converts to $(\epsilon, \delta)$-DP via:
    $$\epsilon(\delta) = \min_{\alpha > 1} \left\{ \text{RDP}(\alpha) + \frac{\ln(1 / \delta)}{\alpha - 1} \right\}$$
    """

    def __init__(
        self,
        orders: Sequence[float] = DEFAULT_RDP_ORDERS,
        target_delta: float = 1e-5,
    ):
        self.orders = tuple(float(a) for a in orders if a > 1.0)
        self.target_delta = float(target_delta)
        self.cumulative_rdp: List[float] = [0.0 for _ in self.orders]
        self.steps_recorded: int = 0

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        r"""
        Records one differentially private aggregation round and returns the
        updated `(epsilon, optimal_alpha)` pair for `self.target_delta`.
        """
        sigma = max(float(noise_multiplier), 1e-6)
        q = min(max(float(sample_rate), 1e-6), 1.0)

        for idx, alpha in enumerate(self.orders):
            base_rdp = alpha / (2.0 * sigma * sigma)
            # Subsampled Gaussian amplification factor
            if q < 1.0:
                amplified_rdp = min(base_rdp, (q * q) * alpha / (sigma * sigma))
            else:
                amplified_rdp = base_rdp
            self.cumulative_rdp[idx] += amplified_rdp

        self.steps_recorded += 1
        return self.get_privacy_spent(self.target_delta)

    def get_privacy_spent(
        self, target_delta: Optional[float] = None
    ) -> Tuple[float, float]:
        r"""
        Computes the tightest $(\epsilon, \alpha^*)$ for the given `target_delta`.
        """
        delta = float(target_delta if target_delta is not None else self.target_delta)
        delta = min(max(delta, 1e-15), 1.0 - 1e-9)
        log_inv_delta = math.log(1.0 / delta)

        best_eps = float("inf")
        best_alpha = self.orders[0]
        for alpha, rdp_val in zip(self.orders, self.cumulative_rdp):
            eps = rdp_val + log_inv_delta / (alpha - 1.0)
            if eps < best_eps:
                best_eps = eps
                best_alpha = alpha

        return best_eps, best_alpha


class DifferentiallyPrivateFedCoordinator:
    r"""
    Coordinator for Differentially Private Federated Averaging (`DP-FedAvg`)
    across institutional clients.

    Workflow per round:
      1. Each client $k \in \{1, \dots, M\}$ computes local parameter delta
         $\Delta_k = \theta_k - \theta_{\text{global}}$ on its private ledger.
      2. The coordinator clips each client's update to maximum L2 sensitivity $C$:
         $$\tilde{\Delta}_k = \Delta_k \cdot \min\left(1, \frac{C}{\|\Delta_k\|_2}\right)$$
      3. Adds calibrated Gaussian noise $\mathcal{N}(0, (\sigma_{\text{dp}} C)^2 I)$ to the sum
         and divides by $M$:
         $$\bar{\Delta} = \frac{1}{M} \left( \sum_{k=1}^M \tilde{\Delta}_k + \mathcal{N}(0, \sigma_{\text{dp}}^2 C^2 I) \right)$$
      4. Updates $\theta_{\text{global}} \leftarrow \theta_{\text{global}} + \eta_{\text{server}} \bar{\Delta}$
         and logs cumulative $(\epsilon, \delta)$-DP expenditure via `RdpPrivacyAccountant`.
    """

    def __init__(
        self,
        global_model: nn.Module,
        clip_norm_c: float = 1.0,
        noise_multiplier: float = 1.1,
        target_delta: float = 1e-5,
        server_lr: float = 1.0,
        sample_rate: float = 1.0,
    ):
        self.global_model = global_model
        self.clip_norm_c = float(clip_norm_c)
        self.noise_multiplier = float(noise_multiplier)
        self.server_lr = float(server_lr)
        self.sample_rate = float(sample_rate)
        self.accountant = RdpPrivacyAccountant(target_delta=target_delta)
        self.round_history: List[FederatedRoundReport] = []

    @staticmethod
    def extract_trainable_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Returns a cloned dictionary of all trainable parameters (`requires_grad=True`).
        """
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @staticmethod
    def compute_client_delta(
        global_params: Dict[str, torch.Tensor],
        client_model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes $\Delta_k = \theta_{\text{client}} - \theta_{\text{global}}$ for
        all trainable parameters.
        """
        delta: Dict[str, torch.Tensor] = {}
        for name, param in client_model.named_parameters():
            if param.requires_grad and name in global_params:
                delta[name] = param.detach() - global_params[name].to(param.device)
        return delta

    def clip_client_delta(
        self, client_delta: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """
        Clips a single client's parameter delta dictionary to L2 norm at most
        `self.clip_norm_c`. Returns `(clipped_delta, unclipped_l2_norm, clipped_l2_norm)`.
        """
        sq_sum = 0.0
        for tensor in client_delta.values():
            sq_sum += float(torch.sum(tensor.float() ** 2).item())
        unclipped_norm = math.sqrt(max(sq_sum, 0.0))

        scale = min(1.0, self.clip_norm_c / max(unclipped_norm, 1e-12))
        clipped_delta = {
            name: tensor * scale for name, tensor in client_delta.items()
        }
        clipped_norm = unclipped_norm * scale
        return clipped_delta, unclipped_norm, clipped_norm

    @torch.no_grad()
    def aggregate_and_update(
        self,
        client_deltas: Sequence[Dict[str, torch.Tensor]],
        rng_seed: Optional[int] = None,
    ) -> FederatedRoundReport:
        """
        Clips all client deltas, adds calibrated Gaussian DP noise, updates
        `self.global_model`, and records the Rényi DP expenditure.
        """
        if not client_deltas:
            raise ValueError("client_deltas must contain at least one client update.")

        num_clients = len(client_deltas)
        unclipped_norms: List[float] = []
        clipped_norms: List[float] = []

        accumulated: Dict[str, torch.Tensor] = {}
        for delta in client_deltas:
            clipped, u_norm, c_norm = self.clip_client_delta(delta)
            unclipped_norms.append(u_norm)
            clipped_norms.append(c_norm)
            for name, tensor in clipped.items():
                if name not in accumulated:
                    accumulated[name] = tensor.clone().float()
                else:
                    accumulated[name] = accumulated[name] + tensor.float()

        noise_std = (self.noise_multiplier * self.clip_norm_c) / float(num_clients)
        gen: Optional[torch.Generator] = None
        if rng_seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(rng_seed))

        agg_sq_sum = 0.0
        named_params = dict(self.global_model.named_parameters())
        for name, sum_tensor in accumulated.items():
            mean_tensor = sum_tensor / float(num_clients)
            if noise_std > 0.0:
                if gen is not None:
                    noise_cpu = torch.randn(
                        mean_tensor.shape, generator=gen, dtype=torch.float32
                    )
                    noise = noise_cpu.to(mean_tensor.device) * noise_std
                else:
                    noise = torch.randn_like(mean_tensor) * noise_std
                noisy_update = mean_tensor + noise
            else:
                noisy_update = mean_tensor

            agg_sq_sum += float(torch.sum(noisy_update ** 2).item())
            if name in named_params and named_params[name].requires_grad:
                target_p = named_params[name]
                target_p.add_(
                    noisy_update.to(device=target_p.device, dtype=target_p.dtype),
                    alpha=self.server_lr,
                )

        eps, opt_alpha = self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
        )
        report = FederatedRoundReport(
            round_index=len(self.round_history) + 1,
            num_clients=num_clients,
            mean_unclipped_l2_norm=sum(unclipped_norms) / float(num_clients),
            mean_clipped_l2_norm=sum(clipped_norms) / float(num_clients),
            clip_norm_c=self.clip_norm_c,
            noise_multiplier=self.noise_multiplier,
            cumulative_epsilon=eps,
            target_delta=self.accountant.target_delta,
            optimal_rdp_order=opt_alpha,
            aggregated_update_l2_norm=math.sqrt(max(agg_sq_sum, 0.0)),
        )
        self.round_history.append(report)
        return report

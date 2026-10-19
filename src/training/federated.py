"""
Cross-Silo Federated Self-Supervised Pre-Training with Differential Privacy.

============================================================================
PURPOSE
============================================================================
Enables consortium banking institutions to collaboratively pre-train a shared
`LargeDataModel` (`MMTT` + `LimiX`/`PRAGMA` + `CoLES` + `RelationalEntityGNN`)
without centralizing raw customer transaction logs or leaking proprietary
`TabR` memory banks.

Key Capabilities:
  1. Split-Architecture Personalization (`FedPer` / Cross-Silo):
     Aggregates only shared representation parameters (`encoder.*`, `ssl_*`,
     `relational_gnn.*`) while keeping institution-specific memory buffers
     (`tabr_retrieval.memory_*`) strictly local to each bank's silo.
  2. Proximal Regularization (`FedProx`, Li et al., MLSys 2020):
     Penalizes client drift under non-IID institutional transaction mixes via
     `(\mu / 2) * ||w_local - w_global||_2^2`.
  3. Update-Level L2 Clipping & Gaussian Differential Privacy (McMahan et al.,
     ICLR 2018; Abadi et al., CCS 2016):
     Bounds each institution's L2 update sensitivity to `max_grad_norm` (`C`)
     and injects calibrated Gaussian noise `N(0, (sigma * C / K)^2 I)`.
  4. Analytical Privacy Accountant (`compute_dp_epsilon`):
     Tracks cumulative `(epsilon, delta)` privacy expenditure across rounds.

============================================================================
REFERENCES
============================================================================
  - McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017):
    "Communication-Efficient Learning of Deep Networks from Decentralized
    Data" (FedAvg, AISTATS 2017)
  - Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V.
    (2020): "Federated Optimization in Heterogeneous Networks" (FedProx,
    MLSys 2020)
  - McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018): "Learning
    Differentially Private Recurrent Language Models" (ICLR 2018)
============================================================================
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math

import torch
import torch.nn as nn

from src.models.ldm_model import LargeDataModel
from src.training.lightning_module import LDMPreTrainingModule


SHARED_PARAMETER_PREFIXES: Sequence[str] = (
    "encoder.",
    "ssl_",
    "relational_gnn.",
)


@dataclass(frozen=True)
class FederatedPrivacyConfig:
    """Configuration for FedProx and (epsilon, delta)-Differential Privacy."""

    enable_dp: bool = True
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.05
    delta: float = 1e-5
    proximal_mu: float = 0.01


@dataclass(frozen=True)
class FederatedRoundReport:
    """Telemetry from a cross-silo federated pre-training round."""

    round_idx: int
    num_clients: int
    mean_client_ssl_loss: float
    mean_update_l2_norm: float
    clipped_clients_count: int
    cumulative_dp_epsilon: float
    dp_delta: float


class FederatedLDMCoordinator:
    """
    Coordinates cross-silo federated self-supervised pre-training (`FedAvg` /
    `FedProx`) with update-level Differential Privacy.

    Args:
        global_model: Master `LargeDataModel` instance.
        privacy_config: `FederatedPrivacyConfig` instance.
        shared_prefixes: Parameter name prefixes eligible for global aggregation.
    """

    def __init__(
        self,
        global_model: LargeDataModel,
        privacy_config: Optional[FederatedPrivacyConfig] = None,
        shared_prefixes: Sequence[str] = SHARED_PARAMETER_PREFIXES,
    ):
        self.global_model = global_model
        self.privacy_config = privacy_config or FederatedPrivacyConfig()
        self.shared_prefixes = tuple(shared_prefixes)
        self.rounds_completed: int = 0

    def is_shared_parameter(self, param_name: str) -> bool:
        """Returns True if `param_name` belongs to the globally shared backbone."""
        return param_name.startswith(self.shared_prefixes)

    def get_shared_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extracts detached clone of shared floating-point parameters."""
        shared: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if self.is_shared_parameter(name) and param.requires_grad:
                shared[name] = param.detach().clone()
        return shared

    def compute_fedprox_penalty(
        self,
        local_model: nn.Module,
        global_shared_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the differentiable FedProx proximal regularization penalty:
            (mu / 2) * sum_{theta in Shared} ||theta_local - theta_global||_2^2
        """
        mu = float(self.privacy_config.proximal_mu)
        device = next(local_model.parameters()).device
        if mu <= 0.0:
            return torch.tensor(0.0, device=device)

        prox_sum = torch.tensor(0.0, device=device)
        for name, param in local_model.named_parameters():
            if name in global_shared_state and param.requires_grad:
                target = global_shared_state[name].to(device=param.device, dtype=param.dtype)
                prox_sum = prox_sum + torch.sum((param - target).pow(2))

        return 0.5 * mu * prox_sum

    def compute_dp_epsilon(self, num_rounds: Optional[int] = None) -> float:
        """
        Computes cumulative privacy budget `epsilon` at target `delta` under
        Gaussian composition across `num_rounds` federated aggregations.
        """
        if not self.privacy_config.enable_dp:
            return float("inf")
        t_rounds = self.rounds_completed if num_rounds is None else num_rounds
        if t_rounds <= 0:
            return 0.0
        sigma = max(float(self.privacy_config.noise_multiplier), 1e-6)
        delta = max(min(float(self.privacy_config.delta), 0.5), 1e-12)
        return float(math.sqrt(2.0 * float(t_rounds) * math.log(1.25 / delta)) / sigma)

    @torch.no_grad()
    def aggregate_client_updates(
        self,
        client_models: Sequence[LargeDataModel],
        client_weights: Optional[Sequence[float]] = None,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Clips each client's shared parameter delta to `max_grad_norm`, applies
        weighted FedAvg aggregation, and injects calibrated Gaussian DP noise.

        Args:
            client_models: Sequence of locally updated `LargeDataModel` instances.
            client_weights: Optional positive weights per client (e.g., sample counts).
            seed: Deterministic seed for Gaussian DP noise generation.

        Returns:
            Dictionary with `"mean_update_l2_norm"` and `"clipped_clients_count"`.
        """
        num_clients = len(client_models)
        if num_clients == 0:
            raise ValueError("client_models must contain at least 1 client model.")

        if client_weights is None:
            weights = [1.0 / float(num_clients)] * num_clients
        else:
            total_w = float(sum(client_weights))
            if total_w <= 0.0:
                raise ValueError("Sum of client_weights must be positive.")
            weights = [float(w) / total_w for w in client_weights]

        global_shared = self.get_shared_state_dict(self.global_model)
        aggregated_deltas: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(tensor) for name, tensor in global_shared.items()
        }

        clip_bound = float(self.privacy_config.max_grad_norm)
        update_norms: List[float] = []
        clipped_count = 0

        for client_idx, client_model in enumerate(client_models):
            client_shared = self.get_shared_state_dict(client_model)
            # Compute global L2 norm of client delta across all shared parameters
            sq_norm = 0.0
            deltas_k: Dict[str, torch.Tensor] = {}
            for name, g_tensor in global_shared.items():
                c_tensor = client_shared[name].to(device=g_tensor.device, dtype=g_tensor.dtype)
                delta = c_tensor - g_tensor
                deltas_k[name] = delta
                sq_norm += float(torch.sum(delta.pow(2)).item())

            l2_norm = math.sqrt(sq_norm)
            update_norms.append(l2_norm)

            clip_scale = 1.0
            if self.privacy_config.enable_dp and l2_norm > clip_bound:
                clip_scale = clip_bound / (l2_norm + 1e-12)
                clipped_count += 1

            w_k = weights[client_idx]
            for name, delta in deltas_k.items():
                aggregated_deltas[name].add_(delta * (clip_scale * w_k))

        # Inject calibrated Gaussian noise and update global model parameters
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + self.rounds_completed * 101)

        noise_std = (
            (float(self.privacy_config.noise_multiplier) * clip_bound) / float(num_clients)
            if self.privacy_config.enable_dp
            else 0.0
        )

        for name, param in self.global_model.named_parameters():
            if name in aggregated_deltas and param.requires_grad:
                delta_update = aggregated_deltas[name]
                if noise_std > 0.0:
                    noise = torch.randn(
                        delta_update.shape,
                        generator=gen,
                        dtype=delta_update.dtype,
                        device="cpu",
                    ).to(delta_update.device) * noise_std
                    delta_update = delta_update + noise
                param.copy_(global_shared[name] + delta_update)

        self.rounds_completed += 1
        return {
            "mean_update_l2_norm": float(sum(update_norms) / len(update_norms)),
            "clipped_clients_count": float(clipped_count),
        }

    def run_federated_ssl_round(
        self,
        client_dataloaders: Sequence[Any],
        local_steps_per_client: int = 2,
        learning_rate: float = 1e-3,
        ssl_masking_prob: float = 0.15,
        seed: int = 42,
    ) -> FederatedRoundReport:
        """
        Executes one complete cross-silo federated self-supervised pre-training
        round (`LimiX` + `PRAGMA` + `CoLES` + `FedProx`) across `K` banking silos.

        Args:
            client_dataloaders: Sequence of PyTorch DataLoaders (one per silo).
            local_steps_per_client: Number of local SGD/AdamW batches per silo.
            learning_rate: Local optimizer learning rate.
            ssl_masking_prob: Masking ratio for LimiX/PRAGMA objective.
            seed: Random seed for reproducible DP noise.

        Returns:
            `FederatedRoundReport` summarizing losses, L2 norms, and DP epsilon.
        """
        num_clients = len(client_dataloaders)
        if num_clients == 0:
            raise ValueError("client_dataloaders must contain at least 1 DataLoader.")

        global_shared_snapshot = self.get_shared_state_dict(self.global_model)
        client_models: List[LargeDataModel] = []
        client_losses: List[float] = []

        for dataloader in client_dataloaders:
            local_model = deepcopy(self.global_model)
            local_model.train()

            ssl_module = LDMPreTrainingModule(
                local_model,
                {
                    "learning_rate": learning_rate,
                    "ssl_masking_prob": ssl_masking_prob,
                    "ssl_temperature": 0.15,
                },
            )
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=learning_rate)

            step_losses: List[float] = []
            for step_idx, batch in enumerate(dataloader):
                if step_idx >= local_steps_per_client:
                    break
                optimizer.zero_grad()
                ssl_loss, _ = ssl_module._compute_ssl_step_loss(batch)
                prox_loss = self.compute_fedprox_penalty(
                    local_model, global_shared_snapshot
                )
                total_loss = ssl_loss + prox_loss
                total_loss.backward()
                optimizer.step()
                step_losses.append(float(ssl_loss.detach().item()))

            client_models.append(local_model)
            client_losses.append(
                sum(step_losses) / max(1, len(step_losses)) if step_losses else 0.0
            )

        agg_stats = self.aggregate_client_updates(client_models, seed=seed)

        return FederatedRoundReport(
            round_idx=self.rounds_completed,
            num_clients=num_clients,
            mean_client_ssl_loss=float(sum(client_losses) / len(client_losses)),
            mean_update_l2_norm=float(agg_stats["mean_update_l2_norm"]),
            clipped_clients_count=int(agg_stats["clipped_clients_count"]),
            cumulative_dp_epsilon=self.compute_dp_epsilon(),
            dp_delta=float(self.privacy_config.delta),
        )

"""
Multi-Gate Mixture-of-Experts (MMoE) Head & Homoscedastic Multi-Task Loss.

============================================================================
PURPOSE
============================================================================
Enables a single shared Large Data Model (LDM) representation (MMTT +
RelBench Bipartite GNN + TabR Retrieval) to simultaneously serve three core
financial decisioning tasks:
  1. Real-Time Payment Fraud Detection (`fraud_logits`, binary classification)
  2. 90-Day Credit Delinquency / Default Risk (`credit_default_logits`, binary
     classification)
  3. Customer Lifetime Value / Future Spend (`ltv_pred`, continuous regression)

Synthesizes two peer-reviewed multi-task architectures:
  1. Multi-Gate Mixture-of-Experts (MMoE) — Ma et al. (KDD 2018, "Modeling
     Task Relationships in Multi-task Learning with Multi-gate
     Mixture-of-Experts"):
     - Replaces hard parameter sharing with a bank of `E` shared expert subnetworks
       and task-specific softmax gating networks `g^{(k)}(x) = Softmax(W_g^{(k)} x)`
       that dynamically route representations according to task correlation.
  2. Homoscedastic Task Uncertainty Weighting — Kendall, Gal, & Cipolla
     (CVPR 2018, "Multi-Task Learning Using Uncertainty to Weigh Losses for
     Scene Geometry and Semantics", arXiv:1705.07115):
     - Learns task-level log-variance parameters `s_k = log(sigma_k^2)` to
       automatically balance binary cross-entropy and continuous Huber
       regression gradients without manual loss-weight grid searches.
============================================================================
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertSubnetwork(nn.Module):
    """
    Single residual MLP expert inside the Multi-Gate Mixture-of-Experts bank.
    """

    def __init__(self, input_dim: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, expert_dim)
        self.norm1 = nn.LayerNorm(expert_dim)
        self.fc1 = nn.Linear(expert_dim, expert_dim)
        self.norm2 = nn.LayerNorm(expert_dim)
        self.fc2 = nn.Linear(expert_dim, expert_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.in_proj(x)))
        res = h
        h = self.dropout(F.gelu(self.norm2(self.fc1(h))))
        h = self.fc2(h) + res
        return h


class MultiGateMixtureOfExpertsHead(nn.Module):
    """
    Multi-Gate Mixture-of-Experts (`MMoE`) Head for Joint Financial Decisioning.

    Args:
        input_dim: Dimension of the joint entity representation (`hidden_size + tabular_dim`).
        expert_dim: Hidden dimension of each expert subnetwork.
        num_experts: Number of shared expert networks (`E`, default: 4).
        num_tasks: Number of downstream tasks (`K`, fixed at 3: Fraud, Credit Default, LTV).
        dropout: Dropout rate inside experts and task towers.
    """

    def __init__(
        self,
        input_dim: int,
        expert_dim: int = 128,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.num_tasks = 3

        self.input_norm = nn.LayerNorm(input_dim)
        self.experts = nn.ModuleList(
            [
                ExpertSubnetwork(input_dim=input_dim, expert_dim=expert_dim, dropout=dropout)
                for _ in range(num_experts)
            ]
        )

        # Task-specific gating networks: [B, input_dim] -> [B, num_experts]
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, num_experts) for _ in range(self.num_tasks)]
        )

        # Task-specific prediction towers
        self.fraud_tower = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, expert_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim // 2, 1),
        )
        self.credit_default_tower = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, expert_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim // 2, 1),
        )
        self.ltv_tower = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, expert_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim // 2, 1),
        )

    def forward(self, joint_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes multi-task predictions and expert routing weights.

        Args:
            joint_features: Concatenated `[enriched_embedding, tabular_features]`
                of shape `[B, input_dim]`.

        Returns:
            Dictionary containing:
              - `"fraud_logits"`: Tensor `[B, 1]`
              - `"credit_default_logits"`: Tensor `[B, 1]`
              - `"ltv_pred"`: Tensor `[B, 1]`
              - `"gate_weights"`: Tensor `[B, 3, num_experts]` summing to 1 along `dim=-1`
        """
        x = self.input_norm(joint_features)

        # Stack expert outputs: [B, num_experts, expert_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        task_mixtures = []
        gate_distributions = []
        for gate in self.gates:
            weights = F.softmax(gate(x), dim=-1)  # [B, num_experts]
            gate_distributions.append(weights)
            # Weighted combination of experts: [B, expert_dim]
            mixture = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
            task_mixtures.append(mixture)

        gate_weights = torch.stack(gate_distributions, dim=1)  # [B, 3, num_experts]

        return {
            "fraud_logits": self.fraud_tower(task_mixtures[0]),
            "credit_default_logits": self.credit_default_tower(task_mixtures[1]),
            "ltv_pred": self.ltv_tower(task_mixtures[2]),
            "gate_weights": gate_weights,
        }


class HomoscedasticMultiTaskLoss(nn.Module):
    """
    Homoscedastic Uncertainty-Weighted Multi-Task Loss (Kendall et al., CVPR 2018).

    Automatically learns task log-variances `log_vars = [s_fraud, s_credit, s_ltv]`
    to balance classification (`BCEWithLogitsLoss`) and continuous regression
    (`HuberLoss`) objectives:
        L_total = exp(-s_0) * L_fraud + 0.5 * s_0
                + exp(-s_1) * L_credit + 0.5 * s_1
                + 0.5 * exp(-s_2) * L_ltv + 0.5 * s_2
    """

    def __init__(self, huber_delta: float = 1.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        fraud_targets: torch.Tensor,
        credit_default_targets: torch.Tensor,
        ltv_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes individual task losses and uncertainty-weighted total loss.
        """
        fraud_loss = F.binary_cross_entropy_with_logits(
            outputs["fraud_logits"].view(-1),
            fraud_targets.view(-1).float(),
        )
        credit_loss = F.binary_cross_entropy_with_logits(
            outputs["credit_default_logits"].view(-1),
            credit_default_targets.view(-1).float(),
        )
        ltv_loss = self.huber(
            outputs["ltv_pred"].view(-1),
            ltv_targets.view(-1).float(),
        )

        # Clamp log_vars to [-4.0, 4.0] to prevent numerical overflow
        s = self.log_vars.clamp(min=-4.0, max=4.0)
        prec_0 = torch.exp(-s[0])
        prec_1 = torch.exp(-s[1])
        prec_2 = torch.exp(-s[2])

        total_loss = (
            prec_0 * fraud_loss + 0.5 * s[0]
            + prec_1 * credit_loss + 0.5 * s[1]
            + 0.5 * prec_2 * ltv_loss + 0.5 * s[2]
        )

        return {
            "loss": total_loss,
            "fraud_loss": fraud_loss.detach(),
            "credit_default_loss": credit_loss.detach(),
            "ltv_loss": ltv_loss.detach(),
            "task_weights": torch.stack([prec_0, prec_1, 0.5 * prec_2]).detach(),
        }

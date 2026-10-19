"""
Causal Counterfactual Treatment Uplift (`DragonNet`) & Intervention Policy Optimizer.

============================================================================
PURPOSE
============================================================================
In real-time payment authorization and credit underwriting, predicting pure
fraud probability P(Y=1 | X) is insufficient when the platform can apply
intermediate interventions (e.g., Step-Up 2FA / biometric challenge T=1 vs.
frictionless approval T=0).

This module implements:
  1. `CausalDragonNetHead` (Shi, Blei, & Veitch, NeurIPS 2019, "Adapting
     Neural Networks for the Estimation of Treatment Effects", arXiv:1906.02120):
     - Maps the shared representation `z(x)` to three simultaneous heads:
       a) Propensity score logits `t_logits = logit(P(T=1 | X=x))` isolating
          treatment-relevant confounding signals,
       b) Untreated potential outcome risk `y0_logits = logit(P(Y(0)=1 | X=x))`
          (fraud risk under frictionless approval),
       c) Treated potential outcome risk `y1_logits = logit(P(Y(1)=1 | X=x))`
          (residual fraud risk after Step-Up 2FA challenge),
     - Computes the Conditional Average Treatment Effect (CATE / risk uplift):
           cate = P(Y(1)=1 | X=x) - P(Y(0)=1 | X=x)
       (typically negative when Step-Up 2FA suppresses fraud risk).
  2. `DragonNetTargetedLoss`:
     - Combines factual outcome binary cross-entropy, propensity score binary
       cross-entropy, and Targeted Maximum Likelihood Estimation (TMLE)
       regularization using the inverse-propensity clever covariate:
           H(t, e(x)) = t / e(x) - (1 - t) / (1 - e(x))
  3. `recommend_optimal_intervention`:
     - Cost-sensitive policy decision rule selecting `"FRICTIONLESS_APPROVE"`,
       `"STEP_UP_2FA"`, or `"HARD_BLOCK"` to minimize expected monetary loss.
============================================================================
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDragonNetHead(nn.Module):
    """
    DragonNet Three-Head Causal Architecture for Counterfactual Risk & Uplift.

    Args:
        input_dim: Dimension of the joint entity representation (`hidden_size + tabular_dim`).
        hidden_dim: Shared representation trunk dimension (default: 128).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Head 1: Propensity score P(T=1 | z(x))
        self.propensity_head = nn.Linear(hidden_dim, 1)

        # Head 2: Control potential outcome P(Y(0)=1 | z(x)) (Frictionless risk)
        self.y0_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Head 3: Treated potential outcome P(Y(1)=1 | z(x)) (Step-Up 2FA risk)
        self.y1_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable TMLE targeted regularization parameter epsilon
        self.tmle_epsilon = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, joint_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes propensity scores, counterfactual outcome probabilities, and CATE.

        Args:
            joint_features: Tensor of shape `[B, input_dim]`.

        Returns:
            Dictionary containing:
              - `"propensity_logits"`: `[B, 1]`
              - `"propensity_prob"`: `[B, 1]`
              - `"y0_logits"`: `[B, 1]` (Frictionless risk logits)
              - `"y1_logits"`: `[B, 1]` (Step-Up 2FA risk logits)
              - `"y0_prob"`: `[B, 1]`
              - `"y1_prob"`: `[B, 1]`
              - `"cate"`: `[B, 1]` (`y1_prob - y0_prob`, risk delta under treatment)
              - `"risk_reduction"`: `[B, 1]` (`y0_prob - y1_prob`)
        """
        z = self.shared_trunk(joint_features)
        t_logits = self.propensity_head(z)
        y0_logits = self.y0_tower(z)
        y1_logits = self.y1_tower(z)

        t_prob = torch.sigmoid(t_logits)
        y0_prob = torch.sigmoid(y0_logits)
        y1_prob = torch.sigmoid(y1_logits)
        cate = y1_prob - y0_prob

        return {
            "propensity_logits": t_logits,
            "propensity_prob": t_prob,
            "y0_logits": y0_logits,
            "y1_logits": y1_logits,
            "y0_prob": y0_prob,
            "y1_prob": y1_prob,
            "cate": cate,
            "risk_reduction": -cate,
        }


class DragonNetTargetedLoss(nn.Module):
    """
    DragonNet Objective with Targeted Maximum Likelihood Regularization (Shi et al., 2019).

    Args:
        alpha_propensity: Weight on the propensity cross-entropy term (default: `1.0`).
        beta_targeted: Weight on the TMLE targeted regularization term (default: `0.5`).
    """

    def __init__(
        self,
        alpha_propensity: float = 1.0,
        beta_targeted: float = 0.5,
    ):
        super().__init__()
        self.alpha_propensity = alpha_propensity
        self.beta_targeted = beta_targeted

    def forward(
        self,
        causal_outputs: Dict[str, torch.Tensor],
        treatments: torch.Tensor,
        outcomes: torch.Tensor,
        tmle_epsilon: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes factual outcome loss, propensity loss, and TMLE targeted loss.

        Args:
            causal_outputs: Output dictionary from `CausalDragonNetHead.forward`.
            treatments: Binary treatment assignment tensor `[B]` or `[B, 1]` (`0` or `1`).
            outcomes: Observed binary fraud outcome tensor `[B]` or `[B, 1]` (`0` or `1`).
            tmle_epsilon: Learnable scalar parameter `model.causal_head.tmle_epsilon`.
        """
        t = treatments.view(-1, 1).float()
        y = outcomes.view(-1, 1).float()

        y0_logits = causal_outputs["y0_logits"]
        y1_logits = causal_outputs["y1_logits"]
        t_logits = causal_outputs["propensity_logits"]

        # Factual outcome logits: Q(t_i, x_i) = (1 - t_i) * Q_0 + t_i * Q_1
        factual_logits = (1.0 - t) * y0_logits + t * y1_logits
        factual_bce = F.binary_cross_entropy_with_logits(factual_logits, y)
        propensity_bce = F.binary_cross_entropy_with_logits(t_logits, t)

        # Clever covariate H(t, e(x)) with propensity clipping in [0.02, 0.98]
        e_prob = torch.sigmoid(t_logits).clamp(min=0.02, max=0.98)
        clever_cov = (t / e_prob) - ((1.0 - t) / (1.0 - e_prob))

        tilted_logits = factual_logits + tmle_epsilon * clever_cov
        targeted_bce = F.binary_cross_entropy_with_logits(tilted_logits, y)

        total_loss = (
            factual_bce
            + self.alpha_propensity * propensity_bce
            + self.beta_targeted * targeted_bce
        )

        return {
            "loss": total_loss,
            "factual_bce": factual_bce.detach(),
            "propensity_bce": propensity_bce.detach(),
            "targeted_bce": targeted_bce.detach(),
        }


@torch.no_grad()
def recommend_optimal_intervention(
    causal_outputs: Dict[str, torch.Tensor],
    fraud_loss_cost: float = 100.0,
    step_up_friction_cost: float = 5.0,
    false_decline_cost: float = 25.0,
) -> Dict[str, torch.Tensor | List[str]]:
    """
    Computes the cost-minimizing intervention policy for each transaction:
      - Action 0 (`"FRICTIONLESS_APPROVE"`): Expected cost = `y0_prob * fraud_loss_cost`
      - Action 1 (`"STEP_UP_2FA"`): Expected cost = `step_up_friction_cost + y1_prob * fraud_loss_cost`
      - Action 2 (`"HARD_BLOCK"`): Expected cost = `(1 - y0_prob) * false_decline_cost`

    Returns:
        Dictionary with `"action_indices"` (`0`, `1`, `2`), `"action_names"`,
        and `"expected_costs"` (`[B, 3]`).
    """
    y0 = causal_outputs["y0_prob"].view(-1, 1)
    y1 = causal_outputs["y1_prob"].view(-1, 1)

    cost_approve = y0 * fraud_loss_cost
    cost_step_up = step_up_friction_cost + y1 * fraud_loss_cost
    cost_block = (1.0 - y0) * false_decline_cost

    costs = torch.cat([cost_approve, cost_step_up, cost_block], dim=-1)  # [B, 3]
    best_idx = torch.argmin(costs, dim=-1)

    label_map = {
        0: "FRICTIONLESS_APPROVE",
        1: "STEP_UP_2FA",
        2: "HARD_BLOCK",
    }
    names = [label_map[int(i.item())] for i in best_idx]

    return {
        "action_indices": best_idx,
        "action_names": names,
        "expected_costs": costs,
    }

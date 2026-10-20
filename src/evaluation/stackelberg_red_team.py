r"""
Multi-Agent Game-Theoretic Cost-Aware Adversarial Red-Teaming (`StackelbergLDM`).

Scientific & Regulatory Anchors:
  - Dalvi, N., Domingos, P., Mausam, Sanghai, S., & Verma, D. (2004).
    "Adversarial Classification." *Proceedings of the 10th ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining (KDD '04)*.
  - Brückner, M., Kanzow, C., & Scheffer, T. (2012). "Static Prediction Games
    for Adversarial Learning Problems." *Journal of Machine Learning Research
    (JMLR)*, 13, 2617-2654.
  - Hardt, M., Megiddo, N., Papadimitriou, C., & Wootters, M. (2016).
    "Strategic Classification." *Proceedings of the 2016 ACM Conference on
    Innovations in Theoretical Computer Science (ITCS '16)*, arXiv:1506.06980.

Models fraud syndicates as rational economic followers in a bilevel Stackelberg
game who perturb transaction amounts and static features only when expected
evasion utility exceeds asymmetric feature manipulation costs:
$$\delta^*(x) = \arg\max_{\|\delta\|_\infty \le \epsilon} \left\{ \log\big(1 - \sigma(f_\theta(x + \delta))\big) - \sum_{j} c_j |\delta_j| - \frac{\lambda_2}{2} \|\delta\|_2^2 \right\}$$
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StackelbergEquilibriumReport:
    r"""
    Audit report summarizing the bilevel Stackelberg equilibrium between the
    cost-constrained rational attacker and the defender's optimal threshold.
    """

    num_attacked_samples: int
    baseline_detection_rate: float
    post_attack_detection_rate: float
    stackelberg_hardened_detection_rate: float
    mean_attacker_net_utility: float
    mean_manipulation_cost: float
    defender_equilibrium_threshold: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_attacked_samples": self.num_attacked_samples,
            "baseline_detection_rate": round(self.baseline_detection_rate, 6),
            "post_attack_detection_rate": round(self.post_attack_detection_rate, 6),
            "stackelberg_hardened_detection_rate": round(
                self.stackelberg_hardened_detection_rate, 6
            ),
            "mean_attacker_net_utility": round(self.mean_attacker_net_utility, 6),
            "mean_manipulation_cost": round(self.mean_manipulation_cost, 6),
            "defender_equilibrium_threshold": round(
                self.defender_equilibrium_threshold, 6
            ),
        }


class CostAwareStackelbergRedTeam:
    r"""
    Bilevel Stackelberg adversarial red-teaming engine for `LargeDataModel`.

    Simulates a rational attacker solving a cost-penalized evasion objective via
    proximal projected gradient ascent (`Cost-PGD`) where each feature coordinate
    has an asymmetric manipulation cost $c_j > 0$ (e.g., splitting transaction
    amounts has low cost, whereas altering verified credit bureau tenure has
    prohibitive cost).

    Args:
        epsilon_max: Maximum $\ell_\infty$ perturbation budget $\epsilon > 0$.
        pgd_steps: Number of proximal gradient steps $T$ for attacker best-response.
        step_size: Per-step learning rate $\alpha$ for attacker optimization.
        value_manipulation_cost: Unit $\ell_1$ cost of perturbing transaction amounts.
        tabular_cost_weights: Optional per-feature manipulation cost vector $c_{\text{tab}}$.
        l2_elastic_penalty: Quadratic manipulation cost coefficient $\lambda_2 \ge 0$.
    """

    def __init__(
        self,
        epsilon_max: float = 0.35,
        pgd_steps: int = 6,
        step_size: float = 0.08,
        value_manipulation_cost: float = 0.15,
        tabular_cost_weights: Optional[torch.Tensor] = None,
        l2_elastic_penalty: float = 0.10,
    ):
        self.epsilon_max = max(float(epsilon_max), 1e-5)
        self.pgd_steps = max(int(pgd_steps), 1)
        self.step_size = max(float(step_size), 1e-4)
        self.value_manipulation_cost = max(float(value_manipulation_cost), 0.0)
        self.tabular_cost_weights = tabular_cost_weights
        self.l2_elastic_penalty = max(float(l2_elastic_penalty), 0.0)

    def _get_tabular_costs(self, tabular_features: torch.Tensor) -> torch.Tensor:
        feat_dim = tabular_features.size(-1)
        if self.tabular_cost_weights is not None:
            c = self.tabular_cost_weights.to(
                device=tabular_features.device, dtype=tabular_features.dtype
            ).view(1, -1)
            if c.size(-1) == feat_dim:
                return c.clamp(min=0.0)
        # Default heterogeneous cost profile: higher-indexed bureau features cost more to spoof
        grid = torch.linspace(
            0.10,
            1.20,
            steps=feat_dim,
            device=tabular_features.device,
            dtype=tabular_features.dtype,
        )
        return grid.unsqueeze(0)

    def compute_attacker_best_response(
        self,
        model: nn.Module,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Computes the rational attacker's best-response perturbations
        $(\delta_{\text{val}}^*, \delta_{\text{tab}}^*)$ via proximal gradient
        ascent on the cost-regularized evasion objective.

        Returns:
            Tuple `(adv_values, adv_tabular, net_utility_per_sample, cost_per_sample)`.
        """
        model.eval()
        c_tab = self._get_tabular_costs(tabular_features)
        c_val = self.value_manipulation_cost

        delta_val = torch.zeros_like(packed_values, requires_grad=True)
        delta_tab = torch.zeros_like(tabular_features, requires_grad=True)

        for _ in range(self.pgd_steps):
            adv_val = (packed_values + delta_val).clamp(min=0.0)
            adv_tab = tabular_features + delta_tab

            logits = model(
                packed_keys=packed_keys,
                packed_values=adv_val,
                packed_times=packed_times,
                tabular_features=adv_tab,
                key_padding_mask=key_padding_mask,
            ).view(-1)

            # Attacker wants to minimize fraud probability -> maximize log(1 - p) = -softplus(logits)
            evasion_gain = -F.softplus(logits)
            val_cost = c_val * torch.mean(torch.abs(delta_val), dim=-1) + (
                0.5 * self.l2_elastic_penalty * torch.mean(delta_val ** 2, dim=-1)
            )
            tab_cost = torch.mean(c_tab * torch.abs(delta_tab), dim=-1) + (
                0.5 * self.l2_elastic_penalty * torch.mean(delta_tab ** 2, dim=-1)
            )
            total_utility = torch.sum(evasion_gain - val_cost - tab_cost)

            grad_val, grad_tab = torch.autograd.grad(
                total_utility, [delta_val, delta_tab]
            )

            with torch.no_grad():
                # Rationality gate: features where manipulation cost exceeds marginal gradient stay zero
                new_d_val = (delta_val + self.step_size * grad_val.sign()).clamp(
                    -self.epsilon_max, self.epsilon_max
                )
                # Ensure perturbed transaction amounts remain non-negative
                new_d_val = torch.maximum(new_d_val, -packed_values)

                # Soft-thresholding proximal operator for asymmetric tabular costs
                raw_d_tab = delta_tab + self.step_size * grad_tab
                thresh = self.step_size * c_tab
                prox_d_tab = torch.sign(raw_d_tab) * F.relu(
                    torch.abs(raw_d_tab) - thresh
                )
                new_d_tab = prox_d_tab.clamp(-self.epsilon_max, self.epsilon_max)

                delta_val.copy_(new_d_val)
                delta_tab.copy_(new_d_tab)

        with torch.no_grad():
            final_val = (packed_values + delta_val).clamp(min=0.0)
            final_tab = tabular_features + delta_tab
            final_logits = model(
                packed_keys=packed_keys,
                packed_values=final_val,
                packed_times=packed_times,
                tabular_features=final_tab,
                key_padding_mask=key_padding_mask,
            ).view(-1)
            final_probs = torch.sigmoid(final_logits)
            sample_cost = c_val * torch.mean(torch.abs(delta_val), dim=-1) + torch.mean(
                c_tab * torch.abs(delta_tab), dim=-1
            )
            sample_utility = (1.0 - final_probs) - sample_cost

        return final_val.detach(), final_tab.detach(), sample_utility.detach(), sample_cost.detach()

    def evaluate_stackelberg_equilibrium(
        self,
        model: nn.Module,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        base_threshold: float = 0.50,
    ) -> StackelbergEquilibriumReport:
        r"""
        Evaluates the attacker's best response and computes the Defender's
        Stackelberg equilibrium threshold $\tau^*$ that maximizes robust
        detection utility against strategic evasion.
        """
        model.eval()
        with torch.no_grad():
            clean_logits = model(
                packed_keys=packed_keys,
                packed_values=packed_values,
                packed_times=packed_times,
                tabular_features=tabular_features,
                key_padding_mask=key_padding_mask,
            ).view(-1)
            clean_probs = torch.sigmoid(clean_logits)
            base_det = float((clean_probs >= base_threshold).float().mean().item())

        adv_val, adv_tab, net_util, manip_cost = self.compute_attacker_best_response(
            model=model,
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            tabular_features=tabular_features,
            key_padding_mask=key_padding_mask,
        )

        with torch.no_grad():
            adv_logits = model(
                packed_keys=packed_keys,
                packed_values=adv_val,
                packed_times=packed_times,
                tabular_features=adv_tab,
                key_padding_mask=key_padding_mask,
            ).view(-1)
            adv_probs = torch.sigmoid(adv_logits)
            post_det = float((adv_probs >= base_threshold).float().mean().item())

            # Defender anticipates the expected logit shift Delta_logit under best response
            mean_logit_drop = float(
                torch.clamp(clean_logits - adv_logits, min=0.0).mean().item()
            )
            base_logit_thresh = torch.logit(
                torch.tensor(base_threshold, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
            ).item()
            eq_logit_thresh = base_logit_thresh - 0.85 * mean_logit_drop
            eq_prob_thresh = float(
                torch.sigmoid(torch.tensor(eq_logit_thresh)).clamp(0.05, 0.95).item()
            )
            hardened_det = float((adv_probs >= eq_prob_thresh).float().mean().item())

        return StackelbergEquilibriumReport(
            num_attacked_samples=int(packed_keys.size(0)),
            baseline_detection_rate=base_det,
            post_attack_detection_rate=post_det,
            stackelberg_hardened_detection_rate=hardened_det,
            mean_attacker_net_utility=float(net_util.mean().item()),
            mean_manipulation_cost=float(manip_cost.mean().item()),
            defender_equilibrium_threshold=eq_prob_thresh,
        )

    def compute_stackelberg_minimax_loss(
        self,
        model: nn.Module,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: torch.Tensor,
        labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        adv_weight: float = 0.50,
    ) -> torch.Tensor:
        r"""
        Computes the bilevel Stackelberg minimax training objective combining
        clean empirical risk and cost-aware strategic best-response risk:
        $$\mathcal{L}_{\text{Stackelberg}}(\theta) = (1 - \lambda)\mathcal{L}_{\text{BCE}}(f_\theta(x), y) + \lambda \mathcal{L}_{\text{BCE}}(f_\theta(x + \delta^*(x)), y)$$
        """
        adv_val, adv_tab, _, _ = self.compute_attacker_best_response(
            model=model,
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            tabular_features=tabular_features,
            key_padding_mask=key_padding_mask,
        )
        model.train()
        targets = labels.view(-1, 1).float()
        clean_logits = model(
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            tabular_features=tabular_features,
            key_padding_mask=key_padding_mask,
        )
        adv_logits = model(
            packed_keys=packed_keys,
            packed_values=adv_val,
            packed_times=packed_times,
            tabular_features=adv_tab,
            key_padding_mask=key_padding_mask,
        )
        clean_loss = F.binary_cross_entropy_with_logits(clean_logits, targets)
        adv_loss = F.binary_cross_entropy_with_logits(adv_logits, targets)
        return (1.0 - adv_weight) * clean_loss + adv_weight * adv_loss

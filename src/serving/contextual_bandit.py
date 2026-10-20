r"""
Online Contextual Bandit Authorization Router (`NeuralLinUCB`) & Doubly-Robust Off-Policy Evaluator (`DR-OPE`).

============================================================================
RESEARCH & ARCHITECTURAL ANCHORS
============================================================================
Real-time payment authorization suffers from selective label feedback ("bandit
censoring"): when a transaction is hard-blocked, the system never observes
whether the user would have completed a legitimate purchase or filed a fraud
chargeback. Furthermore, deploying a new step-up authentication policy requires
unbiased off-policy evaluation on logged historical interactions before risking
live traffic.

Synthesizes two foundational contextual bandit & counterfactual learning paradigms:
  1. Disjoint Linear Upper Confidence Bound (`LinUCB`, Li et al., WWW 2010,
     "A Contextual-Bandit Approach to Personalized News Article Recommendation",
     arXiv:1003.0146) over Neural LDM Representations (`NeuralLinUCB`):
     - Maintains per-action ridge precision inverse $A_a^{-1} \in \mathbb{R}^{D \times D}$
       and reward-weighted context accumulator $b_a \in \mathbb{R}^D$ across
       $K$ authorization actions (`FRICTIONLESS_APPROVE`, `SMS_OTP`,
       `BIOMETRIC_3DS`, `HARD_BLOCK`).
     - Selects action $a^* = \arg\max_a \left( \hat{\theta}_a^\top x + \alpha_{\text{ucb}} \sqrt{x^\top A_a^{-1} x} \right)$
       and updates $A_a^{-1}$ in $O(D^2)$ via the exact Sherman-Morrison formula.
  2. Doubly Robust Off-Policy Evaluation (`DR-OPE`, Dudík, Langford, & Li,
     ICML 2011, "Doubly Robust Policy Evaluation and Learning", arXiv:1103.4601):
     - Combines a direct reward model $\hat{r}(x, a)$ with clipped Inverse
       Propensity Scoring (`IPS`) to remain unbiased if EITHER the logging
       propensity $\hat{\pi}_b(a \mid x)$ OR the reward model $\hat{r}(x, a)$ is
       correctly specified:
       $$\hat{V}_{\text{DR}}(\pi_e) = \frac{1}{N} \sum_{i=1}^N \left[ \hat{r}(x_i, \pi_e(x_i)) + \frac{\mathbb{I}(a_i = \pi_e(x_i))}{\hat{\pi}_b(a_i \mid x_i)} \left( r_i - \hat{r}(x_i, a_i) \right) \right]$$
============================================================================
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_BANDIT_ACTIONS: Tuple[str, ...] = (
    "FRICTIONLESS_APPROVE",
    "SMS_OTP",
    "BIOMETRIC_3DS",
    "HARD_BLOCK",
)


@dataclass(frozen=True)
class BanditActionDecision:
    """
    Single-transaction contextual bandit routing decision with UCB uncertainty.
    """

    action_index: int
    action_name: str
    expected_reward: float
    ucb_uncertainty: float
    ucb_score: float


@dataclass(frozen=True)
class OffPolicyEvaluationReport:
    """
    Counterfactual policy evaluation report comparing `DirectMethod`, `IPS`,
    and `DoublyRobust` (`DR-OPE`) estimators with 95% confidence bounds.
    """

    num_logged_samples: int
    direct_method_value: float
    ips_value: float
    doubly_robust_value: float
    dr_std_error: float
    dr_ci95_lower: float
    dr_ci95_upper: float
    effective_sample_size_pct: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "num_logged_samples": self.num_logged_samples,
            "direct_method_value": round(self.direct_method_value, 6),
            "ips_value": round(self.ips_value, 6),
            "doubly_robust_value": round(self.doubly_robust_value, 6),
            "dr_std_error": round(self.dr_std_error, 6),
            "dr_ci95_lower": round(self.dr_ci95_lower, 6),
            "dr_ci95_upper": round(self.dr_ci95_upper, 6),
            "effective_sample_size_pct": round(self.effective_sample_size_pct, 2),
        }


class NeuralLinUCBRouter:
    r"""
    Online Contextual Bandit authorization router operating over `LargeDataModel`
    joint representations $x \in \mathbb{R}^D$ using exact Sherman-Morrison
    inverse covariance updates.

    Args:
        context_dim: Dimension of the LDM context vector $x$.
        action_names: Sequence of authorization action names.
        alpha_ucb: Exploration bonus multiplier $\alpha_{\text{ucb}} \ge 0$ (default: `0.50`).
        ridge_lambda: L2 regularization constant $\lambda > 0$ initializing
            $A_a = \lambda I_D$ (default: `1.0`).
    """

    def __init__(
        self,
        context_dim: int,
        action_names: Sequence[str] = DEFAULT_BANDIT_ACTIONS,
        alpha_ucb: float = 0.50,
        ridge_lambda: float = 1.0,
    ):
        self.context_dim = int(context_dim)
        self.action_names = tuple(action_names)
        self.num_actions = len(self.action_names)
        self.alpha_ucb = float(alpha_ucb)
        self.ridge_lambda = float(ridge_lambda)

        # Per-arm inverse covariance A_a^{-1} initialized to (1 / lambda) * I
        inv_init = (1.0 / self.ridge_lambda) * torch.eye(
            self.context_dim, dtype=torch.float32
        )
        self.A_inv: List[torch.Tensor] = [
            inv_init.clone() for _ in range(self.num_actions)
        ]
        self.b: List[torch.Tensor] = [
            torch.zeros(self.context_dim, dtype=torch.float32)
            for _ in range(self.num_actions)
        ]
        self.action_counts: List[int] = [0 for _ in range(self.num_actions)]

    def _normalize_context(self, contexts: torch.Tensor) -> torch.Tensor:
        x = contexts.detach().cpu().float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return F.normalize(x, p=2.0, dim=-1)

    def select_actions(
        self,
        contexts: torch.Tensor,
    ) -> List[BanditActionDecision]:
        """
        Computes Upper Confidence Bound (`UCB`) scores across all arms for each
        context vector in `contexts` `[B, D]` and returns the greedy-UCB decisions.
        """
        x_batch = self._normalize_context(contexts)
        batch_size = x_batch.size(0)

        # Precompute theta_a = A_a^{-1} b_a for each arm
        thetas = [
            torch.mv(self.A_inv[a], self.b[a]) for a in range(self.num_actions)
        ]

        decisions: List[BanditActionDecision] = []
        for i in range(batch_size):
            x = x_batch[i]
            best_a = 0
            best_ucb = -float("inf")
            best_mean = 0.0
            best_unc = 0.0

            for a in range(self.num_actions):
                mean_r = float(torch.dot(thetas[a], x).item())
                quad = float(torch.dot(x, torch.mv(self.A_inv[a], x)).item())
                unc = self.alpha_ucb * math.sqrt(max(quad, 1e-12))
                ucb = mean_r + unc
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_a = a
                    best_mean = mean_r
                    best_unc = unc

            decisions.append(
                BanditActionDecision(
                    action_index=best_a,
                    action_name=self.action_names[best_a],
                    expected_reward=round(best_mean, 6),
                    ucb_uncertainty=round(best_unc, 6),
                    ucb_score=round(best_ucb, 6),
                )
            )
        return decisions

    def predict_expected_rewards(self, contexts: torch.Tensor) -> torch.Tensor:
        r"""
        Returns the matrix of estimated expected rewards $\hat{r}(x_i, a)$ of
        shape `[B, num_actions]`.
        """
        x_batch = self._normalize_context(contexts)
        thetas = torch.stack(
            [torch.mv(self.A_inv[a], self.b[a]) for a in range(self.num_actions)],
            dim=1,
        )  # [D, K]
        return torch.matmul(x_batch, thetas)

    def update(
        self,
        contexts: torch.Tensor,
        chosen_actions: torch.Tensor,
        observed_rewards: torch.Tensor,
    ) -> None:
        r"""
        Updates the arm parameters $(A_a^{-1}, b_a)$ online using the exact
        Sherman-Morrison rank-1 update:
        $$A_a^{-1} \leftarrow A_a^{-1} - \frac{(A_a^{-1} x)(A_a^{-1} x)^\top}{1 + x^\top A_a^{-1} x}, \quad b_a \leftarrow b_a + r \cdot x$$
        """
        x_batch = self._normalize_context(contexts)
        actions = chosen_actions.detach().cpu().view(-1).long()
        rewards = observed_rewards.detach().cpu().view(-1).float()

        for i in range(x_batch.size(0)):
            a = int(actions[i].item())
            if a < 0 or a >= self.num_actions:
                continue
            x = x_batch[i]
            r = float(rewards[i].item())

            inv_a = self.A_inv[a]
            ax = torch.mv(inv_a, x)
            denom = 1.0 + float(torch.dot(x, ax).item())
            self.A_inv[a] = inv_a - torch.outer(ax, ax) / max(denom, 1e-8)
            self.b[a] = self.b[a] + r * x
            self.action_counts[a] += 1


class DoublyRobustOffPolicyEvaluator:
    r"""
    Evaluates a target authorization policy $\pi_e$ using logged historical
    bandit feedback $(x_i, a_i, r_i, p_i)$ via the Doubly Robust (`DR-OPE`)
    estimator (Dudík, Langford, & Li, ICML 2011).

    Args:
        min_propensity_clip: Minimum clipping floor for logging propensities
            $\hat{\pi}_b(a_i \mid x_i)$ to bound importance-weight variance (default: `0.02`).
    """

    def __init__(self, min_propensity_clip: float = 0.02):
        self.min_propensity_clip = float(min_propensity_clip)

    def evaluate_policy(
        self,
        logged_actions: torch.Tensor,
        logged_rewards: torch.Tensor,
        logging_propensities: torch.Tensor,
        target_policy_actions: torch.Tensor,
        estimated_reward_matrix: torch.Tensor,
    ) -> OffPolicyEvaluationReport:
        r"""
        Computes Direct Method (`DM`), Inverse Propensity Scoring (`IPS`), and
        Doubly Robust (`DR`) policy value estimates with 95% confidence intervals.

        Args:
            logged_actions: Historical actions taken by logging policy `[N]`.
            logged_rewards: Observed net utilities/rewards `[N]`.
            logging_propensities: Probability $\hat{\pi}_b(a_i \mid x_i)$ of the
                logged action `[N]`.
            target_policy_actions: Deterministic actions $\pi_e(x_i)$ chosen by
                the new candidate policy `[N]`.
            estimated_reward_matrix: Reward predictions $\hat{r}(x_i, a)$ across
                all arms `[N, K]`.

        Returns:
            `OffPolicyEvaluationReport` with DM, IPS, DR, and 95% CI telemetry.
        """
        a_log = logged_actions.detach().cpu().view(-1).long()
        r_obs = logged_rewards.detach().cpu().view(-1).float()
        prop = logging_propensities.detach().cpu().view(-1).float().clamp(
            min=self.min_propensity_clip, max=1.0
        )
        a_tgt = target_policy_actions.detach().cpu().view(-1).long()
        r_hat_mat = estimated_reward_matrix.detach().cpu().float()

        n = int(a_log.numel())
        idx_range = torch.arange(n, dtype=torch.long)

        r_hat_target = r_hat_mat[idx_range, a_tgt]
        r_hat_logged = r_hat_mat[idx_range, a_log]

        match_indicator = (a_log == a_tgt).float()
        importance_weights = match_indicator / prop

        dm_scores = r_hat_target
        ips_scores = importance_weights * r_obs
        dr_scores = r_hat_target + importance_weights * (r_obs - r_hat_logged)

        dm_val = float(dm_scores.mean().item())
        ips_val = float(ips_scores.mean().item())
        dr_val = float(dr_scores.mean().item())

        if n > 1:
            dr_se = float(dr_scores.std(unbiased=True).item()) / math.sqrt(float(n))
        else:
            dr_se = 0.0

        # Effective Sample Size (ESS) ratio: (sum w_i)^2 / (N * sum w_i^2)
        w_sum = float(importance_weights.sum().item())
        w_sq_sum = float(importance_weights.pow(2).sum().item())
        if w_sq_sum > 1e-12:
            ess_pct = 100.0 * (w_sum * w_sum) / (float(n) * w_sq_sum)
        else:
            ess_pct = 0.0

        return OffPolicyEvaluationReport(
            num_logged_samples=n,
            direct_method_value=dm_val,
            ips_value=ips_val,
            doubly_robust_value=dr_val,
            dr_std_error=dr_se,
            dr_ci95_lower=dr_val - 1.96 * dr_se,
            dr_ci95_upper=dr_val + 1.96 * dr_se,
            effective_sample_size_pct=min(100.0, ess_pct),
        )

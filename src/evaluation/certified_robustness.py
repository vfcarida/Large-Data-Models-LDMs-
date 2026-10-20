r"""
Certified $\ell_2$ Adversarial Robustness via Randomized Smoothing.

Scientific & Regulatory Anchors:
  - Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial
    Robustness via Randomized Smoothing." *ICML 2019*, arXiv:1902.02918.
  - Salman, H., Li, J., Razenshteyn, I., Zhang, P., Zhang, H., Bubeck, S.,
    & Yang, G. (2019). "Provably Robust Deep Learning via Adversarially Trained
    Smoothed Classifiers." *NeurIPS 2019*, arXiv:1906.04584.
  - EU AI Act (Article 15) & SR 11-7 Model Risk Management: Provable mathematical
    robustness certificates guaranteeing decision invariance within an $\ell_2$
    ball $\|\delta\|_2 \le R$.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def clopper_pearson_lower_bound(
    successes: int,
    trials: int,
    alpha: float = 0.001,
) -> float:
    r"""
    Computes the exact one-sided Clopper-Pearson lower confidence bound
    $\underline{p_A}$ at level $1 - \alpha$ such that:
    $$\mathbb{P}_{X \sim \text{Binomial}(N, \underline{p_A})}(X \ge n_A) = \alpha$$
    using log-gamma binomial tail summation and 52-step bisection ($< 10^{-15}$ error).
    """
    if trials <= 0 or successes <= 0:
        return 0.0
    if successes >= trials:
        return float(alpha ** (1.0 / float(trials)))

    log_n_fact = math.lgamma(trials + 1.0)

    def binomial_upper_tail(p: float) -> float:
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        log_p = math.log(p)
        log_1mp = math.log(1.0 - p)
        total = 0.0
        for k in range(successes, trials + 1):
            log_comb = (
                log_n_fact
                - math.lgamma(k + 1.0)
                - math.lgamma(trials - k + 1.0)
            )
            total += math.exp(log_comb + k * log_p + (trials - k) * log_1mp)
        return min(total, 1.0)

    low = 0.0
    high = float(successes) / float(trials)
    for _ in range(52):
        mid = 0.5 * (low + high)
        if binomial_upper_tail(mid) < alpha:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def normal_icdf(p: float) -> float:
    r"""
    Exact inverse standard normal CDF $\Phi^{-1}(p)$ via `torch.erfinv`.
    """
    p_clamped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    t = torch.tensor(2.0 * p_clamped - 1.0, dtype=torch.float64)
    return float(math.sqrt(2.0) * torch.erfinv(t).item())


@dataclass
class CertifiedRadiusReport:
    r"""
    Audit report summarizing the Neyman-Pearson certified $\ell_2$ adversarial
    radii across a batch of transactions.
    """

    total_evaluated: int
    certified_count: int
    abstained_count: int
    certification_rate: float
    mean_certified_l2_radius: float
    median_certified_l2_radius: float
    max_certified_l2_radius: float
    noise_sigma: float
    confidence_alpha: float
    num_mc_samples: int
    per_sample_radii: List[float]
    smoothed_predictions: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_evaluated": self.total_evaluated,
            "certified_count": self.certified_count,
            "abstained_count": self.abstained_count,
            "certification_rate": round(self.certification_rate, 6),
            "mean_certified_l2_radius": round(self.mean_certified_l2_radius, 6),
            "median_certified_l2_radius": round(self.median_certified_l2_radius, 6),
            "max_certified_l2_radius": round(self.max_certified_l2_radius, 6),
            "noise_sigma": self.noise_sigma,
            "confidence_alpha": self.confidence_alpha,
            "num_mc_samples": self.num_mc_samples,
            "per_sample_radii": [round(r, 6) for r in self.per_sample_radii],
            "smoothed_predictions": self.smoothed_predictions,
        }


class RandomizedSmoothingCertifier:
    r"""
    Constructs a Gaussian-smoothed classifier $g(x)$ around `LargeDataModel` and
    computes formal Neyman-Pearson $\ell_2$ robustness certificates (Cohen et al.,
    *ICML 2019*):
    $$R(x) = \sigma \cdot \Phi^{-1}(\underline{p_A}), \quad \text{where } \underline{p_A} > \frac{1}{2}$$

    Guarantees with probability at least $1 - \alpha$ over the Monte Carlo draw
    that $g(x + \delta) = g(x)$ for all continuous covariate perturbations
    satisfying $\|\delta\|_2 \le R(x)$.

    Args:
        noise_sigma: Standard deviation $\sigma > 0$ of isotropic Gaussian smoothing noise.
        num_mc_samples: Number of Monte Carlo noise draws $N$ per sample (default: `64`).
        confidence_alpha: One-sided failure probability $\alpha \in (0, 0.5)$ (default: `0.001`).
        decision_threshold: Base probability threshold for class `1` (default: `0.50`).
    """

    def __init__(
        self,
        noise_sigma: float = 0.25,
        num_mc_samples: int = 64,
        confidence_alpha: float = 0.001,
        decision_threshold: float = 0.50,
    ):
        self.noise_sigma = max(float(noise_sigma), 1e-5)
        self.num_mc_samples = max(int(num_mc_samples), 8)
        self.confidence_alpha = min(max(float(confidence_alpha), 1e-6), 0.25)
        self.decision_threshold = float(decision_threshold)

    @torch.no_grad()
    def certify_batch(
        self,
        model: nn.Module,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rng_seed: Optional[int] = 42,
    ) -> CertifiedRadiusReport:
        r"""
        Computes certified $\ell_2$ radii for each transaction in the batch under
        isotropic Gaussian perturbations $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$
        applied to continuous covariates (`packed_values` and `tabular_features`).
        """
        model.eval()
        batch_size = packed_keys.size(0)
        class1_counts = torch.zeros(batch_size, dtype=torch.long, device=packed_keys.device)

        gen: Optional[torch.Generator] = None
        if rng_seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(rng_seed))

        for _ in range(self.num_mc_samples):
            if gen is not None:
                v_noise = torch.randn(
                    packed_values.shape, generator=gen, dtype=torch.float32
                ).to(device=packed_values.device, dtype=packed_values.dtype)
            else:
                v_noise = torch.randn_like(packed_values)

            noisy_values = (packed_values + self.noise_sigma * v_noise).clamp(min=0.0)

            if tabular_features is not None:
                if gen is not None:
                    tab_noise = torch.randn(
                        tabular_features.shape, generator=gen, dtype=torch.float32
                    ).to(device=tabular_features.device, dtype=tabular_features.dtype)
                else:
                    tab_noise = torch.randn_like(tabular_features)
                noisy_tab = tabular_features + self.noise_sigma * tab_noise
            else:
                noisy_tab = None

            logits = model(
                packed_keys=packed_keys,
                packed_values=noisy_values,
                packed_times=packed_times,
                tabular_features=noisy_tab,
                key_padding_mask=key_padding_mask,
            ).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= self.decision_threshold).long()
            class1_counts += preds

        counts_cpu = class1_counts.detach().cpu().tolist()
        radii: List[float] = []
        smoothed_preds: List[int] = []
        certified_count = 0
        abstained_count = 0

        for c1 in counts_cpu:
            c0 = self.num_mc_samples - int(c1)
            if c1 >= c0:
                top_class = 1
                n_a = int(c1)
            else:
                top_class = 0
                n_a = int(c0)

            p_lower = clopper_pearson_lower_bound(
                successes=n_a,
                trials=self.num_mc_samples,
                alpha=self.confidence_alpha,
            )
            if p_lower > 0.5:
                radius = self.noise_sigma * normal_icdf(p_lower)
                radii.append(max(radius, 0.0))
                smoothed_preds.append(top_class)
                certified_count += 1
            else:
                radii.append(0.0)
                smoothed_preds.append(-1)  # Abstain
                abstained_count += 1

        sorted_radii = sorted(radii)
        mid = batch_size // 2
        if batch_size == 0:
            median_r = 0.0
            mean_r = 0.0
            max_r = 0.0
            cert_rate = 0.0
        else:
            if batch_size % 2 == 1:
                median_r = sorted_radii[mid]
            else:
                median_r = 0.5 * (sorted_radii[mid - 1] + sorted_radii[mid])
            mean_r = sum(radii) / float(batch_size)
            max_r = max(radii)
            cert_rate = float(certified_count) / float(batch_size)

        return CertifiedRadiusReport(
            total_evaluated=batch_size,
            certified_count=certified_count,
            abstained_count=abstained_count,
            certification_rate=cert_rate,
            mean_certified_l2_radius=mean_r,
            median_certified_l2_radius=median_r,
            max_certified_l2_radius=max_r,
            noise_sigma=self.noise_sigma,
            confidence_alpha=self.confidence_alpha,
            num_mc_samples=self.num_mc_samples,
            per_sample_radii=radii,
            smoothed_predictions=smoothed_preds,
        )

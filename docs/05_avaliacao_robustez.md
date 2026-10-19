# 5. Calibrated Risk Evaluation, Conformal Abstention & Strategic Robustness

Deploying a foundation model in regulated financial workflows requires moving beyond raw accuracy or single-threshold metrics. In alignment with **Federal Reserve / OCC SR 11-7 & SR 26-2 Model Risk Management** and **Gartner AI TRiSM**, [`src/evaluation/`](../src/evaluation/) implements a four-pillar evaluation and governance harness.

---

## Pillar 1: Imbalanced Discrimination Metrics (`src/evaluation/metrics.py`)

Under severe class imbalance ($1\%$–$5\%$ fraud prevalence), a trivial classifier that predicts "legitimate" on every transaction achieves $95\%$–$99\%$ accuracy while catching zero fraud. We enforce four complementary discrimination metrics:

1. **ROC-AUC (`calculate_auc`):** Threshold-independent rank concordance across all positive-negative pairs.
2. **PR-AUC / Average Precision (`calculate_pr_auc`):** Area under the Precision-Recall curve. As demonstrated by Saito & Rehmsmeier (*PLOS ONE*, 2015), PR-AUC is significantly more informative than ROC-AUC under severe imbalance because it directly reflects false-positive alert inflation.
3. **Recall at Fixed False Positive Rate (`calculate_recall_at_fpr`):** Measures the fraction of total fraud captured when the false-positive rate is capped at an operational budget (e.g., `max_fpr = 0.05` or `0.01`), matching enterprise fraud operations KPIs (BCG / McKinsey).
4. **Flexible-Threshold F1, Precision, Recall & Specificity (`calculate_f1_flexible`, `calculate_precision_recall`):** Evaluates operating-point performance under business-tuned decision thresholds.

---

## Pillar 2: Epistemic Calibration Auditing

In credit underwriting and transaction scoring, predicted probabilities $\hat{p}_i = \sigma(\text{logit}_i)$ are consumed downstream by expected-loss calculators (`EL = PD * LGD * EAD`). Uncalibrated probabilities distort capital allocation. We audit calibration via:

* **Brier Score (`calculate_brier_score`):** Mean squared probability error $\frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2$.
* **Expected Calibration Error (`calculate_ece`):** Partitions predictions into $M$ confidence bins and measures the weighted gap between empirical positive rates and mean predicted probabilities (Guo et al., *ICML* 2017).

---

## Pillar 3: Split-Conformal Risk Abstention (`ConformalRiskAbstainer`)

Even a well-calibrated model encounters ambiguous out-of-distribution trajectories where automated approval or rejection carries unacceptable risk. Following **Conformal Prediction** (Angelopoulos & Bates, 2023) and **Gartner AI TRiSM** runtime enforcement, `ConformalRiskAbstainer` calibrates class-conditional finite-sample probability quantiles $[q_{\text{low}}, q_{\text{high}}]$ at significance level $\alpha$:

$$\text{Decision}(\hat{p}_i) = \begin{cases} 0 \text{ (Auto-Approve)} & \text{if } \hat{p}_i \le q_{\text{low}} \\ 1 \text{ (Auto-Block)} & \text{if } \hat{p}_i \ge q_{\text{high}} \\ -1 \text{ (Abstain }\rightarrow\text{ Analyst Review)} & \text{if } \hat{p}_i \in (q_{\text{low}}, q_{\text{high}}) \end{cases}$$

`RiskMetricsEvaluator` reports `conformal_coverage`, `conformal_abstention_rate`, `conformal_selective_accuracy`, and `conformal_selective_f1` on the automated subset.

---

## Pillar 4: Strategic Shift & Adversarial Stress Testing (`strategic_shift_tester.py`)

Unlike static computer vision tasks, financial fraud involves adaptive adversaries who intentionally manipulate transaction amounts (e.g., smurfing/structuring) to evade detection thresholds. `StrategicShiftTester` evaluates model degradation under two regimes:

1. **Natural Distribution Drift (`apply_random_shift`):** Injects Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ into continuous transaction amounts to simulate inflation or seasonal spending shifts.
2. **White-Box FGSM Evasion (`apply_gradient_adversarial_shift`):** Computes the gradient of the classification loss with respect to the input transaction values and perturbs non-padded amounts in the worst-case direction (Goodfellow et al., *ICLR* 2015):
   $$\mathbf{v}_{\text{adv}} = \mathbf{v} + \epsilon \cdot \text{sign}\!\left(\nabla_{\mathbf{v}} \mathcal{L}_{\text{BCE}}(f_\theta(\mathbf{k}, \mathbf{v}, \mathbf{t}), y)\right)$$
3. **Rapid Emergency Adaptation via LoRA (`inject_lora_adaptation`):** When novel attack patterns emerge in production, Low-Rank Adaptation (Hu et al., *ICLR* 2022) freezes the pre-trained backbone and injects low-rank matrices ($\sim 1\%$ trainable parameters) into attention projections for rapid hot-fix fine-tuning without catastrophic forgetting.

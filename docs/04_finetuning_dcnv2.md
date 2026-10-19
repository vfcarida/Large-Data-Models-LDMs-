# 4. Supervised Fine-Tuning & Parallel DCNv2 Static-Dynamic Fusion

After self-supervised pre-training, the `MMTTTransformerEncoder` produces rich behavioral representations of a customer's transaction history. However, production financial risk systems also rely on static, cross-sectional tabular attributes:

* **Credit Bureau Scores** (e.g., FICO / bureau risk ratings)
* **Verified Annual Income & Debt-to-Income Proxies**
* **Customer Age & Account Tenure**

[`src/models/joint_fusion_network.py`](../src/models/joint_fusion_network.py) implements `EndToEndFusionLayer`, combining deep sequential embeddings with standardized static features via a **Parallel Deep & Cross Network v2 (DCNv2)** (Wang et al., Google Research, *WWW* 2021, `arXiv:2008.13535`).

---

## Why Simple Concatenation + MLP Is Suboptimal

Concatenating the pooled Transformer embedding $\mathbf{z}_{\text{seq}} \in \mathbb{R}^{d_{\text{seq}}}$ with static tabular features $\mathbf{x}_{\text{tab}} \in \mathbb{R}^{d_{\text{tab}}}$ and passing them into a standard Multi-Layer Perceptron (MLP) learns only *implicit* feature interactions. Discovering multiplicative risk rules—such as *"low bureau score $\times$ rapid high-value crypto transfer burst"*—requires excessive depth and large labeled sample sizes.

---

## Parallel Deep & Cross Network v2 (`CrossNetworkV2`)

Let $\mathbf{x}_0 = [\mathbf{z}_{\text{seq}} \;\|\; \text{LayerNorm}(\mathbf{x}_{\text{tab}})] \in \mathbb{R}^{d}$ denote the anchored joint input vector, where $d = d_{\text{seq}} + d_{\text{tab}}$.

### 1. Explicit Polynomial Cross Network (`CrossNetworkV2`)
Each cross layer $l \in \{0, \dots, L-1\}$ applies an explicit Hadamard product with the original anchored input $\mathbf{x}_0$:

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l$$

* **Layer 1** computes bounded degree-2 (quadratic) interactions between every sequence dimension and every static bureau feature.
* **Layer $L$** computes bounded degree-$(L+1)$ polynomial interactions with residual gradient highways.
* **Low-Rank Factorization Option:** When `low_rank = r` ($r \ll d$) is specified, the interaction matrix is parameterized as $\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^\top$ with $\mathbf{U}_l, \mathbf{V}_l \in \mathbb{R}^{d \times r}$, reducing per-layer complexity from $\mathcal{O}(d^2)$ to $\mathcal{O}(2dr)$.

### 2. Parallel Deep MLP (`deep_net`)
In parallel with the Cross Network, $\mathbf{x}_0$ is transformed through a two-stage dense network with `LayerNorm`, `GELU` activations, and dropout (`deep_dropout`). Using `LayerNorm` instead of `BatchNorm1d` ensures deterministic stability even during single-transaction real-time inference (`batch_size = 1`).

### 3. Final Logit Projection
The outputs of the Cross Network and Deep Network are concatenated and projected to a scalar binary classification logit:

$$\hat{y}_{\text{logit}} = \mathbf{w}_{\text{out}}^\top [\mathbf{x}_L \;\|\; \text{DeepMLP}(\mathbf{x}_0)] + b_{\text{out}}$$

---

## Differential Learning Rates & Imbalanced Loss Weighting

In [`LDMFineTuningModule`](../src/training/lightning_module.py):
1. **Differential Learning Rates:** The pre-trained `MMTTTransformerEncoder` is updated at $0.25\times$ the learning rate of the `EndToEndFusionLayer`, preserving self-supervised temporal representations while rapidly adapting the fusion head.
2. **Cost-Sensitive Binary Cross-Entropy (`pos_weight`):** Because fraud affects only $1\%$–$5\%$ of accounts, unweighted cross-entropy degenerates toward predicting the majority class. Applying `pos_weight` scales positive-class gradients proportionally to the economic cost of missed fraud.

---

**Next Step:** Read [05_avaliacao_robustez.md](05_avaliacao_robustez.md) to explore our imbalanced discrimination metrics, calibration auditing, conformal abstention guardrails, and FGSM adversarial stress testing.

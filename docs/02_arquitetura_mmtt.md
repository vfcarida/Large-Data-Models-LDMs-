# 2. MMTT Architecture & Periodic Numerical Embeddings

To process heterogeneous financial transaction histories with a Transformer, the model must jointly represent discrete categories, heavy-tailed continuous amounts, and irregular temporal intervals without exploding sequence length.

The `MMTTTransformerEncoder` in [`src/models/transformer_backbone.py`](../src/models/transformer_backbone.py) implements a **Multi-Modal-Temporal-Tabular (MMTT)** architecture grounded in *TransactionGPT* (Visa Research, 2025), *PRAGMA* (Revolut/NVIDIA, 2026), and *Periodic Numerical Embeddings* (Gorishniy et al., *NeurIPS* 2022).

---

## Key-Value-Time (KVT) Tokenization

Each transaction $t_i = (k_i, v_i, \Delta t_i)$ is decomposed into three orthogonal modalities:

### 1. Key Modality (`packed_keys` — Discrete Categorical)
* **Domain Semantics:** Merchant Category Codes (MCCs), payment channels, or transaction types.
* **Encoding:** Mapped to a discrete vocabulary $\{0, \dots, V-1\}$ and embedded via `nn.Embedding(vocab_size, hidden_size)`:
  $$\mathbf{e}_i^{(K)} = \text{Embed}_{\text{key}}(k_i) \in \mathbb{R}^d$$

### 2. Value Modality (`packed_values` — Continuous Numerical via PLR)
* **Domain Semantics:** Log-normalized transaction amounts (`amount`), which exhibit heavy tails and sharp threshold semantics (e.g., structuring just below reporting limits).
* **Periodic Numerical Embedding (`PeriodicNumericalEmbedding`):** Instead of a naive linear projection `nn.Linear(1, d)`—which suffers from spectral bias—we implement the **PLR (Periodic + Linear + Activation)** module (Gorishniy et al., *NeurIPS* 2022, `arXiv:2203.05556`):
  $$\mathbf{p}(v_i) = \left[v_i, \;\sin(2\pi \mathbf{c} v_i), \;\cos(2\pi \mathbf{c} v_i)\right] \in \mathbb{R}^{2m + 1}$$
  $$\mathbf{e}_i^{(V)} = \mathbf{W}_2 \,\text{GELU}(\mathbf{W}_1 \mathbf{p}(v_i) + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^d$$
  where $\mathbf{c} \in \mathbb{R}^m$ is a vector of trainable frequency parameters. Concatenating the raw scalar $v_i$ with sinusoidal frequencies preserves global monotonicity while resolving fine-grained local thresholds.

### 3. Time Modality (`packed_times` — Inter-Event Temporal Rhythm)
* **Domain Semantics:** Quantized time elapsed since the user's previous transaction ($\Delta t_i = \tau_i - \tau_{i-1}$).
* **Encoding:** Quantized into $10,000$ discrete temporal bins and mapped via `nn.Embedding(10000, hidden_size)`:
  $$\mathbf{e}_i^{(T)} = \text{Embed}_{\text{time}}(\Delta t_i) \in \mathbb{R}^d$$
  This captures behavioral cadence—distinguishing normal daily/weekly shopping intervals from automated sub-second velocity bursts.

---

## Additive Modality Fusion & Sequence Positional Encoding

Rather than concatenating the three modality vectors (which would triple the token dimension or triple the sequence length, increasing self-attention cost by $9\times$), we fuse them additively alongside **Sinusoidal Sequence Positional Encoding** (`SinusoidalPositionalEncoding`):

$$\mathbf{h}_i^{(0)} = \text{LayerNorm}\!\left(\mathbf{e}_i^{(K)} + \mathbf{e}_i^{(V)} + \mathbf{e}_i^{(T)} + \gamma \,\mathbf{p}_{\text{seq}}(i)\right)$$

Why include $\mathbf{p}_{\text{seq}}(i)$ in addition to $\mathbf{e}_i^{(T)}$? Because $\Delta t_i$ measures *inter-arrival intervals* rather than absolute sequence order; adding scaled sinusoidal sequence positions breaks permutation equivariance when two sub-trajectories share similar $\Delta t$ values.

---

## Virtual Token Layer & Padding Mask Guardrails

1. **Learnable Virtual Tokens (`VirtualTokenLayer`):**
   A small set of learnable continuous prompt vectors $[\mathbf{v}_1, \dots, \mathbf{v}_k]$ is prepended to every sequence:
   $$\mathbf{X} = [\mathbf{v}_1, \dots, \mathbf{v}_k, \mathbf{h}_1^{(0)}, \dots, \mathbf{h}_S^{(0)}] \in \mathbb{R}^{(k + S) \times d}$$
   These virtual tokens attend across the entire trajectory and aggregate global behavioral context.
2. **Strict Padding Isolation (`key_padding_mask`):**
   Variable-length user histories are collated with a boolean `key_padding_mask` (`True` = padded position). The mask is prepended with `False` for virtual tokens and passed to both `nn.TransformerEncoder` and the downstream sequence pooler (`cls`, `mean`, `max`, or `attentive`), guaranteeing that padded positions have zero influence on valid representations.

---

**Next Step:** Read [03_pretraining_limix.md](03_pretraining_limix.md) to see how the MMTT backbone is pre-trained without human labels.

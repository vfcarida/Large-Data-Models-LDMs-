# Referências Científicas e Bibliografia

As implementações neste repositório não são aleatórias; elas são fundamentadas na mais recente literatura acadêmica (2024-2026) sobre o uso de Foundation Models para dados estruturados.

Recomendamos a leitura dos seguintes papers para aprofundamento:

### 1. Modelagem de Transações Financeiras (KVT e 3D-Transformers)
*   **TransactionGPT: Foundation Models for Payment Trajectories** (Visa Research, 2025).
    *   *Resumo*: Propõe a arquitetura 3D-Transformer para dados MMTT (Multi-Modal-Temporal-Tabular) e a introdução da Camada de Tokens Virtuais para fusão eficiente de metadados sem explodir o custo quadrático da atenção.
*   **PRAGMA: A Self-Supervised Foundation Model for Banking Events** (Revolut/NVIDIA, 2025).
    *   *Resumo*: Introduz a tokenização Key-Value-Time (KVT) para eventos discretos, provando que modelagem bidirecional em dados bancários massivos supera drasticamente os tradicionais Gradient Boosted Trees em tarefas downstream.

### 2. Tabular Foundation Models e Pré-Treinamento (LimiX e TabPFN)
*   **LimiX: Masked Joint-Distribution Pre-training for Structured Data** (2024).
    *   *Resumo*: Demonstra que o comportamento tabular/estruturado não é puramente temporal, mas uma distribuição conjunta. Introduz o *Heterogeneous Mask Schedule* (usado no nosso `objectives.py`) para evitar a dedução trivial de blocos temporais.
*   **TabPFN-2.5: Prior-Data Fitted Networks for Tabular Data** (2025/2026).
    *   *Resumo*: Escala o In-Context Learning para dados tabulares, mostrando como Transformers podem aproximar inferência Bayesiana instantaneamente, sem precisar de fine-tuning explícito por épocas.
*   **Schema-1: A Data Language Model for Native Cell Processing** (Schema Labs, 2026).
    *   *Resumo*: Discute as limitações de serializar tabelas como texto para LLMs e propõe um processamento celular nativo. Nossa normalização (Z-score contínuo) segue o princípio de não quebrar a geometria matemática dos números, defendida por eles.

### 3. Fusão Tabular Explicita
*   **DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems** (Google Research, 2021).
    *   *Resumo*: Explica matematicamente por que a concatenação simples não é suficiente para dados estruturados. A rede aprende interações de *features* explícitas em um grau polinomial controlado (implementada no `joint_fusion_network.py`).

### 4. Robustez e Avaliação em Séries Imbalanceadas
*   **The Precision-Recall Plot Is More Informative than the ROC Plot when Evaluating Binary Classifiers on Imbalanced Datasets** (Saito & Rehmsmeier, 2015).
    *   *Resumo*: Fundamenta nossa decisão de prover o F1-Score com *Flexible Threshold* para detecção de fraudes.
*   **Explaining and Harnessing Adversarial Examples** (Goodfellow et al., 2015).
    *   *Resumo*: O paper original sobre o Fast Gradient Sign Method (FGSM), o qual adaptamos no `strategic_shift_tester.py` para simular fraudadores que manipulam intencionalmente transações contínuas para escapar do modelo.
*   **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2022).
    *   *Resumo*: A base para a nossa função de `inject_lora_adaptation()`, que permite atualização emergencial de parâmetros sob ataque estratégico sem "esquecimento catastrófico".

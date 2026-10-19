"""
In-Context Prior-Data Fitted Network (PFN) Classification Head (TabPFN v2).

============================================================================
PURPOSE
============================================================================
Implements an In-Context Learning (ICL) classification head inspired by
TabPFN v2 (Hollmann et al., Nature 2025) over joint MMTT + Static Tabular
representations.

When launching a new payment rail, card product, or regional market with
fewer than 1,000 labeled fraud examples, gradient-based fine-tuning can
overfit. `InContextPFNHead` enables rapid cold-start adaptation in a single
forward pass by conditioning query predictions directly on a support set of
labeled entity representations `D_sup = {(x_i, y_i)}_{i=1}^{N_s}`:
  1. Embeds support features + ground-truth labels into support tokens.
  2. Embeds query features + a learnable `[QUERY]` token into query tokens.
  3. Applies an asymmetric attention mask where:
       - Support tokens attend to all Support tokens (S <-> S).
       - Query tokens attend strictly to Support tokens (Q -> S) and oneself,
         blocking Query-to-Query interaction so predictions are strictly
         invariant to query batch size and ordering.
  4. Outputs calibrated posterior logits `p(y_q | x_q, D_sup)`.

============================================================================
REFERENCES
============================================================================
  - Hollmann, N., Müller, S., Purucker, L., et al. (2025): "Accurate
    predictions on small data with a tabular foundation model" (TabPFN v2,
    Nature 637:319-326, DOI: 10.1038/s41586-024-08328-6)
  - Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F.
    (2022): "Transformers Can Do Bayesian Inference" (ICLR 2022)
============================================================================
"""

from typing import Optional
import torch
import torch.nn as nn


class InContextPFNHead(nn.Module):
    """
    TabPFN-inspired In-Context Posterior Approximation Head.

    Args:
        input_dim: Dimension of input entity representations (e.g., `hidden_size + tabular_dim`).
        d_model: Internal Transformer dimension for in-context attention.
        num_heads: Number of attention heads.
        num_layers: Number of in-context Transformer layers.
        num_classes: Number of output logits (default: 1 for binary fraud logits).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0 or d_model <= 0:
            raise ValueError("input_dim and d_model must be positive.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        self.feature_proj = nn.Linear(input_dim, d_model)
        # Binary label embedding for support set {0, 1}
        self.support_label_emb = nn.Embedding(2, d_model)
        # Learnable query token embedding for unlabeled query instances
        self.query_token_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.icl_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.posterior_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    @staticmethod
    def build_in_context_mask(
        num_support: int,
        num_query: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Constructs the boolean attention mask `(N_s + N_q, N_s + N_q)` for
        `nn.TransformerEncoder` (`True` = blocked/masked position):
          - Support tokens (`0 .. N_s-1`) attend exclusively to Support tokens.
          - Query tokens (`N_s .. N_s+N_q-1`) attend to all Support tokens and
            their own diagonal self-position, never to other Query tokens.

        Args:
            num_support: Number of support instances `N_s`.
            num_query: Number of query instances `N_q`.
            device: Target PyTorch device.

        Returns:
            Boolean attention mask of shape `(N_s + N_q, N_s + N_q)`.
        """
        total = num_support + num_query
        # Initialize all positions as blocked (True)
        mask = torch.ones((total, total), dtype=torch.bool, device=device)

        # Allow all rows (both Support and Query) to attend to Support columns [0 : num_support]
        mask[:, :num_support] = False

        # Allow each Query token to attend to its own diagonal position
        if num_query > 0:
            query_idx = torch.arange(num_support, total, device=device)
            mask[query_idx, query_idx] = False

        return mask

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes posterior classification logits for `query_features` conditioned
        on `(support_features, support_labels)`.

        Args:
            query_features: Query representations `(N_q, input_dim)`.
            support_features: Labeled support representations `(N_s, input_dim)`.
            support_labels: Binary support labels `(N_s,)` or `(N_s, 1)` in `{0, 1}`.

        Returns:
            Posterior logits for query instances of shape `(N_q, num_classes)`.
        """
        num_support = support_features.size(0)
        num_query = query_features.size(0)
        if num_support == 0:
            raise ValueError("support_features must contain at least 1 support example.")

        flat_labels = support_labels.view(-1).long().clamp(0, 1)

        sup_tokens = self.input_norm(
            self.feature_proj(support_features) + self.support_label_emb(flat_labels)
        )  # (N_s, d_model)

        qry_tokens = self.input_norm(
            self.feature_proj(query_features)
            + self.query_token_emb.expand(num_query, -1)
        )  # (N_q, d_model)

        # Concatenate into a single sequence of length (1, N_s + N_q, d_model)
        combined_seq = torch.cat([sup_tokens, qry_tokens], dim=0).unsqueeze(0)

        attn_mask = self.build_in_context_mask(
            num_support=num_support,
            num_query=num_query,
            device=query_features.device,
        )

        contextualized = self.icl_transformer(
            combined_seq, mask=attn_mask
        ).squeeze(0)  # (N_s + N_q, d_model)

        query_contextualized = contextualized[num_support:]  # (N_q, d_model)
        logits = self.posterior_head(query_contextualized)  # (N_q, num_classes)
        return logits

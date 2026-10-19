"""
Retrieval-Augmented Tabular & Behavioral Attention (TabR).

============================================================================
PURPOSE
============================================================================
Implements a differentiable k-Nearest-Neighbor (kNN) context retrieval and
attention module inspired by TabR (Gorishniy et al., ICLR 2024).

In financial fraud detection and credit risk modeling, an entity's risk is
strongly informed by analogous historical trajectories (e.g., confirmed
account takeover patterns or velocity bursts). `TabRRetrievalModule`
enables the model to attend directly to the top-k nearest historical entity
representations and their known risk outcomes:
  1. Projects query and candidate embeddings into a normalized metric space.
  2. Computes scaled negative squared L2 distances:
       s(q, i) = -||k_q - k_i||_2^2 / sqrt(d)
  3. Strictly masks self-retrieval (i == q) during in-batch training to
     prevent target leakage.
  4. Aggregates directional differences T(k_q - k_i) and neighbor label
     embeddings e_y(y_i) weighted by softmax attention over the top-k
     neighbors using a Pre-Norm LayerScale residual connection.
  5. Maintains a persistent registered memory buffer of historical embeddings
     and labels so single-sample or batch inference (`eval()` mode) can
     retrieve context from past training trajectories automatically.

============================================================================
REFERENCES
============================================================================
  - Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A.,
    & Babenko, A. (2024): "TabR: Tabular Deep Learning Meets Nearest
    Neighbors in 2023" (ICLR 2024, arXiv:2307.14338)
============================================================================
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabRRetrievalModule(nn.Module):
    """
    Differentiable k-Nearest-Neighbor Retrieval-Augmented Attention (TabR).

    Args:
        embed_dim: Dimension of input/output entity representations.
        top_k: Number of nearest neighbors to retrieve per query.
        memory_size: Capacity of the persistent historical memory buffer.
        dropout: Dropout probability applied inside the directional value MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        top_k: int = 8,
        memory_size: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if memory_size < top_k:
            raise ValueError(
                f"memory_size ({memory_size}) must be >= top_k ({top_k})"
            )

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.memory_size = memory_size
        self.scale = math.sqrt(float(embed_dim))
        self._active_mem_count: int = 0

        # Shared key projection with Pre-Norm LayerNorm (Gorishniy et al., ICLR 2024)
        self.norm = nn.LayerNorm(embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Neighbor target label embedding: maps binary outcome {0, 1} + unknown {2}
        self.label_embedding = nn.Embedding(3, embed_dim)

        # Directional value transformation T(k_q - k_i)
        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

        # Output projection with Pre-Norm LayerScale residual gate
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

        # Persistent circular memory buffer for inference-time retrieval
        self.register_buffer("memory_keys", torch.zeros(memory_size, embed_dim))
        self.register_buffer(
            "memory_labels", torch.full((memory_size,), 2, dtype=torch.long)
        )
        self.register_buffer(
            "memory_count", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "memory_ptr", torch.tensor(0, dtype=torch.long)
        )

    @torch.no_grad()
    def update_memory(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Updates the circular memory bank with detached entity embeddings and labels.

        Args:
            embeddings: Tensor of shape `(B, embed_dim)`.
            labels: Optional binary label tensor of shape `(B,)` or `(B, 1)`.
        """
        batch_size = int(embeddings.shape[0])
        if batch_size == 0:
            return

        detached_emb = embeddings.detach()
        if labels is not None:
            flat_labels = labels.detach().view(-1).long().clamp(0, 1)
        else:
            flat_labels = torch.full(
                (batch_size,), 2, dtype=torch.long, device=embeddings.device
            )

        if batch_size >= self.memory_size:
            self.memory_keys.copy_(detached_emb[-self.memory_size :])
            self.memory_labels.copy_(flat_labels[-self.memory_size :])
            self.memory_count.fill_(self.memory_size)
            self.memory_ptr.fill_(0)
            self._active_mem_count = self.memory_size
            return

        ptr = int(self.memory_ptr.item())
        end_ptr = ptr + batch_size

        if end_ptr <= self.memory_size:
            self.memory_keys[ptr:end_ptr] = detached_emb
            self.memory_labels[ptr:] if False else None
            self.memory_labels[ptr:end_ptr] = flat_labels
        else:
            first_part = self.memory_size - ptr
            self.memory_keys[ptr:] = detached_emb[:first_part]
            self.memory_labels[ptr:] = flat_labels[:first_part]
            second_part = batch_size - first_part
            self.memory_keys[:second_part] = detached_emb[first_part:]
            self.memory_labels[:second_part] = flat_labels[first_part:]

        self.memory_ptr.fill_(end_ptr % self.memory_size)
        new_count = min(self.memory_size, int(self.memory_count.item()) + batch_size)
        self.memory_count.fill_(new_count)
        self._active_mem_count = new_count

    def forward(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: Optional[torch.Tensor] = None,
        candidate_labels: Optional[torch.Tensor] = None,
        update_bank: bool = False,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs differentiable top-k neighbor retrieval and Pre-Norm context fusion.

        Args:
            query_embeddings: Query representations of shape `(B, embed_dim)`.
            candidate_embeddings: Optional candidate pool `(M, embed_dim)`.
            candidate_labels: Optional candidate labels `(M,)` in `{0, 1}`.
            update_bank: If True and `self.training` is True, appends `query_embeddings`
                and `query_labels` to the persistent memory bank after retrieval.
            query_labels: Optional ground-truth labels `(B,)` for updating the bank.

        Returns:
            Tuple `(enriched_embeddings, attention_weights)`:
              - `enriched_embeddings`: Tensor of shape `(B, embed_dim)`
              - `attention_weights`: Top-k attention distribution `(B, actual_k)`
        """
        batch_size = query_embeddings.size(0)
        q_norm = self.norm(query_embeddings)
        k_q = self.key_proj(q_norm)  # (B, D)

        self_mask_diagonal = False
        if not torch.jit.is_tracing():
            self._active_mem_count = int(self.memory_count.item())
        active_mem = self._active_mem_count

        if candidate_embeddings is not None:
            cand_emb = candidate_embeddings
            num_candidates = int(cand_emb.shape[0])
            if candidate_labels is not None:
                cand_y = candidate_labels.view(-1).long().clamp(0, 2)
            else:
                cand_y = torch.full(
                    (num_candidates,),
                    2,
                    dtype=torch.long,
                    device=query_embeddings.device,
                )
        elif active_mem >= 2:
            cand_emb = self.memory_keys[:active_mem].clone().detach()
            cand_y = self.memory_labels[:active_mem].clone().detach()
            num_candidates = active_mem
        elif not self.training:
            # In eval/inference mode before memory population, attend to the initial
            # top_k prior slots identically in both eager and TorchScript execution
            fallback_k = min(self.top_k, self.memory_size)
            cand_emb = self.memory_keys[:fallback_k].clone().detach()
            cand_y = self.memory_labels[:fallback_k].clone().detach()
            num_candidates = fallback_k
        else:
            cand_emb = query_embeddings.detach()
            num_candidates = int(query_embeddings.shape[0])
            if query_labels is not None:
                cand_y = query_labels.detach().view(-1).long().clamp(0, 1)
            else:
                cand_y = torch.full(
                    (num_candidates,), 2, dtype=torch.long, device=query_embeddings.device
                )
            self_mask_diagonal = True

        if self.training and (
            num_candidates == 0 or (self_mask_diagonal and num_candidates <= 1)
        ):
            if update_bank:
                self.update_memory(query_embeddings, query_labels)
            uniform_attn = torch.ones(
                (batch_size, 1),
                dtype=query_embeddings.dtype,
                device=query_embeddings.device,
            )
            return query_embeddings, uniform_attn

        k_cand = self.key_proj(self.norm(cand_emb))  # (M, D)

        dists_sq = torch.cdist(k_q, k_cand, p=2.0).pow(2)
        sim_logits = -dists_sq / self.scale

        if self_mask_diagonal and self.training and int(batch_size) == num_candidates:
            diag_mask = torch.eye(
                int(batch_size), dtype=torch.bool, device=query_embeddings.device
            )
            sim_logits = sim_logits.masked_fill(diag_mask, float("-inf"))

        actual_k = min(
            self.top_k,
            num_candidates - (1 if self_mask_diagonal else 0),
        )
        actual_k = max(1, actual_k)

        topk_logits, topk_indices = torch.topk(sim_logits, k=actual_k, dim=-1)  # (B, K)
        attn_weights = F.softmax(topk_logits, dim=-1)  # (B, K)

        gathered_keys = k_cand[topk_indices]
        gathered_labels = cand_y[topk_indices]

        key_diff = k_q.unsqueeze(1) - gathered_keys
        directional_values = self.value_mlp(key_diff)
        label_values = self.label_embedding(gathered_labels)

        neighbor_values = directional_values + label_values

        retrieved_context = torch.bmm(
            attn_weights.unsqueeze(1), neighbor_values
        ).squeeze(1)

        # Pre-Norm residual update preserves the scale of `query_embeddings`
        enriched = query_embeddings + self.residual_scale * self.out_proj(retrieved_context)

        if update_bank and self.training:
            self.update_memory(query_embeddings, query_labels)

        return enriched, attn_weights

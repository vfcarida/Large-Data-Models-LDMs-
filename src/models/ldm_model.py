"""
Large Data Model (LDM) — Complete End-to-End Foundation Model.

============================================================================
PURPOSE
============================================================================
Unifies the MMTT Transformer Encoder backbone, multi-modal self-supervised
reconstruction heads, Relational Entity-Merchant Graph Message Passing
(`RelBench` / `RelationalEntityGNN`), Retrieval-Augmented Behavioral
Attention (`TabR` / `TabRRetrievalModule`), the DCNv2 static-dynamic fusion
layer, and an In-Context Prior-Data Fitted Network head (`TabPFN v2` /
`InContextPFNHead`) into a cohesive foundation model supporting three
operational modes:

1. PRE-TRAINING (Self-Supervised Learning):
   - Input: Unlabeled transaction sequences (Key-Value-Time)
   - Objectives:
     a) Masked Joint-Distribution Reconstruction (LimiX / PRAGMA): predicts
        masked merchant categories (`keys`), continuous amounts (`values`),
        and temporal rhythms (`times`) from bidirectional context.
     b) Contrastive Event Sequence Learning (CoLES): aligns pooled user
        subsequence views in embedding space to prevent representation collapse.

2. FINE-TUNING (Supervised Risk Classification):
   - Input: Transaction sequences + static bureau features + fraud labels
   - Architecture:
     a) MMTT Transformer Encoder with Periodic Numerical Embeddings (PLR)
     b) Bipartite User-Merchant Graph Message Passing (`RelationalEntityGNN`)
     c) Differentiable kNN Behavioral Context Retrieval (`TabRRetrievalModule`)
     d) Parallel Deep & Cross Network v2 (`EndToEndFusionLayer`)

3. IN-CONTEXT COLD-START INFERENCE (Prior-Data Fitted Network / TabPFN v2):
   - Input: Support set of labeled entities + unlabeled query entities
   - Performs single-forward-pass posterior approximation via asymmetric
     support-to-query attention (`InContextPFNHead`).

============================================================================
REFERENCES
============================================================================
  - Hollmann et al. (2025): "Accurate predictions on small data with a
    tabular foundation model" (TabPFN v2, Nature 637:319-326)
  - Dou et al. (2025): "TransactionGPT: Foundation Models for Payment
    Trajectories" (Visa Research, arXiv:2511.08939)
  - Ostroukhov et al. (2026): "PRAGMA: A Self-Supervised Foundation Model
    for Banking Events" (Revolut/NVIDIA, arXiv:2604.08649)
  - Gorishniy et al. (2024): "TabR: Tabular Deep Learning Meets Nearest
    Neighbors in 2023" (ICLR 2024, arXiv:2307.14338)
  - Robinson et al. (2024): "RelBench: A Benchmark for Deep Learning on
    Relational Databases" (NeurIPS 2024, arXiv:2407.20060)
  - Babaev et al. (2022): "CoLES: Contrastive Learning for Event Sequences
    with Self-Supervision" (SIGMOD 2022, arXiv:2002.08232)
============================================================================
"""

from typing import Any, Dict, Literal, Optional
import logging

import torch
import torch.nn as nn

from src.models.transformer_backbone import MMTTTransformerEncoder
from src.models.ssm_backbone import HybridSSMTransformerEncoder
from src.models.joint_fusion_network import EndToEndFusionLayer
from src.models.retrieval_attention import TabRRetrievalModule
from src.models.relational_gnn import RelationalEntityGNN
from src.models.in_context_pfn import InContextPFNHead
from src.models.multi_task_moe import MultiGateMixtureOfExpertsHead
from src.models.causal_uplift import CausalDragonNetHead
from src.models.hypergraph_contagion import TemporalHypergraphContagionModule
from src.models.flow_matching import ConditionalTrajectoryFlowMatcher

logger = logging.getLogger(__name__)


class LargeDataModel(nn.Module):
    """
    Complete Large Data Model supporting self-supervised pre-training,
    relational-retrieval-augmented supervised fine-tuning, in-context
    prior-data fitted inference, multi-task Mixture-of-Experts (`MMoE`),
    Causal Counterfactual Uplift (`DragonNet`), and Temporal Hypergraph
    Contagion (`HGNN`) decisioning.

    Args:
        vocab_size: Number of unique categorical tokens (MCC codes).
        hidden_size: Transformer / SSM hidden dimension (`d_model`).
        num_layers: Number of sequence encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        tabular_feature_size: Number of static tabular features for fusion.
        dcn_layers: Number of DCNv2 cross layers.
        use_virtual_tokens: Enable virtual token global conditioning.
        num_virtual_tokens: Number of virtual tokens.
        pooling: Sequence pooling strategy (`"cls"`, `"mean"`, `"max"`, `"attentive"`).
        use_relational_gnn: Enable bipartite user-merchant GNN message passing (`RelBench`).
        gnn_layers: Number of bipartite GNN layers.
        use_tabr_retrieval: Enable differentiable kNN context retrieval (`TabR`).
        tabr_top_k: Number of nearest neighbors retrieved in `TabRRetrievalModule`.
        tabr_memory_size: Capacity of the persistent historical memory bank.
        backbone_type: Sequence encoder architecture (`"mmtt"`, `"hybrid_ssm"`, or `"ssm"`).
        num_experts: Number of shared experts in `MultiGateMixtureOfExpertsHead`.
    """

    def __init__(
        self,
        vocab_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        tabular_feature_size: int = 3,
        dcn_layers: int = 3,
        use_virtual_tokens: bool = True,
        num_virtual_tokens: int = 2,
        pooling: Literal["cls", "mean", "max", "attentive"] = "mean",
        use_relational_gnn: bool = True,
        gnn_layers: int = 2,
        use_tabr_retrieval: bool = True,
        tabr_top_k: int = 8,
        tabr_memory_size: int = 512,
        backbone_type: Literal["mmtt", "hybrid_ssm", "ssm"] = "mmtt",
        num_experts: int = 4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.tabular_feature_size = tabular_feature_size
        self.dcn_layers = dcn_layers
        self.use_virtual_tokens = use_virtual_tokens
        self.num_virtual_tokens = num_virtual_tokens
        self.pooling = pooling
        self.use_relational_gnn = use_relational_gnn
        self.gnn_layers = gnn_layers
        self.use_tabr_retrieval = use_tabr_retrieval
        self.tabr_top_k = tabr_top_k
        self.tabr_memory_size = tabr_memory_size
        self.backbone_type = backbone_type
        self.num_experts = num_experts

        # 1. Shared Sequence Backbone (MMTT Transformer or Hybrid Selective SSM)
        if backbone_type == "mmtt":
            self.encoder: nn.Module = MMTTTransformerEncoder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_virtual_tokens=use_virtual_tokens,
                num_virtual_tokens=num_virtual_tokens,
                pooling=pooling,
            )
        elif backbone_type in ("hybrid_ssm", "ssm"):
            self.encoder = HybridSSMTransformerEncoder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_virtual_tokens=use_virtual_tokens,
                num_virtual_tokens=num_virtual_tokens,
                pooling=pooling,
                mode=backbone_type,
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        # 2. Self-Supervised Multi-Modal Reconstruction Heads (PRAGMA / LimiX)
        self.ssl_key_head = nn.Linear(hidden_size, vocab_size)
        self.ssl_value_head = nn.Linear(hidden_size, 1)
        self.ssl_time_head = nn.Linear(hidden_size, 1)
        self.ssl_latent_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # 3. Supervised Fine-Tuning DCNv2 Fusion Head
        self.fusion_head = EndToEndFusionLayer(
            transformer_hidden_size=hidden_size,
            tabular_feature_size=tabular_feature_size,
            dcn_layers=dcn_layers,
        )

        # 4. Horizon Track 2: Relational Entity-Merchant Graph Message Passing (RelBench)
        self.relational_gnn: Optional[RelationalEntityGNN] = (
            RelationalEntityGNN(
                vocab_size=vocab_size,
                embed_dim=hidden_size,
                num_layers=gnn_layers,
                dropout=dropout,
            )
            if use_relational_gnn
            else None
        )

        # 5. Horizon Track 1: Retrieval-Augmented Behavioral Attention (TabR)
        self.tabr_retrieval: Optional[TabRRetrievalModule] = (
            TabRRetrievalModule(
                embed_dim=hidden_size,
                top_k=tabr_top_k,
                memory_size=tabr_memory_size,
                dropout=dropout,
            )
            if use_tabr_retrieval
            else None
        )

        # 6. Horizon Track 3: In-Context Prior-Data Fitted Network Head (TabPFN v2)
        self.pfn_head = InContextPFNHead(
            input_dim=hidden_size + tabular_feature_size,
            d_model=hidden_size,
            num_heads=num_heads,
            num_layers=max(1, num_layers // 2),
            num_classes=1,
            dropout=dropout,
        )

        # 7. Horizon Track 4: Multi-Gate Mixture-of-Experts Multi-Task Head (MMoE)
        self.mmoe_head = MultiGateMixtureOfExpertsHead(
            input_dim=hidden_size + tabular_feature_size,
            expert_dim=hidden_size,
            num_experts=num_experts,
            dropout=dropout,
        )

        # 8. Horizon Track 5: Causal Counterfactual Uplift Head (DragonNet)
        self.causal_head = CausalDragonNetHead(
            input_dim=hidden_size + tabular_feature_size,
            hidden_dim=hidden_size,
            dropout=dropout,
        )

        # 9. Horizon Track 6: Temporal Hypergraph Contagion & Collusion Ring Scorer (HGNN)
        self.hypergraph_contagion = TemporalHypergraphContagionModule(
            vocab_size=vocab_size,
            embed_dim=hidden_size,
            num_layers=max(1, gnn_layers),
            dropout=dropout,
        )

        # 10. Horizon Track 7: Conditional Optimal-Transport Flow Matcher (FlowLDM)
        self.flow_matcher = ConditionalTrajectoryFlowMatcher(
            latent_dim=hidden_size + tabular_feature_size,
            hidden_dim=hidden_size * 2,
            num_classes=2,
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"LargeDataModel created ({backbone_type}): {total_params:,} total params, "
            f"{trainable_params:,} trainable params"
        )

    def encode(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encodes transaction sequences into contextual per-token embeddings.

        Args:
            packed_keys: Categorical token indices `[batch, seq]`.
            packed_values: Continuous values `[batch, seq]`.
            packed_times: Temporal delta indices `[batch, seq]`.
            key_padding_mask: Optional sequence padding mask `[batch, seq]`.

        Returns:
            Full sequence embeddings `[batch, seq_len + num_virtual, hidden_size]`.
        """
        return self.encoder(
            packed_keys,
            packed_values,
            packed_times,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

    def predict_masked_modalities(
        self,
        sequence_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Projects sequence token embeddings to reconstruct Key, Value, Time, and Latent targets.

        Automatically strips prepended virtual tokens so output shapes match
        the input transaction sequence length `[batch, seq_len, ...]`.

        Args:
            sequence_embeddings: Output of `encode()` `[B, num_virtual + S, D]`.

        Returns:
            Dictionary with `"key_logits"`, `"value_pred"`, `"time_pred"`, `"latent_pred"`.
        """
        num_v = self.encoder.num_virtual_tokens if self.encoder.use_virtual_tokens else 0
        token_emb = sequence_embeddings[:, num_v:, :]
        return {
            "key_logits": self.ssl_key_head(token_emb),
            "value_pred": self.ssl_value_head(token_emb).squeeze(-1),
            "time_pred": self.ssl_time_head(token_emb).squeeze(-1),
            "latent_pred": self.ssl_latent_predictor(sequence_embeddings),
        }

    def get_pooled_embedding(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes a single pooled representation vector per user sequence.

        Returns:
            Pooled embeddings `[batch, hidden_size]`.
        """
        return self.encoder.get_pooled_output(
            packed_keys,
            packed_values,
            packed_times,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

    def get_enriched_representation(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        update_memory: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the entity representation enriched with Relational Bipartite
        GNN message passing (`RelBench`) and Retrieval-Augmented Attention (`TabR`).

        Args:
            packed_keys: Categorical MCC indices `[B, S]`.
            packed_values: Continuous transaction amounts `[B, S]`.
            packed_times: Temporal delta indices `[B, S]`.
            key_padding_mask: Optional boolean padding mask `[B, S]`.
            labels: Optional ground-truth labels `[B]` for updating the TabR memory bank.
            update_memory: Whether to update the TabR memory bank during training.

        Returns:
            Enriched entity embeddings `[B, hidden_size]`.
        """
        pooled = self.get_pooled_embedding(
            packed_keys,
            packed_values,
            packed_times,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

        if self.relational_gnn is not None:
            pooled = self.relational_gnn(
                user_embeddings=pooled,
                event_keys=packed_keys,
                event_values=packed_values,
                key_padding_mask=key_padding_mask,
            )

        if self.tabr_retrieval is not None:
            pooled, _ = self.tabr_retrieval(
                query_embeddings=pooled,
                update_bank=update_memory,
                query_labels=labels,
            )

        return pooled

    def forward(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        update_memory: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full forward pass for supervised classification.

        Args:
            packed_keys: Categorical token indices `[batch, seq]`.
            packed_values: Continuous values `[batch, seq]`.
            packed_times: Temporal delta indices `[batch, seq]`.
            tabular_features: Static tabular features `[batch, tabular_dim]`.
            key_padding_mask: Optional sequence padding mask `[batch, seq]`.
            labels: Optional labels `[batch]` to populate TabR memory bank during training.
            update_memory: If True, updates the TabR historical memory bank.

        Returns:
            Binary classification logits `[batch, 1]` (pre-sigmoid).
        """
        enriched = self.get_enriched_representation(
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            key_padding_mask=key_padding_mask,
            labels=labels,
            update_memory=update_memory,
            **kwargs,
        )

        if tabular_features is not None:
            logits = self.fusion_head(enriched, tabular_features)
        else:
            dummy_tabular = torch.zeros(
                enriched.size(0),
                self.fusion_head.combined_dim - self.hidden_size,
                device=enriched.device,
                dtype=enriched.dtype,
            )
            logits = self.fusion_head(enriched, dummy_tabular)

        return logits

    def forward_in_context(
        self,
        query_keys: torch.Tensor,
        query_values: torch.Tensor,
        query_times: torch.Tensor,
        query_tabular: torch.Tensor,
        support_keys: torch.Tensor,
        support_values: torch.Tensor,
        support_times: torch.Tensor,
        support_tabular: torch.Tensor,
        support_labels: torch.Tensor,
        query_padding_mask: Optional[torch.Tensor] = None,
        support_padding_mask: Optional[torch.Tensor] = None,
        detach_backbone: bool = False,
    ) -> torch.Tensor:
        """
        Performs TabPFN-style In-Context Learning (ICL) classification over
        encoded MMTT + Static Tabular representations in a single forward pass.
        """
        if detach_backbone:
            with torch.no_grad():
                q_emb = self.get_enriched_representation(
                    packed_keys=query_keys,
                    packed_values=query_values,
                    packed_times=query_times,
                    key_padding_mask=query_padding_mask,
                    update_memory=False,
                )
                s_emb = self.get_enriched_representation(
                    packed_keys=support_keys,
                    packed_values=support_values,
                    packed_times=support_times,
                    key_padding_mask=support_padding_mask,
                    update_memory=False,
                )
            q_joint = torch.cat([q_emb.detach(), query_tabular.detach()], dim=-1)
            s_joint = torch.cat([s_emb.detach(), support_tabular.detach()], dim=-1)
        else:
            q_emb = self.get_enriched_representation(
                packed_keys=query_keys,
                packed_values=query_values,
                packed_times=query_times,
                key_padding_mask=query_padding_mask,
                update_memory=False,
            )
            s_emb = self.get_enriched_representation(
                packed_keys=support_keys,
                packed_values=support_values,
                packed_times=support_times,
                key_padding_mask=support_padding_mask,
                update_memory=False,
            )
            q_joint = torch.cat([q_emb, query_tabular], dim=-1)
            s_joint = torch.cat([s_emb, support_tabular], dim=-1)

        return self.pfn_head(
            query_features=q_joint,
            support_features=s_joint,
            support_labels=support_labels,
        )

    def forward_multitask(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        detach_backbone: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes Multi-Gate Mixture-of-Experts (`MMoE`) joint inference for
        Real-Time Fraud Risk, 90-Day Credit Delinquency, and Customer LTV.

        Args:
            packed_keys: Categorical token indices `[B, S]`.
            packed_values: Continuous transaction amounts `[B, S]`.
            packed_times: Temporal delta indices `[B, S]`.
            tabular_features: Static tabular features `[B, T]`.
            key_padding_mask: Optional boolean padding mask `[B, S]`.
            detach_backbone: If True, detaches backbone representations before MMoE.

        Returns:
            Dictionary with `"fraud_logits"`, `"credit_default_logits"`,
            `"ltv_pred"`, and `"gate_weights"`.
        """
        if detach_backbone:
            with torch.no_grad():
                enriched = self.get_enriched_representation(
                    packed_keys=packed_keys,
                    packed_values=packed_values,
                    packed_times=packed_times,
                    key_padding_mask=key_padding_mask,
                    update_memory=False,
                )
            enriched = enriched.detach()
        else:
            enriched = self.get_enriched_representation(
                packed_keys=packed_keys,
                packed_values=packed_values,
                packed_times=packed_times,
                key_padding_mask=key_padding_mask,
                update_memory=False,
            )

        if tabular_features is None:
            tabular_features = torch.zeros(
                enriched.size(0),
                self.tabular_feature_size,
                device=enriched.device,
                dtype=enriched.dtype,
            )
        elif detach_backbone:
            tabular_features = tabular_features.detach()

        joint_features = torch.cat([enriched, tabular_features], dim=-1)
        return self.mmoe_head(joint_features)

    def forward_causal_uplift(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        detach_backbone: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes Causal DragonNet inference returning propensity scores,
        potential outcomes `(y0_prob, y1_prob)`, and CATE uplift.
        """
        if detach_backbone:
            with torch.no_grad():
                enriched = self.get_enriched_representation(
                    packed_keys=packed_keys,
                    packed_values=packed_values,
                    packed_times=packed_times,
                    key_padding_mask=key_padding_mask,
                    update_memory=False,
                )
            enriched = enriched.detach()
        else:
            enriched = self.get_enriched_representation(
                packed_keys=packed_keys,
                packed_values=packed_values,
                packed_times=packed_times,
                key_padding_mask=key_padding_mask,
                update_memory=False,
            )

        if tabular_features is None:
            tabular_features = torch.zeros(
                enriched.size(0),
                self.tabular_feature_size,
                device=enriched.device,
                dtype=enriched.dtype,
            )
        elif detach_backbone:
            tabular_features = tabular_features.detach()

        joint_features = torch.cat([enriched, tabular_features], dim=-1)
        return self.causal_head(joint_features)

    def forward_hypergraph_contagion(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Temporal Hypergraph Contagion representations and multi-hop
        `collusion_ring_score` `[B, 1]` over `(MCC, velocity_tier)` hyperedges.
        """
        pooled = self.get_pooled_embedding(
            packed_keys=packed_keys,
            packed_values=packed_values,
            packed_times=packed_times,
            key_padding_mask=key_padding_mask,
        )
        hg_emb, collusion_score = self.hypergraph_contagion(
            user_embeddings=pooled,
            event_keys=packed_keys,
            event_values=packed_values,
            event_times=packed_times,
            key_padding_mask=key_padding_mask,
        )
        return {
            "hypergraph_embeddings": hg_emb,
            "collusion_ring_score": collusion_score,
        }

    def compute_trajectory_flow_loss(
        self,
        packed_keys: torch.Tensor,
        packed_values: torch.Tensor,
        packed_times: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        detach_backbone: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the Optimal-Transport Conditional Flow Matching (`OT-CFM`) loss
        over joint `[h_enriched || x_tab]` representations.
        """
        if detach_backbone:
            with torch.no_grad():
                enriched = self.get_enriched_representation(
                    packed_keys=packed_keys,
                    packed_values=packed_values,
                    packed_times=packed_times,
                    key_padding_mask=key_padding_mask,
                    update_memory=False,
                ).detach()
        else:
            enriched = self.get_enriched_representation(
                packed_keys=packed_keys,
                packed_values=packed_values,
                packed_times=packed_times,
                key_padding_mask=key_padding_mask,
                update_memory=False,
            )

        if tabular_features is None:
            tabular_features = torch.zeros(
                enriched.size(0),
                self.tabular_feature_size,
                device=enriched.device,
                dtype=enriched.dtype,
            )
        elif detach_backbone:
            tabular_features = tabular_features.detach()

        z_target = torch.cat([enriched, tabular_features], dim=-1)
        return self.flow_matcher.compute_loss(z_target, labels=labels)

    @torch.no_grad()
    def synthesize_counterfactual_trajectories(
        self,
        num_samples: int = 8,
        target_class: int = 1,
        num_steps: int = 16,
        solver: str = "heun",
        guidance_scale: float = 1.5,
        seed: Optional[int] = 42,
    ) -> Dict[str, torch.Tensor]:
        """
        Synthesizes conditional joint behavioral-tabular representations and
        evaluates their downstream DCNv2 fraud logits & probabilities.
        """
        z_syn = self.flow_matcher.sample(
            num_samples=num_samples,
            target_class=target_class,
            num_steps=num_steps,
            solver=solver,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        syn_seq = z_syn[:, : self.hidden_size]
        syn_tab = z_syn[:, self.hidden_size :]
        syn_logits = self.fusion_head(syn_seq, syn_tab)
        return {
            "joint_latent": z_syn,
            "sequence_embedding": syn_seq,
            "tabular_features": syn_tab,
            "fraud_logits": syn_logits,
            "fraud_probs": torch.sigmoid(syn_logits),
        }

    def freeze_encoder(self) -> None:
        """Freezes the Transformer encoder parameters during linear/head probing."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Transformer encoder frozen. Only fusion head will be trained.")

    def unfreeze_encoder(self) -> None:
        """Unfreezes the Transformer encoder for end-to-end fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Transformer encoder unfrozen. Full model will be trained.")

    def save_checkpoint(self, path: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves a model checkpoint using safe tensor-compatible metadata.

        Args:
            path: File path for the checkpoint.
            extra_info: Optional primitive metadata (floats/ints/strings/dicts).
        """
        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": int(self.vocab_size),
                "hidden_size": int(self.hidden_size),
                "num_layers": int(self.num_layers),
                "num_heads": int(self.num_heads),
                "dropout": float(self.dropout),
                "tabular_feature_size": int(self.tabular_feature_size),
                "dcn_layers": int(self.dcn_layers),
                "use_virtual_tokens": bool(self.use_virtual_tokens),
                "num_virtual_tokens": int(self.num_virtual_tokens),
                "pooling": str(self.pooling),
                "use_relational_gnn": bool(self.use_relational_gnn),
                "gnn_layers": int(self.gnn_layers),
                "use_tabr_retrieval": bool(self.use_tabr_retrieval),
                "tabr_top_k": int(self.tabr_top_k),
                "tabr_memory_size": int(self.tabr_memory_size),
                "backbone_type": str(self.backbone_type),
                "num_experts": int(self.num_experts),
            },
        }
        if extra_info:
            checkpoint["extra_info"] = extra_info
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, **kwargs) -> "LargeDataModel":
        """
        Loads a `LargeDataModel` from a checkpoint using safe `weights_only=True`.

        Args:
            path: Path to the checkpoint file.
            **kwargs: Optional configuration overrides.

        Returns:
            Initialized and restored `LargeDataModel` instance.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = dict(checkpoint.get("config", {}))
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Model loaded safely from {path}")
        return model

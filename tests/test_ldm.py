import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.data_module import LDMDataModule
from src.data.mmtt_dataset import (
    FinancialEventDataset,
    NormalizationStats,
    dynamic_sequence_packing_collate_fn,
)
from src.data.synthetic_generator import SyntheticTransactionGenerator
from src.evaluation.benchmarks import BenchmarkSuite
from src.evaluation.metrics import (
    ConformalRiskAbstainer,
    RiskMetricsEvaluator,
    calculate_auc,
    calculate_brier_score,
    calculate_ece,
    calculate_f1_flexible,
    calculate_imputation_rmse,
    calculate_pr_auc,
    calculate_recall_at_fpr,
)
from src.evaluation.strategic_shift_tester import StrategicShiftTester
from src.models.joint_fusion_network import CrossNetworkV2, EndToEndFusionLayer
from src.models.ldm_model import LargeDataModel
from src.models.transformer_backbone import (
    MMTTTransformerEncoder,
    PeriodicNumericalEmbedding,
    SinusoidalPositionalEncoding,
)
from src.training.lightning_module import LDMFineTuningModule, LDMPreTrainingModule
from src.training.objectives import (
    ContextConditionalMaskedLoss,
    ContrastiveEventSequenceLoss,
    NextTokenPredictionLoss,
)


def test_synthetic_generator():
    """Verify synthetic transaction generator output structure, types, and determinism."""
    generator_a = SyntheticTransactionGenerator(num_users=12, seed=42)
    generator_b = SyntheticTransactionGenerator(num_users=12, seed=42)
    df_a = generator_a.generate()
    df_b = generator_b.generate()

    assert isinstance(df_a, pd.DataFrame)
    assert not df_a.empty
    for col in [
        "user_id",
        "mcc_token",
        "amount",
        "time_delta",
        "bureau_score",
        "income",
        "age",
        "is_fraud",
    ]:
        assert col in df_a.columns

    pd.testing.assert_frame_equal(df_a, df_b)


def test_dataset_and_collate():
    """Verify dataset loading, normalization stats, and sequence collation padding masks."""
    generator = SyntheticTransactionGenerator(num_users=10, seed=42)
    df = generator.generate()
    stats = NormalizationStats.fit(df)
    dataset = FinancialEventDataset(df, max_seq_len=20, stats=stats)
    assert len(dataset) == 10

    sample = dataset[0]
    for key in ["keys", "values", "times", "tabular", "label", "seq_len"]:
        assert key in sample

    item1 = {
        "keys": torch.tensor([1, 2, 3], dtype=torch.long),
        "values": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float),
        "times": torch.tensor([0, 1, 2], dtype=torch.long),
        "tabular": torch.tensor([0.5, 0.6, 0.7], dtype=torch.float),
        "label": torch.tensor([0], dtype=torch.long),
        "seq_len": torch.tensor([3], dtype=torch.long),
    }
    item2 = {
        "keys": torch.tensor([4, 5], dtype=torch.long),
        "values": torch.tensor([4.0, 5.0], dtype=torch.float),
        "times": torch.tensor([0, 1], dtype=torch.long),
        "tabular": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float),
        "label": torch.tensor([1], dtype=torch.long),
        "seq_len": torch.tensor([2], dtype=torch.long),
    }

    collated = dynamic_sequence_packing_collate_fn([item1, item2])

    assert "packed_keys" in collated
    assert "key_padding_mask" in collated
    assert collated["packed_keys"].shape == (2, 3)
    assert collated["key_padding_mask"].shape == (2, 3)
    assert not collated["key_padding_mask"][0].any()
    assert not collated["key_padding_mask"][1, 0].item()
    assert not collated["key_padding_mask"][1, 1].item()
    assert collated["key_padding_mask"][1, 2].item()


def test_periodic_numerical_and_positional_encoding():
    """Verify PLR numerical embeddings and sequence order awareness."""
    plr = PeriodicNumericalEmbedding(hidden_size=16, num_frequencies=8)
    scalars = torch.tensor([[0.1, 1.5, -0.8], [2.2, 0.0, 3.4]], dtype=torch.float32)
    emb = plr(scalars)
    assert emb.shape == (2, 3, 16)
    assert not torch.isnan(emb).any()

    pos = SinusoidalPositionalEncoding(hidden_size=16, max_len=32)
    pe = pos(10)
    assert pe.shape == (1, 10, 16)
    # Distinct sequence positions must receive distinct positional encodings
    assert not torch.allclose(pe[:, 0, :], pe[:, 1, :])


def test_cross_network_v2_and_single_sample_fusion():
    """Verify DCNv2 full-rank, low-rank, and single-sample inference stability."""
    batch_size = 4
    input_dim = 16

    cross_full = CrossNetworkV2(input_dim=input_dim, num_layers=3, low_rank=None)
    x = torch.randn(batch_size, input_dim)
    out_full = cross_full(x)
    assert out_full.shape == (batch_size, input_dim)

    cross_low = CrossNetworkV2(input_dim=input_dim, num_layers=3, low_rank=4)
    out_low = cross_low(x)
    assert out_low.shape == (batch_size, input_dim)

    # Single-sample inference (batch_size=1) must not fail
    fusion = EndToEndFusionLayer(transformer_hidden_size=16, tabular_feature_size=3, dcn_layers=2)
    fusion.eval()
    logits_1 = fusion(torch.randn(1, 16), torch.randn(1, 3))
    assert logits_1.shape == (1, 1)


@pytest.mark.parametrize("pooling_mode", ["cls", "mean", "max", "attentive"])
def test_attention_padding_mask_effectiveness(pooling_mode):
    """Verify that padding mask prevents representation contamination across all pooling modes."""
    encoder = MMTTTransformerEncoder(
        vocab_size=10,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        use_virtual_tokens=True,
        num_virtual_tokens=2,
        pooling=pooling_mode,
    )
    encoder.eval()

    keys_a = torch.tensor([[1, 2, 0]], dtype=torch.long)
    values_a = torch.tensor([[0.5, 0.8, 0.0]], dtype=torch.float)
    times_a = torch.tensor([[0, 1, 0]], dtype=torch.long)

    keys_b = torch.tensor([[1, 2, 9]], dtype=torch.long)
    values_b = torch.tensor([[0.5, 0.8, -99.9]], dtype=torch.float)
    times_b = torch.tensor([[0, 1, 999]], dtype=torch.long)

    mask = torch.tensor([[False, False, True]], dtype=torch.bool)

    with torch.no_grad():
        out_a = encoder(keys_a, values_a, times_a, key_padding_mask=mask)
        out_b = encoder(keys_b, values_b, times_b, key_padding_mask=mask)
        pool_a = encoder.get_pooled_output(keys_a, values_a, times_a, key_padding_mask=mask)
        pool_b = encoder.get_pooled_output(keys_b, values_b, times_b, key_padding_mask=mask)

    torch.testing.assert_close(out_a[:, :4, :], out_b[:, :4, :], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pool_a, pool_b, rtol=1e-5, atol=1e-5)


def test_ssl_objectives_and_coles():
    """Verify NTP, LimiX multi-modal reconstruction, and CoLES contrastive objectives."""
    ntp = NextTokenPredictionLoss(vocab_size=10, hidden_size=16)
    hidden = torch.randn(3, 5, 16, requires_grad=True)
    target_k = torch.randint(0, 10, (3, 5))
    target_v = torch.randn(3, 5)
    ntp_loss, ntp_metrics = ntp(hidden, target_k, target_v)
    assert ntp_loss.requires_grad
    assert "loss_ce" in ntp_metrics and "loss_mse" in ntp_metrics

    coles = ContrastiveEventSequenceLoss(temperature=0.15)
    z1 = torch.randn(4, 16, requires_grad=True)
    z2 = z1 + 0.05 * torch.randn(4, 16)
    c_loss = coles(z1, z2)
    assert c_loss.requires_grad
    assert c_loss.item() > 0.0


def test_end_to_end_model_and_lightning(tmp_path):
    """Verify pretraining, finetuning, and safe checkpoint serialization."""
    config = {
        "vocab_size": 20,
        "hidden_size": 16,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "tabular_feature_size": 3,
        "dcn_layers": 2,
        "ssl_masking_prob": 0.15,
        "learning_rate": 1e-3,
        "pos_weight": 2.0,
        "metric_threshold": 0.5,
    }

    model = LargeDataModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        tabular_feature_size=config["tabular_feature_size"],
        dcn_layers=config["dcn_layers"],
        use_virtual_tokens=True,
        num_virtual_tokens=2,
    )

    pretrain_module = LDMPreTrainingModule(model, config)
    dummy_batch = {
        "packed_keys": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "packed_values": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], dtype=torch.float),
        "packed_times": torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long),
        "key_padding_mask": torch.tensor(
            [[False, False, False], [False, False, True]], dtype=torch.bool
        ),
    }

    loss = pretrain_module.training_step(dummy_batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)

    val_loss = pretrain_module.validation_step(dummy_batch, 0)
    assert isinstance(val_loss, torch.Tensor)

    finetune_module = LDMFineTuningModule(model, config)
    dummy_ft_batch = {
        **dummy_batch,
        "tabular": torch.randn(2, 3),
        "labels": torch.tensor([[0], [1]], dtype=torch.long),
    }

    ft_loss = finetune_module.training_step(dummy_ft_batch, 0)
    assert isinstance(ft_loss, torch.Tensor)
    assert not torch.isnan(ft_loss)

    # Verify safe checkpoint save/load roundtrip (weights_only=True)
    ckpt_file = tmp_path / "model_safe.pt"
    model.save_checkpoint(str(ckpt_file), extra_info={"epoch": 1})
    loaded_model = LargeDataModel.load_checkpoint(str(ckpt_file))
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        out_orig = model(
            dummy_ft_batch["packed_keys"],
            dummy_ft_batch["packed_values"],
            dummy_ft_batch["packed_times"],
            tabular_features=dummy_ft_batch["tabular"],
            key_padding_mask=dummy_ft_batch["key_padding_mask"],
        )
        out_loaded = loaded_model(
            dummy_ft_batch["packed_keys"],
            dummy_ft_batch["packed_values"],
            dummy_ft_batch["packed_times"],
            tabular_features=dummy_ft_batch["tabular"],
            key_padding_mask=dummy_ft_batch["key_padding_mask"],
        )
    torch.testing.assert_close(out_orig, out_loaded)


def test_calibrated_metrics_and_conformal_abstention():
    """Verify PR-AUC, Recall@FPR, Brier Score, ECE, and ConformalRiskAbstainer."""
    # Well-separated logits: negatives around -2.5, positives around +2.5, 2 ambiguous near 0.0
    logits = torch.tensor([-3.0, -2.5, -2.0, -1.8, -0.05, 0.05, 2.0, 2.5, 3.0, 3.2])
    targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    roc_auc = calculate_auc(logits, targets)
    pr_auc = calculate_pr_auc(logits, targets)
    rec_fpr = calculate_recall_at_fpr(logits, targets, max_fpr=0.25)
    brier = calculate_brier_score(logits, targets)
    ece = calculate_ece(logits, targets, n_bins=5)
    f1 = calculate_f1_flexible(logits, targets, threshold=0.5)

    assert roc_auc >= 0.95
    assert pr_auc >= 0.95
    assert rec_fpr >= 0.80
    assert 0.0 <= brier < 0.15
    assert 0.0 <= ece < 0.25
    assert f1 >= 0.90

    abstainer = ConformalRiskAbstainer(alpha=0.15)
    bounds = abstainer.calibrate(logits, targets)
    assert bounds["lower_threshold"] < bounds["upper_threshold"]

    decisions = abstainer.predict_with_abstention(logits)
    # Extreme negative and positive examples must be classified (0 and 1), while near-0 abstains (-1)
    assert decisions[0].item() == 0
    assert decisions[-1].item() == 1
    assert (decisions == -1).any()

    eval_dict = RiskMetricsEvaluator(threshold=0.5, conformal_alpha=0.15).evaluate(logits, targets)
    assert "conformal_selective_accuracy" in eval_dict
    assert eval_dict["conformal_selective_accuracy"] >= 0.95

    # Imputation RMSE check
    preds = torch.tensor([1.0, 2.0, 4.0])
    trues = torch.tensor([1.0, 2.0, 2.0])
    mask = torch.tensor([False, False, True])
    assert abs(calculate_imputation_rmse(preds, trues, mask) - 2.0) < 1e-5


def test_strategic_shift_and_benchmark_suite():
    """Verify adversarial shift tester and BenchmarkSuite execution."""
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    model.eval()

    tester = StrategicShiftTester(noise_std=0.1, adversarial_epsilon=0.05)

    keys = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    values = torch.tensor([[0.5, -0.2, 1.1], [0.1, 0.9, 0.0]], dtype=torch.float)
    times = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
    pad_mask = torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool)
    targets = torch.tensor([[0], [1]], dtype=torch.long)
    tabular = torch.randn(2, 3)

    noisy = tester.apply_random_shift(values)
    assert noisy.shape == values.shape

    _, perturbed_values, _ = tester.apply_gradient_adversarial_shift(
        model, (keys, values, times), targets, tabular=tabular, key_padding_mask=pad_mask
    )
    assert perturbed_values.shape == values.shape
    assert not torch.equal(perturbed_values, values)
    # Padded index [1, 2] must remain 0.0
    assert perturbed_values[1, 2].item() == 0.0

    # Test BenchmarkSuite on a small DataLoader
    gen = SyntheticTransactionGenerator(num_users=20, min_transactions=6, max_transactions=12, fraud_rate=0.25, seed=42)
    df = gen.generate()
    ds = FinancialEventDataset(df, max_seq_len=12)
    loader = DataLoader(ds, batch_size=10, collate_fn=dynamic_sequence_packing_collate_fn)

    suite = BenchmarkSuite(model, {"metric_threshold": 0.5, "enable_strategic_shift": True})
    results = suite.run_all(loader, val_dataloader=loader)
    assert "auc" in results
    assert "pr_auc" in results
    assert "fgsm_auc" in results
    assert "conformal_coverage" in results
    assert "torchscript_p99_ms" in results


# =============================================================================
# 6. Next-Horizon Frontier Tracks Tests (TabR, RelBench GNN, TabPFN v2, Export)
# =============================================================================


def test_tabr_retrieval_self_masking_and_memory_bank():
    """Verify TabRRetrievalModule circular memory bank and batch_size=1 eval retrieval."""
    from src.models.retrieval_attention import TabRRetrievalModule

    torch.manual_seed(42)
    tabr = TabRRetrievalModule(embed_dim=16, top_k=3, memory_size=10, dropout=0.0)
    tabr.train()

    batch_emb = torch.randn(6, 16)
    batch_labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    # First forward pass in training mode with empty memory bank (uses leave-one-out in-batch)
    enriched, attn = tabr(batch_emb, update_bank=True, query_labels=batch_labels)
    assert enriched.shape == (6, 16)
    assert attn.shape == (6, 3)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert int(tabr.memory_count.item()) == 6

    # Switch to eval mode and verify batch_size=1 single-swipe retrieval from memory bank
    tabr.eval()
    single_query = torch.randn(1, 16)
    single_out, single_attn = tabr(single_query)
    assert single_out.shape == (1, 16)
    assert single_attn.shape == (1, 3)
    assert torch.isfinite(single_out).all()


def test_relational_entity_gnn_mask_invariance_and_b1():
    """Verify RelationalEntityGNN respects key_padding_mask and supports batch_size=1."""
    from src.models.relational_gnn import RelationalEntityGNN

    torch.manual_seed(42)
    gnn = RelationalEntityGNN(vocab_size=10, embed_dim=16, num_layers=2, dropout=0.0)
    gnn.eval()

    user_emb = torch.randn(1, 16)
    keys = torch.tensor([[1, 2, 3, 9, 9]], dtype=torch.long)
    vals_a = torch.tensor([[0.5, 1.2, 0.8, 0.0, 0.0]], dtype=torch.float32)
    vals_b = torch.tensor([[0.5, 1.2, 0.8, 500.0, 999.0]], dtype=torch.float32)
    pad_mask = torch.tensor([[False, False, False, True, True]], dtype=torch.bool)

    out_a = gnn(user_emb, keys, vals_a, key_padding_mask=pad_mask)
    out_b = gnn(user_emb, keys, vals_b, key_padding_mask=pad_mask)

    assert out_a.shape == (1, 16)
    # Padded positions must have zero influence on bipartite adjacency
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_in_context_pfn_head_query_order_invariance():
    """Verify InContextPFNHead and LargeDataModel.forward_in_context query-order invariance."""
    from src.models.in_context_pfn import InContextPFNHead

    torch.manual_seed(42)
    pfn = InContextPFNHead(input_dim=12, d_model=16, num_heads=2, num_layers=2, dropout=0.0)
    pfn.eval()

    support_x = torch.randn(8, 12)
    support_y = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.long)
    query_x = torch.randn(4, 12)

    logits_orig = pfn(query_x, support_x, support_y)
    # Reverse query order
    perm = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    logits_perm = pfn(query_x[perm], support_x, support_y)

    assert logits_orig.shape == (4, 1)
    # Asymmetric attention mask guarantees each query depends solely on Support + self
    assert torch.allclose(logits_orig[perm], logits_perm, atol=1e-5)

    # Also test end-to-end LargeDataModel.forward_in_context
    model = LargeDataModel(vocab_size=10, hidden_size=16, num_layers=1, num_heads=2, tabular_feature_size=3)
    model.eval()
    q_logits = model.forward_in_context(
        query_keys=torch.randint(0, 10, (3, 6)),
        query_values=torch.randn(3, 6),
        query_times=torch.randint(0, 50, (3, 6)),
        query_tabular=torch.randn(3, 3),
        support_keys=torch.randint(0, 10, (6, 6)),
        support_values=torch.randn(6, 6),
        support_times=torch.randint(0, 50, (6, 6)),
        support_tabular=torch.randn(6, 3),
        support_labels=torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    assert q_logits.shape == (3, 1)
    assert torch.isfinite(q_logits).all()


def test_torchscript_export_parity_and_latency_benchmark(tmp_path):
    """Verify TorchScript export numerical parity (<1e-5) and latency benchmarking."""
    from src.serving import export_to_torchscript, benchmark_inference_latency

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    model.eval()

    ts_path = tmp_path / "ldm_model.torchscript.pt"
    traced_mod, max_diff = export_to_torchscript(
        model, save_path=str(ts_path), batch_size=1, seq_len=16
    )
    assert ts_path.exists()
    assert max_diff < 1e-5

    latency_stats = benchmark_inference_latency(
        model, batch_size=1, seq_len=16, num_warmup=2, num_runs=10
    )
    assert latency_stats["torchscript_parity_max_diff"] < 1e-5
    assert latency_stats["torchscript_p99_ms"] > 0.0
    assert latency_stats["torchscript_throughput_rps"] > 0.0


# =============================================================================
# 7. Round 3 (v0.4.0) Tests: Explainability, Federated DP-SSL, Ablation & Auth
# =============================================================================


def test_explainability_integrated_gradients_and_adverse_reasons():
    """Verify Integrated Gradients padding-mask zeroing and Adverse Action Reason extraction."""
    from src.evaluation.explainability import LDMExplainabilityEngine

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=20,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    model.eval()
    explainer = LDMExplainabilityEngine(model)

    keys = torch.tensor([[14, 18, 19, 0, 0]], dtype=torch.long)
    values = torch.tensor([[1.5, 2.8, -0.5, 99.0, 99.0]], dtype=torch.float32)
    times = torch.tensor([[0, 2, 1, 0, 0]], dtype=torch.long)
    tabular = torch.tensor([[-1.2, 0.8, -0.4]], dtype=torch.float32)
    pad_mask = torch.tensor([[False, False, False, True, True]], dtype=torch.bool)

    ig_dict = explainer.compute_integrated_gradients(
        keys, values, times, tabular, key_padding_mask=pad_mask, num_steps=8
    )
    val_attr = ig_dict["value_attributions"]
    ev_share = ig_dict["event_importance_share"]

    assert val_attr.shape == (1, 5)
    # Padded positions [0, 3] and [0, 4] must have strictly 0.0 attribution
    assert val_attr[0, 3].item() == 0.0
    assert val_attr[0, 4].item() == 0.0
    assert torch.allclose(ev_share[:, :3].sum(dim=-1), torch.ones(1), atol=1e-5)

    reasons = explainer.extract_adverse_action_reasons(
        keys, values, times, tabular, key_padding_mask=pad_mask, top_k_events=2, top_k_tabular=2
    )
    assert len(reasons) == 1
    assert len(reasons[0]) == 4
    assert all(r.event_index not in (3, 4) for r in reasons[0])


def test_counterfactual_recourse_probability_reduction():
    """Verify sparse counterfactual recourse reduces risk probability while respecting padding."""
    from src.evaluation.explainability import LDMExplainabilityEngine

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=20,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    # Bias final classification layer slightly positive so initial probability > 0.40
    with torch.no_grad():
        model.fusion_head.final_projection.bias.fill_(1.0)

    explainer = LDMExplainabilityEngine(model)
    keys = torch.tensor([[14, 18, 0]], dtype=torch.long)
    values = torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float32)
    times = torch.tensor([[0, 1, 0]], dtype=torch.long)
    tabular = torch.tensor([[1.0, -0.5, 0.2]], dtype=torch.float32)
    pad_mask = torch.tensor([[False, False, True]], dtype=torch.bool)

    recourse = explainer.generate_counterfactual_recourse(
        keys,
        values,
        times,
        tabular,
        key_padding_mask=pad_mask,
        target_probability=0.30,
        max_steps=25,
        step_size=0.25,
    )
    assert recourse.counterfactual_probability < recourse.initial_probability
    assert recourse.success
    # Padded index [0, 2] must have zero counterfactual perturbation
    assert recourse.delta_values[0, 2].item() == 0.0


def test_federated_dp_ssl_coordinator_and_privacy_accountant():
    """Verify Cross-Silo Federated SSL (FedProx + L2 Clipping + DP Noise)."""
    from src.training.federated import FederatedLDMCoordinator, FederatedPrivacyConfig

    torch.manual_seed(42)
    global_model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    privacy_cfg = FederatedPrivacyConfig(
        enable_dp=True,
        max_grad_norm=0.5,
        noise_multiplier=0.05,
        delta=1e-5,
        proximal_mu=0.02,
    )
    coordinator = FederatedLDMCoordinator(global_model, privacy_config=privacy_cfg)

    gen_1 = SyntheticTransactionGenerator(num_users=12, min_transactions=5, max_transactions=8, fraud_rate=0.25, seed=11)
    gen_2 = SyntheticTransactionGenerator(num_users=12, min_transactions=5, max_transactions=8, fraud_rate=0.25, seed=22)
    loader_1 = DataLoader(FinancialEventDataset(gen_1.generate(), max_seq_len=8), batch_size=6, collate_fn=dynamic_sequence_packing_collate_fn)
    loader_2 = DataLoader(FinancialEventDataset(gen_2.generate(), max_seq_len=8), batch_size=6, collate_fn=dynamic_sequence_packing_collate_fn)

    report = coordinator.run_federated_ssl_round(
        [loader_1, loader_2], local_steps_per_client=1, learning_rate=1e-3, seed=42
    )
    assert report.round_idx == 1
    assert report.num_clients == 2
    assert report.mean_client_ssl_loss > 0.0
    assert report.cumulative_dp_epsilon > 0.0


def test_ablation_suite_and_realtime_authorization_service():
    """Verify AblationBenchmarkSuite and RealTimeAuthorizationService end-to-end."""
    from src.evaluation.ablation_suite import AblationBenchmarkSuite
    from src.serving.authorization_service import RealTimeAuthorizationService

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    model.eval()

    gen = SyntheticTransactionGenerator(num_users=16, min_transactions=5, max_transactions=8, fraud_rate=0.25, seed=42)
    ds = FinancialEventDataset(gen.generate(), max_seq_len=8)
    loader = DataLoader(ds, batch_size=8, collate_fn=dynamic_sequence_packing_collate_fn)

    ablation = AblationBenchmarkSuite(model)
    variants = ablation.run_ablation_study(loader, support_dataloader=loader)
    assert len(variants) == 6
    md_table = AblationBenchmarkSuite.format_markdown_table(variants)
    assert "Full LDM" in md_table
    assert "Multi-Task MMoE Head" in md_table

    auth_service = RealTimeAuthorizationService(
        model, conformal_alpha=0.10, use_torchscript=True, seq_len=8
    )
    cal_logits = torch.tensor([[-2.0], [2.0], [-1.5], [1.8]], dtype=torch.float32)
    cal_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    auth_service.calibrate_governance(cal_logits, cal_targets)

    sample = next(iter(loader))
    decision = auth_service.authorize(
        packed_keys=sample["packed_keys"][:1],
        packed_values=sample["packed_values"][:1],
        packed_times=sample["packed_times"][:1],
        tabular_features=sample["tabular"][:1],
        key_padding_mask=sample["key_padding_mask"][:1],
        explain_on_adverse=True,
    )
    payload = decision.to_dict()
    assert payload["decision"] in ("APPROVE", "DECLINE", "HUMAN_REVIEW")
    assert 0.0 <= payload["fraud_probability"] <= 1.0
    assert payload["latency_ms"] > 0.0


def test_selective_ssm_and_hybrid_backbone_mask_invariance():
    """Verify SelectiveSSMBlock and HybridSSMTransformerEncoder O(S) padding-mask invariance."""
    torch.manual_seed(42)
    model_hybrid = LargeDataModel(
        vocab_size=12,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        tabular_feature_size=3,
        backbone_type="hybrid_ssm",
    )
    model_ssm = LargeDataModel(
        vocab_size=12,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        tabular_feature_size=3,
        backbone_type="ssm",
    )
    model_hybrid.eval()
    model_ssm.eval()

    keys = torch.randint(0, 12, (2, 10), dtype=torch.long)
    values = torch.randn(2, 10)
    times = torch.randint(1, 50, (2, 10), dtype=torch.long)
    tabular = torch.randn(2, 3)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 6:] = True

    keys_corrupted = keys.clone()
    values_corrupted = values.clone()
    times_corrupted = times.clone()
    keys_corrupted[:, 6:] = 11
    values_corrupted[:, 6:] = 999.0
    times_corrupted[:, 6:] = 5000

    with torch.no_grad():
        p1 = model_ssm.get_pooled_embedding(keys, values, times, key_padding_mask=mask)
        p2 = model_ssm.get_pooled_embedding(
            keys_corrupted, values_corrupted, times_corrupted, key_padding_mask=mask
        )
        assert torch.allclose(p1, p2, atol=1e-5)

        out_h1 = model_hybrid(
            keys, values, times, tabular_features=tabular, key_padding_mask=mask
        )
        out_h2 = model_hybrid(
            keys_corrupted,
            values_corrupted,
            times_corrupted,
            tabular_features=tabular,
            key_padding_mask=mask,
        )
        assert out_h1.shape == (2, 1)
        assert torch.allclose(out_h1, out_h2, atol=1e-5)


def test_mmoe_multitask_head_and_homoscedastic_loss():
    """Verify MultiGateMixtureOfExpertsHead routing simplex and HomoscedasticMultiTaskLoss."""
    from src.models.multi_task_moe import HomoscedasticMultiTaskLoss

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
        num_experts=4,
    )
    keys = torch.randint(0, 10, (5, 8), dtype=torch.long)
    values = torch.randn(5, 8)
    times = torch.randint(1, 30, (5, 8), dtype=torch.long)
    tabular = torch.randn(5, 3)

    mt_out = model.forward_multitask(
        packed_keys=keys,
        packed_values=values,
        packed_times=times,
        tabular_features=tabular,
    )
    assert mt_out["fraud_logits"].shape == (5, 1)
    assert mt_out["credit_default_logits"].shape == (5, 1)
    assert mt_out["ltv_pred"].shape == (5, 1)
    assert mt_out["gate_weights"].shape == (5, 3, 4)
    gate_sums = mt_out["gate_weights"].sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)

    loss_fn = HomoscedasticMultiTaskLoss()
    fraud_y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)
    credit_y = torch.tensor([0, 1, 1, 0, 0], dtype=torch.float32)
    ltv_y = torch.randn(5)
    loss_dict = loss_fn(mt_out, fraud_y, credit_y, ltv_y)
    loss_dict["loss"].backward()
    assert loss_fn.log_vars.grad is not None
    assert float(loss_dict["loss"].item()) > -10.0


def test_concept_drift_monitor_and_adaptive_conformal_abstainer():
    """Verify ConceptDriftMonitor (PSI, MMD, Wasserstein) and AdaptiveConformalAbstainer."""
    from src.evaluation.drift_monitor import (
        AdaptiveConformalAbstainer,
        ConceptDriftMonitor,
    )

    torch.manual_seed(42)
    monitor = ConceptDriftMonitor(
        psi_warning_threshold=0.10,
        psi_critical_threshold=0.25,
        feature_names=["income", "age", "credit_score"],
    )
    ref_tab = torch.randn(200, 3)
    ref_amt = torch.randn(200, 10)
    ref_emb = torch.randn(200, 16)
    ref_prob = torch.sigmoid(torch.randn(200))
    monitor.fit_reference(ref_tab, ref_amt, ref_emb, ref_prob)

    # 1. Stationary window -> STABLE
    stable_report = monitor.evaluate_window(
        ref_tab + 0.01 * torch.randn_like(ref_tab),
        ref_amt + 0.01 * torch.randn_like(ref_amt),
        ref_emb + 0.01 * torch.randn_like(ref_emb),
        ref_prob,
    )
    assert stable_report.status == "STABLE"
    assert stable_report.max_feature_psi < 0.10

    # 2. Shifted window -> CRITICAL_DRIFT
    drifted_report = monitor.evaluate_window(
        ref_tab + 2.5,
        ref_amt + 3.0,
        ref_emb + 2.0,
        torch.clamp(ref_prob + 0.4, 0.0, 1.0),
    )
    assert drifted_report.status == "CRITICAL_DRIFT"
    assert drifted_report.max_feature_psi >= 0.25
    assert drifted_report.embedding_mmd > 0.10
    assert drifted_report.probability_wasserstein > 0.10

    # 3. Online Adaptive Conformal Inference (ACI)
    aci = AdaptiveConformalAbstainer(alpha_target=0.10, gamma=0.10)
    cal_logits = torch.cat([torch.full((50,), -2.0), torch.full((50,), 2.0)])
    cal_targets = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    aci.calibrate(cal_logits, cal_targets)
    initial_alpha = aci.effective_alpha

    # Simulate high-error drifted window -> effective_alpha must decrease
    bad_logits = torch.full((20,), 2.5)
    bad_targets = torch.zeros(20, dtype=torch.long)
    step_info = aci.update_online(bad_logits, bad_targets)
    assert step_info["observed_error_rate"] > 0.50
    assert step_info["effective_alpha"] < initial_alpha


def test_dynamic_int8_quantization_and_cascade_early_exit_router():
    """Verify quantize_ldm_dynamic_int8 compression and CascadeEarlyExitRouter execution."""
    from src.serving.quantization_and_cascade import (
        CascadeEarlyExitRouter,
        quantize_ldm_dynamic_int8,
    )

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=12,
        hidden_size=32,
        num_layers=2,
        num_heads=2,
        tabular_feature_size=3,
    )
    model.eval()

    quant_model, q_metrics = quantize_ldm_dynamic_int8(model, seq_len=8)
    assert q_metrics["fp32_size_mb"] > 0.0
    assert q_metrics["int8_size_mb"] > 0.0
    assert q_metrics["int8_max_prob_diff"] < 0.25

    router = CascadeEarlyExitRouter(model, fast_exit_threshold=0.90)
    keys = torch.randint(0, 12, (12, 8), dtype=torch.long)
    vals = torch.randn(12, 8)
    times = torch.randint(1, 50, (12, 8), dtype=torch.long)
    tab = torch.randn(12, 3)
    pad = torch.zeros(12, 8, dtype=torch.bool)

    distill_loss = router.fit_fast_stage(keys, vals, times, tab, pad, num_steps=15)
    assert distill_loss >= 0.0

    logits, exited_early = router(keys, vals, times, tab, pad)
    assert logits.shape == (12, 1)
    assert exited_early.shape == (12,)

    report = router.benchmark_cascade(keys, vals, times, tab, pad, num_runs=5)
    assert report.compression_ratio > 0.5
    assert 0.0 <= report.fast_exit_rate <= 1.0


def test_causal_dragonnet_uplift_and_policy_optimizer():
    """Verify CausalDragonNetHead, DragonNetTargetedLoss, and recommend_optimal_intervention."""
    from src.models.causal_uplift import (
        DragonNetTargetedLoss,
        recommend_optimal_intervention,
    )

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    keys = torch.randint(0, 10, (6, 8), dtype=torch.long)
    values = torch.randn(6, 8)
    times = torch.randint(1, 30, (6, 8), dtype=torch.long)
    tabular = torch.randn(6, 3)

    causal_out = model.forward_causal_uplift(
        packed_keys=keys,
        packed_values=values,
        packed_times=times,
        tabular_features=tabular,
    )
    assert causal_out["propensity_prob"].shape == (6, 1)
    assert causal_out["y0_prob"].shape == (6, 1)
    assert causal_out["y1_prob"].shape == (6, 1)
    assert causal_out["cate"].shape == (6, 1)
    assert torch.allclose(
        causal_out["cate"], causal_out["y1_prob"] - causal_out["y0_prob"], atol=1e-6
    )

    loss_fn = DragonNetTargetedLoss()
    treatments = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float32)
    outcomes = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.float32)
    loss_dict = loss_fn(
        causal_out, treatments, outcomes, model.causal_head.tmle_epsilon
    )
    loss_dict["loss"].backward()
    assert model.causal_head.tmle_epsilon.grad is not None

    policy = recommend_optimal_intervention(causal_out)
    assert len(policy["action_names"]) == 6
    assert all(
        name in ("FRICTIONLESS_APPROVE", "STEP_UP_2FA", "HARD_BLOCK")
        for name in policy["action_names"]
    )


def test_temporal_hypergraph_contagion_mask_invariance():
    """Verify TemporalHypergraphContagionModule output shapes and strict padding-mask invariance."""
    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=12,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=3,
    )
    model.eval()

    keys = torch.randint(0, 12, (4, 10), dtype=torch.long)
    values = torch.randn(4, 10)
    times = torch.randint(1, 40, (4, 10), dtype=torch.long)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, 6:] = True

    keys_corrupted = keys.clone()
    values_corrupted = values.clone()
    times_corrupted = times.clone()
    keys_corrupted[:, 6:] = 11
    values_corrupted[:, 6:] = 500.0
    times_corrupted[:, 6:] = 1

    with torch.no_grad():
        res_1 = model.forward_hypergraph_contagion(
            keys, values, times, key_padding_mask=mask
        )
        res_2 = model.forward_hypergraph_contagion(
            keys_corrupted, values_corrupted, times_corrupted, key_padding_mask=mask
        )
        assert res_1["hypergraph_embeddings"].shape == (4, 16)
        assert res_1["collusion_ring_score"].shape == (4, 1)
        assert torch.allclose(
            res_1["hypergraph_embeddings"], res_2["hypergraph_embeddings"], atol=1e-5
        )
        assert torch.allclose(
            res_1["collusion_ring_score"], res_2["collusion_ring_score"], atol=1e-5
        )


def test_certified_machine_unlearning_tabr_and_fisher_scrubbing():
    """Verify CertifiedLDMUnlearner purges TabR memory slots and issues UnlearningCertificate."""
    from src.training.unlearning import CertifiedLDMUnlearner

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        tabular_feature_size=3,
        tabr_memory_size=32,
        tabr_top_k=4,
    )

    gen = SyntheticTransactionGenerator(
        num_users=12, min_transactions=5, max_transactions=8, fraud_rate=0.25, seed=42
    )
    ds = FinancialEventDataset(gen.generate(), max_seq_len=8)
    loader = DataLoader(ds, batch_size=6, collate_fn=dynamic_sequence_packing_collate_fn)
    batches = list(loader)
    forget_batch, retain_batch = batches[0], batches[1]

    # Populate TabR memory bank with both cohorts
    model.train()
    _ = model(
        forget_batch["packed_keys"],
        forget_batch["packed_values"],
        forget_batch["packed_times"],
        tabular_features=forget_batch["tabular"],
        key_padding_mask=forget_batch["key_padding_mask"],
        labels=forget_batch["labels"],
        update_memory=True,
    )
    _ = model(
        retain_batch["packed_keys"],
        retain_batch["packed_values"],
        retain_batch["packed_times"],
        tabular_features=retain_batch["tabular"],
        key_padding_mask=retain_batch["key_padding_mask"],
        labels=retain_batch["labels"],
        update_memory=True,
    )
    assert int(model.tabr_retrieval.memory_count.item()) == 12

    unlearner = CertifiedLDMUnlearner(model, noise_scale=1e-4)
    cert = unlearner.unlearn_cohort(
        forget_batch=forget_batch,
        retain_batch=retain_batch,
        num_steps=2,
        unlearn_lr=5e-3,
    )
    assert cert.num_forgotten_users == 6
    assert cert.tabr_entries_purged >= 1
    assert int(model.tabr_retrieval.memory_count.item()) < 12
    assert cert.parameter_l2_drift > 0.0
    assert cert.certified_epsilon > 0.0


def test_fairness_governance_auditor_and_equalized_odds_optimizer():
    """Verify FairnessGovernanceAuditor (DPD, DIR 80% rule, EOD) and EqualizedOddsThresholdOptimizer."""
    from src.evaluation.fairness import (
        EqualizedOddsThresholdOptimizer,
        FairnessGovernanceAuditor,
    )

    torch.manual_seed(42)
    auditor = FairnessGovernanceAuditor(four_fifths_threshold=0.80)

    # Group 0 has low logits (high approval), Group 1 has artificially inflated logits (low approval)
    g0_logits = torch.linspace(-2.0, 0.4, 50)
    g1_logits = torch.linspace(-0.5, 2.0, 50)
    logits = torch.cat([g0_logits, g1_logits])
    targets = torch.cat([torch.zeros(40), torch.ones(10), torch.zeros(40), torch.ones(10)]).long()
    groups = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    raw_report = auditor.audit_logits(logits, targets, groups, threshold=0.5)
    assert not raw_report.passes_four_fifths_rule
    assert raw_report.disparate_impact_ratio < 0.80

    optimizer = EqualizedOddsThresholdOptimizer(default_threshold=0.5, target_fpr=0.10)
    thresholds = optimizer.fit(logits, targets, groups)
    assert thresholds[1] > thresholds[0]

    fair_preds = optimizer.predict(logits, groups)
    mitigated_report = auditor.audit_predictions(fair_preds, targets, groups)
    assert mitigated_report.disparate_impact_ratio > raw_report.disparate_impact_ratio
    assert mitigated_report.demographic_parity_diff < raw_report.demographic_parity_diff


def test_conditional_flow_matching_loss_and_heun_synthesis():
    """Verify ConditionalTrajectoryFlowMatcher OT-CFM training loss and Euler/Heun ODE synthesis."""
    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=12,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=4,
    )
    keys = torch.randint(1, 12, (6, 8))
    values = torch.randn(6, 8)
    times = torch.randint(0, 10, (6, 8))
    tab = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    flow_out = model.compute_trajectory_flow_loss(
        packed_keys=keys,
        packed_values=values,
        packed_times=times,
        tabular_features=tab,
        labels=labels,
    )
    assert flow_out["loss"].requires_grad
    assert float(flow_out["loss"].item()) > 0.0
    flow_out["loss"].backward()

    syn_fraud = model.synthesize_counterfactual_trajectories(
        num_samples=5, target_class=1, num_steps=8, solver="heun", seed=42
    )
    assert syn_fraud["joint_latent"].shape == (5, 20)
    assert syn_fraud["sequence_embedding"].shape == (5, 16)
    assert syn_fraud["tabular_features"].shape == (5, 4)
    assert syn_fraud["fraud_probs"].shape == (5, 1)
    assert torch.all((syn_fraud["fraud_probs"] >= 0.0) & (syn_fraud["fraud_probs"] <= 1.0))

    # Deterministic synthesis with same seed
    syn_fraud_repeat = model.synthesize_counterfactual_trajectories(
        num_samples=5, target_class=1, num_steps=8, solver="heun", seed=42
    )
    assert torch.allclose(syn_fraud["joint_latent"], syn_fraud_repeat["joint_latent"], atol=1e-6)


def test_continual_ewc_and_dark_experience_lifelong_adaptation():
    """Verify ContinualLDMAdapter consolidates Fisher importance and adapts without catastrophic forgetting."""
    from src.training.continual import ContinualLDMAdapter

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=12,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        tabular_feature_size=3,
    )

    gen = SyntheticTransactionGenerator(
        num_users=12, min_transactions=5, max_transactions=8, fraud_rate=0.25, seed=42
    )
    ds = FinancialEventDataset(gen.generate(), max_seq_len=8)
    loader = DataLoader(ds, batch_size=6, collate_fn=dynamic_sequence_packing_collate_fn)
    batches = list(loader)
    regime_1_batch, regime_2_batch = batches[0], batches[1]

    adapter = ContinualLDMAdapter(model, ewc_lambda=20.0, der_alpha=0.5)
    adapter.consolidate_regime(regime_1_batch)
    assert adapter.num_consolidated_regimes == 1
    assert len(adapter.fisher_omega) > 0

    report = adapter.adapt_to_new_regime(
        new_regime_batch=regime_2_batch,
        historical_probe_batch=regime_1_batch,
        num_steps=4,
        lr=5e-3,
    )
    assert report.num_consolidated_regimes == 1
    assert report.new_regime_loss_after < report.new_regime_loss_before
    assert report.ewc_penalty >= 0.0
    assert report.parameter_l2_drift > 0.0


def test_peft_dora_adapters_zero_latency_merge_and_nf4_quantization():
    """Verify LoRALinear (DoRA) exact initialization, zero-latency weight merging, and 4-bit NF4 compression."""
    from src.serving.peft_and_nf4 import (
        inject_lora_adapters,
        merge_lora_adapters,
        quantize_ldm_nf4,
    )

    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        tabular_feature_size=4,
    )
    model.eval()

    keys = torch.randint(1, 16, (4, 8))
    values = torch.randn(4, 8)
    times = torch.randint(0, 10, (4, 8))
    tab = torch.randn(4, 4)

    with torch.no_grad():
        base_logits = model(keys, values, times, tabular_features=tab)

    # Test 4-bit NormalFloat (NF4) quantization
    nf4_model, nf4_report = quantize_ldm_nf4(model, block_size=32)
    assert nf4_report["compression_ratio"] > 1.10
    assert nf4_report["linear_compression_ratio"] > 4.0
    with torch.no_grad():
        nf4_logits = nf4_model(keys, values, times, tabular_features=tab)
    assert nf4_logits.shape == base_logits.shape
    assert torch.isfinite(nf4_logits).all()

    # Test DoRA injection: at initialization (B=0), output must match base_logits identically
    peft_summary = inject_lora_adapters(model, rank=4, alpha=8.0, use_dora=True)
    assert peft_summary["trainable_pct"] < 50.0
    model.eval()
    with torch.no_grad():
        dora_init_logits = model(keys, values, times, tabular_features=tab)
    assert torch.allclose(base_logits, dora_init_logits, atol=1e-5)

    # Take one gradient step on DoRA parameters, then merge and verify zero-diff parity
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss = model(keys, values, times, tabular_features=tab).pow(2).mean()
    loss.backward()
    opt.step()

    with torch.no_grad():
        pre_merge_logits = model(keys, values, times, tabular_features=tab)

    merged_count = merge_lora_adapters(model)
    assert merged_count > 0
    with torch.no_grad():
        post_merge_logits = model(keys, values, times, tabular_features=tab)
    assert torch.allclose(pre_merge_logits, post_merge_logits, atol=1e-5)


def test_tamper_evident_audit_ledger_and_logit_watermarking():
    """Verify TamperEvidentAuditLedger HMAC-SHA256 hash chain and StatisticalLogitWatermarker detection."""
    from src.evaluation.audit_ledger import (
         AuditRecord,
        StatisticalLogitWatermarker,
        TamperEvidentAuditLedger,
    )

    ledger = TamperEvidentAuditLedger(secret_key="unit-test-secret", model_version="0.7.0")
    r0 = ledger.record_decision(
        transaction_id="tx_001",
        entity_id="usr_10",
        risk_probability=0.0412,
        decision="APPROVE",
        conformal_interval=(0.05, 0.92),
        adverse_reasons=[],
    )
    r1 = ledger.record_decision(
        transaction_id="tx_002",
        entity_id="usr_99",
        risk_probability=0.9640,
        decision="DECLINE",
        conformal_interval=(0.05, 0.92),
        adverse_reasons=["HIGH_VELOCITY_MCC_BURST"],
    )
    assert r1.prev_hash == r0.record_hash
    is_valid, tampered_idx = ledger.verify_chain_integrity()
    assert is_valid
    assert tampered_idx is None

    # Simulate retroactive tampering of record 0's risk_probability
    ledger._records[0] = AuditRecord(
        sequence_index=r0.sequence_index,
        transaction_id=r0.transaction_id,
        entity_id=r0.entity_id,
        risk_probability=0.0001,  # tampered!
        decision=r0.decision,
        conformal_lower=r0.conformal_lower,
        conformal_upper=r0.conformal_upper,
        adverse_reasons=r0.adverse_reasons,
        model_version=r0.model_version,
        prev_hash=r0.prev_hash,
        record_hash=r0.record_hash,
    )
    is_valid_after, tampered_idx_after = ledger.verify_chain_integrity()
    assert not is_valid_after
    assert tampered_idx_after == 0

    # Statistical Logit Watermarking verification
    watermarker = StatisticalLogitWatermarker(
        secret_key="watermark-test-key", delta_w=1e-4, z_threshold=3.0
    )
    base_logits = torch.linspace(-2.0, 2.0, 64)
    ctx_ids = torch.arange(1000, 1064)

    wm_logits = watermarker.embed_watermark(base_logits, ctx_ids)
    det_res = watermarker.detect_watermark(wm_logits, base_logits, ctx_ids)
    assert det_res["is_watermarked"]
    assert det_res["z_score"] >= 7.5
    assert det_res["max_prob_shift"] < 1e-4


def test_neural_hawkes_point_process_intensity_and_nll():
    """Verify NeuralHawkesProcessHead conditional intensity, branching_ratio, padding invariance, and NLL."""
    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=12,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        tabular_feature_size=3,
    )
    model.eval()

    keys = torch.randint(1, 12, (4, 6))
    values = torch.randn(4, 6).abs()
    times = torch.randint(0, 8, (4, 6))
    pad_mask = torch.zeros(4, 6, dtype=torch.bool)
    pad_mask[0, 4:] = True

    out = model.forward_hawkes_intensity(
        packed_keys=keys,
        packed_values=values,
        packed_times=times,
        key_padding_mask=pad_mask,
    )
    assert out["conditional_intensity"].shape == (4, 6)
    assert torch.all(out["conditional_intensity"][0, 4:] == 0.0)
    assert torch.all(out["conditional_intensity"][0, :4] > 0.0)
    assert out["branching_ratio"].shape == (4, 1)
    assert torch.all((out["branching_ratio"] > 0.0) & (out["branching_ratio"] < 1.0))
    assert out["expected_next_delta_t"].shape == (4, 1)
    assert torch.all(out["expected_next_delta_t"] > 0.0)
    assert out["next_mark_logits"].shape == (4, 12)

    # Verify point-process NLL backward pass
    full_seq = model.encode(keys, values, times, key_padding_mask=pad_mask)
    token_emb = full_seq[:, model.encoder.num_virtual_tokens :, :]
    nll_dict = model.hawkes_head.compute_point_process_nll(
        token_emb, times, key_padding_mask=pad_mask
    )
    assert nll_dict["loss"].requires_grad
    nll_dict["loss"].backward()


def test_cross_border_fx_normalizer_and_deep_coral_alignment():
    """Verify FXPurchasingPowerNormalizer and CrossBorderDomainAligner Deep CORAL domain adaptation."""
    torch.manual_seed(42)
    model = LargeDataModel(
        vocab_size=12,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        tabular_feature_size=3,
    )

    keys = torch.randint(1, 12, (8, 6))
    # Simulate 4 USD users (currency 0) and 4 JPY users (currency 4 with ~150x nominal scale)
    usd_vals = torch.full((4, 6), 50.0)
    jpy_vals = torch.full((4, 6), 7500.0)
    values = torch.cat([usd_vals, jpy_vals], dim=0)
    times = torch.randint(0, 5, (8, 6))
    currencies = torch.tensor([0, 0, 0, 0, 4, 4, 4, 4], dtype=torch.long)
    tab = torch.randn(8, 3)

    cb_out = model.forward_cross_border_aligned(
        packed_keys=keys,
        packed_values=values,
        packed_times=times,
        currency_ids=currencies,
        tabular_features=tab,
    )
    assert cb_out["ppp_values"].shape == (8, 6)
    assert cb_out["aligned_embeddings"].shape == (8, 16)
    assert cb_out["fraud_logits"].shape == (8, 1)
    # 50 USD and 7500 JPY (~50.25 USD) should normalize to nearly identical PPP log values
    assert torch.allclose(cb_out["ppp_values"][:4], cb_out["ppp_values"][4:], atol=0.05)

    # Verify Deep CORAL alignment loss reduces domain discrepancy
    src_emb = cb_out["aligned_embeddings"][:4]
    tgt_emb = cb_out["aligned_embeddings"][4:] + 1.5
    align_loss_dict = model.cross_border_aligner.compute_alignment_loss(
        src_emb, tgt_emb, currencies[:4], currencies[4:]
    )
    assert align_loss_dict["loss"].requires_grad
    assert float(align_loss_dict["coral_loss"].item()) >= 0.0
    assert float(align_loss_dict["centroid_loss"].item()) > 0.0
    align_loss_dict["loss"].backward()


def test_certified_partially_monotonic_risk_head_and_auditor():
    """Verify MonotonicConstrainedHead guarantees 0.0000 monotonicity violation rate across R^D."""
    from src.models.monotonic_constraints import (
        MonotonicConstrainedHead,
        MonotonicityCertificationAuditor,
    )

    torch.manual_seed(42)
    # 6 features: first 3 unconstrained (0), feature 3 increasing (+1), feature 4 decreasing (-1), feature 5 increasing (+1)
    head = MonotonicConstrainedHead(
        input_dim=6,
        monotonicity_signs=[0, 0, 0, +1, -1, +1],
        hidden_dim=24,
    )
    auditor = MonotonicityCertificationAuditor(tolerance=1e-6)

    ref_inputs = torch.randn(32, 6)
    report = auditor.certify(head, ref_inputs, delta_grid=(0.05, 0.25, 1.0, 3.0))
    assert report.num_constrained_features == 3
    assert report.total_counterfactual_pairs == 32 * 3 * 4
    assert report.num_violations == 0
    assert report.violation_rate == 0.0
    assert report.is_certified_monotonic

    # Verify LargeDataModel.forward_monotonic_risk integration
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        tabular_feature_size=3,
    )
    model.eval()
    keys = torch.randint(1, 10, (4, 5))
    vals = torch.randn(4, 5)
    times = torch.randint(0, 5, (4, 5))
    tab_base = torch.randn(4, 3)

    # Last 2 tabular features have default signs (+1, -1)
    tab_inc_debt = tab_base.clone()
    tab_inc_debt[:, 1] += 2.0  # s = +1 -> risk must increase
    tab_inc_income = tab_base.clone()
    tab_inc_income[:, 2] += 2.0  # s = -1 -> risk must decrease

    with torch.no_grad():
        r_base = model.forward_monotonic_risk(keys, vals, times, tab_base)
        r_debt = model.forward_monotonic_risk(keys, vals, times, tab_inc_debt)
        r_income = model.forward_monotonic_risk(keys, vals, times, tab_inc_income)

    assert torch.all(r_debt >= r_base - 1e-6)
    assert torch.all(r_income <= r_base + 1e-6)


def test_neural_linucb_bandit_router_and_doubly_robust_ope():
    """Verify NeuralLinUCBRouter Sherman-Morrison updates and DoublyRobustOffPolicyEvaluator (DR-OPE)."""
    from src.serving.contextual_bandit import (
        DoublyRobustOffPolicyEvaluator,
        NeuralLinUCBRouter,
    )

    torch.manual_seed(42)
    router = NeuralLinUCBRouter(context_dim=12, alpha_ucb=0.40, ridge_lambda=1.0)
    contexts = torch.randn(16, 12) * 0.25 + 1.0

    initial_decisions = router.select_actions(contexts)
    assert len(initial_decisions) == 16
    assert all(d.ucb_uncertainty > 0.0 for d in initial_decisions)

    # Reward arm 2 ("BIOMETRIC_3DS") strongly on these contexts
    chosen_arms = torch.full((16,), 2, dtype=torch.long)
    rewards = torch.ones(16, dtype=torch.float32) * 2.5
    router.update(contexts, chosen_arms, rewards)
    assert router.action_counts[2] == 16

    updated_decisions = router.select_actions(contexts)
    # After positive reward updates on arm 2, expected reward for arm 2 must increase
    assert all(d.action_index == 2 for d in updated_decisions)
    assert all(d.expected_reward > 0.5 for d in updated_decisions)

    # Verify Doubly Robust Off-Policy Evaluation (DR-OPE)
    ope = DoublyRobustOffPolicyEvaluator(min_propensity_clip=0.05)
    logged_actions = torch.randint(0, 4, (40,))
    logged_rewards = torch.randn(40) + 1.0
    logging_props = torch.full((40,), 0.25)
    target_actions = torch.full((40,), 2, dtype=torch.long)
    r_hat_matrix = router.predict_expected_rewards(torch.randn(40, 12))

    ope_report = ope.evaluate_policy(
        logged_actions=logged_actions,
        logged_rewards=logged_rewards,
        logging_propensities=logging_props,
        target_policy_actions=target_actions,
        estimated_reward_matrix=r_hat_matrix,
    )
    assert ope_report.num_logged_samples == 40
    assert ope_report.dr_ci95_lower <= ope_report.doubly_robust_value <= ope_report.dr_ci95_upper
    assert ope_report.effective_sample_size_pct > 0.0


def test_in_context_tabular_predictor_tabicl_posterior_and_entropy():
    """Verify TabICL / TabPFN v2 Bayesian support-query cross-attention and LargeDataModel.forward_tabicl."""
    from src.models.in_context_ldm import InContextTabularPredictor

    torch.manual_seed(42)
    predictor = InContextTabularPredictor(repr_dim=24, num_heads=4, temperature=0.20)

    # Construct 8 support examples: 4 negative cluster (-1) and 4 positive cluster (+1)
    s_neg = torch.randn(4, 24) * 0.15 - 1.0
    s_pos = torch.randn(4, 24) * 0.15 + 1.0
    support_repr = torch.cat([s_neg, s_pos], dim=0)
    support_labels = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    # Query 0 near negative cluster, Query 1 near positive cluster
    q_neg = torch.full((1, 24), -1.0)
    q_pos = torch.full((1, 24), 1.0)
    query_repr = torch.cat([q_neg, q_pos], dim=0)

    out = predictor(query_repr, support_repr, support_labels)
    probs = out["icl_probabilities"].squeeze(-1)
    assert probs[1] > 0.70
    assert probs[0] < 0.30
    assert torch.allclose(
        out["attention_weights"].sum(dim=-1), torch.ones(2), atol=1e-5
    )
    assert (out["effective_support_neighbors"] >= 1.0).all()

    loss = predictor.compute_icl_episodic_loss(
        query_repr, support_repr, support_labels, torch.tensor([0.0, 1.0])
    )
    assert loss.item() > 0.0

    # Verify end-to-end integration on LargeDataModel
    ldm = LargeDataModel(
        vocab_size=12, hidden_size=16, num_layers=1, num_heads=4, tabular_feature_size=6
    )
    ldm_out = ldm.forward_tabicl(
        query_keys=torch.randint(1, 12, (3, 6)),
        query_values=torch.rand(3, 6) * 50.0,
        query_times=torch.randint(1, 10, (3, 6)),
        query_tabular=torch.randn(3, 6),
        support_keys=torch.randint(1, 12, (6, 6)),
        support_values=torch.rand(6, 6) * 50.0,
        support_times=torch.randint(1, 10, (6, 6)),
        support_tabular=torch.randn(6, 6),
        support_labels=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32),
    )
    assert ldm_out["icl_probabilities"].shape == (3, 1)


def test_competing_risks_survival_head_deephit_and_ifrs9_ecl():
    """Verify CompetingRisksSurvivalHead (DeepHit) CIF monotonicity, right-censoring loss, and IFRS 9 ECL."""
    from src.models.survival_hazard import CompetingRisksSurvivalHead

    torch.manual_seed(42)
    head = CompetingRisksSurvivalHead(repr_dim=22, num_causes=3, num_horizons=8)
    repr_batch = torch.randn(6, 22)
    ead = torch.full((6,), 1000.0)

    out = head(repr_batch, exposure_at_default=ead, stage1_horizon_idx=2)
    joint_sum = out["joint_event_probs"].sum(dim=(1, 2)) + out["survival_beyond_horizon"].squeeze(-1)
    assert torch.allclose(joint_sum, torch.ones(6), atol=1e-5)

    # CIF must be monotonically non-decreasing across horizons
    cif_diffs = out["cif"][:, :, 1:] - out["cif"][:, :, :-1]
    assert (cif_diffs >= -1e-6).all()

    # Survival curve must be monotonically non-increasing across horizons
    surv_diffs = out["survival_curve"][:, 1:] - out["survival_curve"][:, :-1]
    assert (surv_diffs <= 1e-6).all()

    # Lifetime ECL >= Stage-1 ECL everywhere
    assert (out["lifetime_ecl"] >= out["stage1_ecl"] - 1e-5).all()

    # Right-censored + uncensored DeepHit loss
    event_times = torch.tensor([1, 3, 5, 2, 7, 4])
    event_causes = torch.tensor([1, 0, 2, 3, 0, 1])  # 0 = right-censored
    loss_dict = head.compute_deephit_loss(repr_batch, event_times, event_causes)
    assert loss_dict["total_loss"].item() > 0.0

    # Verify LargeDataModel.forward_survival_hazard
    ldm = LargeDataModel(
        vocab_size=12, hidden_size=16, num_layers=1, num_heads=4, tabular_feature_size=6
    )
    surv_out = ldm.forward_survival_hazard(
        packed_keys=torch.randint(1, 12, (4, 6)),
        packed_values=torch.rand(4, 6) * 100.0,
        packed_times=torch.randint(1, 10, (4, 6)),
        tabular_features=torch.randn(4, 6),
        exposure_at_default=torch.tensor([500.0, 1000.0, 2500.0, 750.0]),
    )
    assert surv_out["cif"].shape == (4, 3, 12)
    assert surv_out["lifetime_ecl"].shape == (4, 1)


def test_federated_dp_coordinator_and_renyi_privacy_accountant():
    """Verify DP-FedAvg per-client L2 norm clipping and Rényi Differential Privacy (RDP) accounting."""
    import copy
    from src.training.federated_dp import (
        DifferentiallyPrivateFedCoordinator,
        RdpPrivacyAccountant,
    )

    accountant = RdpPrivacyAccountant(target_delta=1e-5)
    eps1, _ = accountant.step(noise_multiplier=1.2, sample_rate=0.5)
    eps2, _ = accountant.step(noise_multiplier=1.2, sample_rate=0.5)
    assert eps1 > 0.0
    assert eps2 > eps1

    torch.manual_seed(42)
    global_model = torch.nn.Sequential(
        torch.nn.Linear(10, 8), torch.nn.GELU(), torch.nn.Linear(8, 1)
    )
    coordinator = DifferentiallyPrivateFedCoordinator(
        global_model=global_model,
        clip_norm_c=0.50,
        noise_multiplier=1.0,
        target_delta=1e-5,
    )

    base_params = coordinator.extract_trainable_parameters(global_model)
    client_deltas = []
    for k in range(3):
        client_m = copy.deepcopy(global_model)
        with torch.no_grad():
            for p in client_m.parameters():
                p.add_(torch.randn_like(p) * (1.5 + k))
        client_deltas.append(coordinator.compute_client_delta(base_params, client_m))

    report = coordinator.aggregate_and_update(client_deltas, rng_seed=99)
    assert report.round_index == 1
    assert report.num_clients == 3
    assert report.mean_clipped_l2_norm <= 0.50 + 1e-5
    assert report.cumulative_epsilon > 0.0


def test_randomized_smoothing_certified_l2_robustness():
    """Verify RandomizedSmoothingCertifier and exact Clopper-Pearson certified L2 radii."""
    from src.evaluation.certified_robustness import (
        RandomizedSmoothingCertifier,
        clopper_pearson_lower_bound,
    )

    p_low_all = clopper_pearson_lower_bound(successes=32, trials=32, alpha=0.01)
    p_low_half = clopper_pearson_lower_bound(successes=16, trials=32, alpha=0.01)
    assert p_low_all > 0.85
    assert p_low_half < 0.50

    torch.manual_seed(42)
    ldm = LargeDataModel(
        vocab_size=12, hidden_size=16, num_layers=1, num_heads=4, tabular_feature_size=6
    )
    # Bias fusion head output strongly positive so smoothed predictions are deterministic class 1
    with torch.no_grad():
        ldm.fusion_head.final_projection.bias.fill_(8.0)

    certifier = RandomizedSmoothingCertifier(
        noise_sigma=0.30,
        num_mc_samples=24,
        confidence_alpha=0.01,
    )
    report = certifier.certify_batch(
        model=ldm,
        packed_keys=torch.randint(1, 12, (4, 6)),
        packed_values=torch.rand(4, 6) * 20.0,
        packed_times=torch.randint(1, 10, (4, 6)),
        tabular_features=torch.randn(4, 6),
        rng_seed=42,
    )
    assert report.total_evaluated == 4
    assert report.certified_count == 4
    assert report.mean_certified_l2_radius > 0.25
    assert all(pred == 1 for pred in report.smoothed_predictions)


def test_cost_aware_stackelberg_red_team_and_minimax_equilibrium():
    """Verify CostAwareStackelbergRedTeam asymmetric cost gating, equilibrium threshold, and minimax loss."""
    from src.evaluation.stackelberg_red_team import CostAwareStackelbergRedTeam

    torch.manual_seed(42)
    ldm = LargeDataModel(
        vocab_size=12, hidden_size=16, num_layers=1, num_heads=4, tabular_feature_size=6
    )

    # First 3 tabular features are cheap (0.01), last 3 are immutable/expensive (500.0)
    cost_weights = torch.tensor([0.01, 0.01, 0.01, 500.0, 500.0, 500.0])
    red_team = CostAwareStackelbergRedTeam(
        epsilon_max=0.40,
        pgd_steps=5,
        step_size=0.10,
        value_manipulation_cost=0.05,
        tabular_cost_weights=cost_weights,
    )

    keys = torch.randint(1, 12, (6, 5))
    vals = torch.rand(6, 5) * 40.0 + 5.0
    times = torch.randint(1, 10, (6, 5))
    tab = torch.randn(6, 6)
    labels = torch.ones(6)

    adv_vals, adv_tab, _, _ = red_team.compute_attacker_best_response(
        model=ldm,
        packed_keys=keys,
        packed_values=vals,
        packed_times=times,
        tabular_features=tab,
    )
    # Prohibitive-cost features (indices 3, 4, 5) must remain unperturbed
    assert torch.allclose(adv_tab[:, 3:], tab[:, 3:], atol=1e-6)
    assert (adv_vals >= 0.0).all()

    eq_report = red_team.evaluate_stackelberg_equilibrium(
        model=ldm,
        packed_keys=keys,
        packed_values=vals,
        packed_times=times,
        tabular_features=tab,
    )
    assert eq_report.num_attacked_samples == 6
    assert (
        eq_report.stackelberg_hardened_detection_rate
        >= eq_report.post_attack_detection_rate
    )

    minimax_loss = red_team.compute_stackelberg_minimax_loss(
        model=ldm,
        packed_keys=keys,
        packed_values=vals,
        packed_times=times,
        tabular_features=tab,
        labels=labels,
    )
    assert minimax_loss.requires_grad
    assert minimax_loss.item() > 0.0


def test_verifiable_ldm_commitment_engine_merkle_trace_and_tamper_detection():
    """Verify VerifiableLDMCommitmentEngine Merkle weight/trace commitments and tamper detection."""
    import dataclasses
    from src.serving.verifiable_commitment import VerifiableLDMCommitmentEngine

    torch.manual_seed(42)
    ldm = LargeDataModel(
        vocab_size=12, hidden_size=16, num_layers=1, num_heads=4, tabular_feature_size=6, dropout=0.0
    )
    engine = VerifiableLDMCommitmentEngine(
        signing_secret="unit-test-zk-secret",
        decision_threshold=0.50,
    )

    keys = torch.randint(1, 12, (4, 5))
    vals = torch.rand(4, 5) * 30.0
    times = torch.randint(1, 10, (4, 5))
    tab = torch.randn(4, 6)

    receipt = engine.prove_inference(
        model=ldm,
        packed_keys=keys,
        packed_values=vals,
        packed_times=times,
        tabular_features=tab,
    )
    assert len(receipt.weight_merkle_root) == 64
    assert len(receipt.trace_merkle_root) == 64
    assert engine.verify_receipt(
        receipt=receipt,
        model=ldm,
        packed_keys=keys,
        packed_values=vals,
        packed_times=times,
        tabular_features=tab,
    )

    # 1. Tampering with a decision bit invalidates HMAC signature
    forged_receipt = dataclasses.replace(
        receipt, decisions=[1 - d for d in receipt.decisions]
    )
    assert not engine.verify_receipt(forged_receipt, model=ldm)

    # 2. Tampering with input values invalidates input & trace commitments
    assert not engine.verify_receipt(
        receipt=receipt,
        model=ldm,
        packed_keys=keys,
        packed_values=vals + 5.0,
        packed_times=times,
        tabular_features=tab,
    )

    # 3. Mutating a single model parameter invalidates weight_merkle_root
    with torch.no_grad():
        ldm.fusion_head.final_projection.bias.add_(0.05)
    assert not engine.verify_receipt(receipt, model=ldm)




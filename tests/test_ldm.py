import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import numpy as np
import pandas as pd

from src.data.synthetic_generator import SyntheticTransactionGenerator
from src.data.mmtt_dataset import FinancialEventDataset, dynamic_sequence_packing_collate_fn
from src.models.transformer_backbone import MMTTTransformerEncoder
from src.models.joint_fusion_network import CrossNetworkV2, EndToEndFusionLayer
from src.models.ldm_model import LargeDataModel
from src.training.lightning_module import LDMPreTrainingModule, LDMFineTuningModule
from src.evaluation.strategic_shift_tester import StrategicShiftTester


def test_synthetic_generator():
    """Verify synthetic transaction generator output structure and types."""
    generator = SyntheticTransactionGenerator(num_users=10, seed=42)
    df = generator.generate()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "user_id" in df.columns
    assert "mcc_token" in df.columns
    assert "amount" in df.columns
    assert "time_delta" in df.columns
    assert "is_fraud" in df.columns


def test_dataset_and_collate():
    """Verify dataset loading and sequence collation padding masks."""
    generator = SyntheticTransactionGenerator(num_users=10, seed=42)
    df = generator.generate()
    dataset = FinancialEventDataset(df, max_seq_len=20)
    assert len(dataset) == 10

    sample = dataset[0]
    assert "keys" in sample
    assert "values" in sample
    assert "times" in sample
    assert "tabular" in sample
    assert "label" in sample
    assert "seq_len" in sample

    # Create dummy batch of different sequence lengths to test collation
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

    batch = [item1, item2]
    collated = dynamic_sequence_packing_collate_fn(batch)

    assert "packed_keys" in collated
    assert "key_padding_mask" in collated
    assert collated["packed_keys"].shape == (2, 3)
    assert collated["key_padding_mask"].shape == (2, 3)

    # First item has length 3, so no padding: all False
    assert not collated["key_padding_mask"][0].any()
    # Second item has length 2, so third element is padding: True at index 2
    assert not collated["key_padding_mask"][1, 0].item()
    assert not collated["key_padding_mask"][1, 1].item()
    assert collated["key_padding_mask"][1, 2].item()


def test_cross_network_v2():
    """Verify DCNv2 full-rank and low-rank layer forward pass shapes."""
    batch_size = 4
    input_dim = 16

    # Full-rank DCNv2
    cross_full = CrossNetworkV2(input_dim=input_dim, num_layers=3, low_rank=None)
    x = torch.randn(batch_size, input_dim)
    out_full = cross_full(x)
    assert out_full.shape == (batch_size, input_dim)

    # Low-rank DCNv2
    cross_low = CrossNetworkV2(input_dim=input_dim, num_layers=3, low_rank=4)
    out_low = cross_low(x)
    assert out_low.shape == (batch_size, input_dim)


def test_attention_padding_mask_effectiveness():
    """Verify that padding mask prevents representation contamination."""
    # Instantiating the backbone
    encoder = MMTTTransformerEncoder(
        vocab_size=10,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        use_virtual_tokens=True,
        num_virtual_tokens=2,
        pooling="mean"
    )
    encoder.eval()

    # We will build two batches. Both have 1 sample of length 2.
    # Batch A has padding element as 0. Batch B has padding element as 9.
    keys_a = torch.tensor([[1, 2, 0]], dtype=torch.long)
    values_a = torch.tensor([[0.5, 0.8, 0.0]], dtype=torch.float)
    times_a = torch.tensor([[0, 1, 0]], dtype=torch.long)

    keys_b = torch.tensor([[1, 2, 9]], dtype=torch.long)
    values_b = torch.tensor([[0.5, 0.8, -99.9]], dtype=torch.float)
    times_b = torch.tensor([[0, 1, 999]], dtype=torch.long)

    # Padding mask says index 2 is padding (True)
    mask = torch.tensor([[False, False, True]], dtype=torch.bool)

    with torch.no_grad():
        out_a = encoder(keys_a, values_a, times_a, key_padding_mask=mask)
        out_b = encoder(keys_b, values_b, times_b, key_padding_mask=mask)

    # Validate that the representation of valid indices (virtual tokens and tokens 1, 2)
    # is identical despite the padding token's values changing
    # out shapes are [1, 2 (virtual) + 3 (sequence) = 5, 16]
    # Padded index in the output sequence is index 4
    torch.testing.assert_close(out_a[:, :4, :], out_b[:, :4, :], rtol=1e-5, atol=1e-5)


def test_end_to_end_model_and_lightning():
    """Verify pretraining and finetuning Lightning step executions."""
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

    # Test pretraining module step
    pretrain_module = LDMPreTrainingModule(model, config)
    dummy_batch = {
        "packed_keys": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "packed_values": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], dtype=torch.float),
        "packed_times": torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long),
        "key_padding_mask": torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool),
    }

    loss = pretrain_module.training_step(dummy_batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)

    val_loss = pretrain_module.validation_step(dummy_batch, 0)
    assert isinstance(val_loss, torch.Tensor)

    # Test finetuning module step
    finetune_module = LDMFineTuningModule(model, config)
    dummy_ft_batch = {
        "packed_keys": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "packed_values": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], dtype=torch.float),
        "packed_times": torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long),
        "key_padding_mask": torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool),
        "tabular": torch.randn(2, 3),
        "labels": torch.tensor([[0], [1]], dtype=torch.long),
    }

    ft_loss = finetune_module.training_step(dummy_ft_batch, 0)
    assert isinstance(ft_loss, torch.Tensor)
    assert not torch.isnan(ft_loss)


def test_strategic_shift_robustness():
    """Verify adversarial shift tester application and metrics logic."""
    model = LargeDataModel(
        vocab_size=10,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        tabular_feature_size=2,
    )
    model.eval()

    tester = StrategicShiftTester(noise_std=0.1, adversarial_epsilon=0.01)

    keys = torch.tensor([[1, 2]], dtype=torch.long)
    values = torch.tensor([[0.5, -0.2]], dtype=torch.float)
    times = torch.tensor([[0, 1]], dtype=torch.long)
    targets = torch.tensor([0], dtype=torch.long)
    tabular = torch.tensor([[0.1, 0.2]], dtype=torch.float)

    # Test random noise
    noisy = tester.apply_random_shift(values)
    assert noisy.shape == values.shape

    # Test FGSM shift
    perturbed_keys, perturbed_values, perturbed_times = tester.apply_gradient_adversarial_shift(
        model, (keys, values, times), targets
    )
    assert perturbed_values.shape == values.shape
    assert not torch.equal(perturbed_values, values)

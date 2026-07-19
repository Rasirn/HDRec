import inspect

import torch

from analyze_residual_calibration import choose_train_configs
from paired_bootstrap import paired_bootstrap
from reliability_features import (
    FEATURE_NAMES,
    RESIDUAL_SCALE_FEATURE_NAMES,
    compute_residual_scale_features,
    compute_sequence_features,
)
from residual_calibration import calibrate_residual


def test_all_residual_transforms_are_finite_and_reproducible():
    torch.manual_seed(31)
    ids = torch.randn(4, 12)
    configs = [('R0', None), ('R1', None), ('R2', None), ('R3', None), ('R4', None), ('R5', None), ('R6', 5)]
    for transform, topk in configs:
        first = calibrate_residual(ids, transform, fusion_temperature=1.0, topk=topk)
        second = calibrate_residual(ids, transform, fusion_temperature=1.0, topk=topk)
        assert torch.isfinite(first).all()
        assert torch.equal(first, second)


def test_normalized_residual_scales_match_definitions():
    ids = torch.randn(5, 20)
    rms = calibrate_residual(ids, 'R1')
    std = calibrate_residual(ids, 'R2')
    maximum = calibrate_residual(ids, 'R3')
    assert torch.allclose(rms.square().mean(dim=-1).sqrt(), torch.ones(5), atol=1e-5)
    assert torch.allclose(std.std(dim=-1, unbiased=False), torch.ones(5), atol=1e-5)
    assert torch.allclose(maximum.abs().max(dim=-1).values, torch.ones(5), atol=1e-5)


def test_topk_sparse_residual_keeps_only_k_candidates():
    ids = torch.randn(3, 30)
    sparse = calibrate_residual(ids, 'R6', topk=7)
    assert torch.equal((sparse != 0).sum(dim=-1), torch.full((3,), 7))


def test_residual_scale_features_are_finite_without_changing_old_schema():
    text, ids = torch.randn(4, 15), torch.randn(4, 15)
    residual = calibrate_residual(ids, 'R0')
    old_features = compute_sequence_features(text, ids)
    scale_features = compute_residual_scale_features(text, ids, residual)
    assert old_features.shape[-1] == len(FEATURE_NAMES) == 10
    assert scale_features.shape[-1] == len(RESIDUAL_SCALE_FEATURE_NAMES)
    assert torch.isfinite(scale_features).all()


def test_train_config_selector_has_no_valid_argument():
    assert list(inspect.signature(choose_train_configs).parameters) == ['train_cache', 'alpha_grid', 'device']


def test_train_config_selector_runs_from_train_payload_only():
    torch.manual_seed(37)
    cache = {
        'logits_text': torch.randn(4, 8),
        'logits_id': torch.randn(4, 8),
        'labels': torch.tensor([0, 1, 2, 3]),
        'fusion_temperature': 1.0,
    }
    selected, per_transform = choose_train_configs(cache, torch.tensor([0.0, 0.5, 1.0]), 'cpu')
    assert selected['transform'] in {f'R{index}' for index in range(7)}
    assert set(per_transform) == {f'R{index}' for index in range(7)}


def test_paired_bootstrap_is_seed_reproducible():
    differences = torch.linspace(-0.1, 0.2, 40)
    first = paired_bootstrap(differences, num_samples=100, seed=42)
    second = paired_bootstrap(differences, num_samples=100, seed=42)
    assert first == second

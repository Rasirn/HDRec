import pytest
import torch

from alpha_targets import (
    build_alpha_target_payload,
    compute_alpha_outcomes,
    make_alpha_grid,
    rank_first_hard_target,
    soft_targets,
    validate_alpha_target_source,
)


def test_alpha_grid_has_expected_21_bins():
    grid = make_alpha_grid(1.0, 0.05)
    assert grid.numel() == 21
    assert grid[0].item() == 0.0
    assert grid[-1].item() == 1.0
    assert torch.allclose(grid[1:] - grid[:-1], torch.full((20,), 0.05), atol=1e-7)


def test_alpha_outcomes_use_zero_based_rank_and_correct_dcg():
    text = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    ids = torch.zeros_like(text)
    labels = torch.tensor([0, 1])
    outcomes = compute_alpha_outcomes(text, ids, labels, torch.tensor([0.0]), 1.0, chunk_size=2)

    assert torch.equal(outcomes['target_rank'][:, 0], torch.tensor([0, 1], dtype=torch.int32))
    assert outcomes['reciprocal_rank'][0, 0].item() == 1.0
    assert outcomes['reciprocal_rank'][1, 0].item() == 0.5
    assert torch.allclose(outcomes['dcg_at_10'][:, 0], torch.tensor([1.0, 1.0 / torch.log2(torch.tensor(3.0))]))
    assert torch.equal(outcomes['hit_at_5'][:, 0], torch.ones(2))


def test_rank_first_tie_breaks_by_ce_then_smaller_alpha():
    ranks = torch.tensor([[2, 1, 1], [1, 1, 1]])
    ce = torch.tensor([[0.1, 0.8, 0.2], [0.4, 0.4, 0.5]])
    indices = rank_first_hard_target(ranks, ce)
    assert torch.equal(indices, torch.tensor([2, 0]))


def test_soft_target_distributions_sum_to_one():
    ce = torch.tensor([[1.0, 0.5, 2.0], [0.1, 0.2, 0.3]])
    dcg = torch.tensor([[0.0, 1.0, 0.5], [0.2, 0.2, 0.2]])
    rr = torch.tensor([[0.1, 1.0, 0.5], [0.3, 0.3, 0.3]])
    ce_prob, metric_prob, reward = soft_targets(ce, dcg, rr)
    assert torch.allclose(ce_prob.sum(dim=-1), torch.ones(2))
    assert torch.allclose(metric_prob.sum(dim=-1), torch.ones(2))
    assert torch.isfinite(reward).all()


def test_alpha_target_payload_and_legacy_failure(tmp_path):
    cache_path = tmp_path / 'cache.pt'
    cache = {
        'dataset': 'synthetic',
        'split': 'calibration_train',
        'version': 'v4_cache_v1',
        'feature_schema': 'features-v1',
        'feature_names': ['x'],
        'checkpoint_path': '/tmp/base.bin',
        'fusion_temperature': 1.0,
        'alpha0': 0.5,
        'user_ids': torch.tensor([1]),
    }
    grid = torch.tensor([0.0, 0.5, 1.0])
    outcomes = {
        'cross_entropy': torch.tensor([[1.0, 0.5, 0.7]]),
        'target_rank': torch.tensor([[2, 1, 1]], dtype=torch.int32),
        'reciprocal_rank': torch.tensor([[1 / 3, 0.5, 0.5]]),
        'dcg_at_5': torch.tensor([[0.5, 0.6, 0.6]]),
        'dcg_at_10': torch.tensor([[0.5, 0.6, 0.6]]),
        'hit_at_5': torch.ones(1, 3),
        'hit_at_10': torch.ones(1, 3),
    }
    payload = build_alpha_target_payload(cache, cache_path, grid, outcomes)
    validate_alpha_target_source(payload, cache, cache_path)
    assert payload['best_alpha_rank'].item() == 0.5
    with pytest.raises(ValueError, match='legacy alpha target'):
        validate_alpha_target_source({'dataset': 'synthetic'}, cache, cache_path)

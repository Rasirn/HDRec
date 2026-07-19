from copy import deepcopy

import pytest
import torch

from cache_compatibility import validate_cache_compatibility, validate_fuser_cache_compatibility


def make_cache(users):
    n = len(users)
    return {
        'dataset': 'Industrial_and_Scientific',
        'version': 'v4_cache_v1',
        'feature_schema': 'schema-v2',
        'feature_names': ['a', 'b'],
        'checkpoint_path': '/tmp/v1.bin',
        'alpha0': 0.5,
        'utility_alpha0': 0.5,
        'fusion_temperature': 1.0,
        'utility_target': 'ce_at_alpha0',
        'logits_text': torch.randn(n, 5),
        'logits_id': torch.randn(n, 5),
        'utility_ce_label': torch.arange(n).remainder(2),
        'utility_rank_label': torch.arange(n).remainder(2),
        'user_ids': torch.tensor(users),
    }


def test_cache_compatibility_accepts_matching_user_disjoint_caches():
    train = make_cache([1, 2, 3])
    valid = make_cache([4, 5])
    validate_cache_compatibility(train, valid, '/tmp/train.pt', '/tmp/valid.pt')


def test_cache_compatibility_rejects_metadata_mismatch():
    train = make_cache([1, 2, 3])
    valid = make_cache([4, 5])
    valid['fusion_temperature'] = 1.2
    with pytest.raises(ValueError, match='fusion_temperature'):
        validate_cache_compatibility(train, valid, '/tmp/train.pt', '/tmp/valid.pt')


def test_fuser_cache_compatibility_checks_base_provenance():
    cache = make_cache([4, 5])
    fuser = {
        'dataset': cache['dataset'],
        'feature_schema': cache['feature_schema'],
        'feature_names': cache['feature_names'],
        'alpha0': cache['alpha0'],
        'fusion_temperature': cache['fusion_temperature'],
        'base_checkpoint_path': cache['checkpoint_path'],
        'utility_target': cache['utility_target'],
    }
    validate_fuser_cache_compatibility(fuser, cache)
    mismatched = deepcopy(fuser)
    mismatched['base_checkpoint_path'] = '/tmp/other.bin'
    with pytest.raises(ValueError, match='base checkpoints'):
        validate_fuser_cache_compatibility(mismatched, cache)

from pathlib import Path

import torch


EXACT_CACHE_FIELDS = (
    'dataset',
    'version',
    'feature_schema',
    'feature_names',
    'checkpoint_path',
)
FLOAT_CACHE_FIELDS = ('alpha0', 'utility_alpha0', 'fusion_temperature')


def _normalized_path(value):
    return str(Path(value).expanduser().resolve())


def _require_fields(payload, fields, name):
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f'{name} is missing required fields: {missing}')


def validate_cache_compatibility(train_cache, valid_cache, train_path=None, valid_path=None, atol=1e-8):
    required = EXACT_CACHE_FIELDS + FLOAT_CACHE_FIELDS + (
        'logits_text', 'logits_id', 'utility_ce_label', 'utility_rank_label'
    )
    _require_fields(train_cache, required, 'train cache')
    _require_fields(valid_cache, required, 'valid cache')

    if train_path is not None and valid_path is not None:
        if _normalized_path(train_path) == _normalized_path(valid_path):
            raise ValueError('train_cache and valid_cache must be different files.')

    for field in EXACT_CACHE_FIELDS:
        train_value = train_cache[field]
        valid_value = valid_cache[field]
        if field == 'checkpoint_path':
            train_value = _normalized_path(train_value)
            valid_value = _normalized_path(valid_value)
        if train_value != valid_value:
            raise ValueError(
                f'Cache incompatibility for {field}: train={train_value!r}, valid={valid_value!r}'
            )

    for field in FLOAT_CACHE_FIELDS:
        if not torch.isclose(
            torch.tensor(float(train_cache[field])),
            torch.tensor(float(valid_cache[field])),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError(
                f'Cache incompatibility for {field}: '
                f'train={train_cache[field]!r}, valid={valid_cache[field]!r}'
            )

    train_dim = train_cache['logits_text'].size(-1)
    valid_dim = valid_cache['logits_text'].size(-1)
    if train_cache['logits_id'].size(-1) != train_dim:
        raise ValueError('Train text/id logits candidate dimensions differ.')
    if valid_cache['logits_id'].size(-1) != valid_dim:
        raise ValueError('Valid text/id logits candidate dimensions differ.')
    if train_dim != valid_dim:
        raise ValueError(f'Cache candidate dimensions differ: train={train_dim}, valid={valid_dim}')

    if 'user_ids' in train_cache and 'user_ids' in valid_cache:
        overlap = set(train_cache['user_ids'].tolist()).intersection(valid_cache['user_ids'].tolist())
        if overlap:
            raise ValueError(f'Calibration user leakage detected: {len(overlap)} overlapping users.')


def validate_fuser_cache_compatibility(fuser_checkpoint, cache, atol=1e-8):
    fuser_fields = (
        'dataset', 'feature_schema', 'feature_names', 'alpha0',
        'fusion_temperature', 'base_checkpoint_path', 'utility_target',
    )
    cache_fields = (
        'dataset', 'feature_schema', 'feature_names', 'alpha0',
        'fusion_temperature', 'checkpoint_path', 'utility_target',
    )
    _require_fields(fuser_checkpoint, fuser_fields, 'fuser checkpoint')
    _require_fields(cache, cache_fields, 'cache')

    exact_pairs = (
        ('dataset', 'dataset'),
        ('feature_schema', 'feature_schema'),
        ('feature_names', 'feature_names'),
        ('utility_target', 'utility_target'),
    )
    for fuser_field, cache_field in exact_pairs:
        if fuser_checkpoint[fuser_field] != cache[cache_field]:
            raise ValueError(
                f'Fuser/cache incompatibility for {fuser_field}: '
                f'fuser={fuser_checkpoint[fuser_field]!r}, cache={cache[cache_field]!r}'
            )

    if _normalized_path(fuser_checkpoint['base_checkpoint_path']) != _normalized_path(cache['checkpoint_path']):
        raise ValueError('Fuser and cache use different base checkpoints.')
    for field in ('alpha0', 'fusion_temperature'):
        if not torch.isclose(
            torch.tensor(float(fuser_checkpoint[field])),
            torch.tensor(float(cache[field])),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError(
                f'Fuser/cache incompatibility for {field}: '
                f'fuser={fuser_checkpoint[field]!r}, cache={cache[field]!r}'
            )

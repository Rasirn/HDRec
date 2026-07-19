import argparse
import json
import random
from pathlib import Path

import torch


def _stratified_sample_split(labels, valid_ratio, seed):
    generator = torch.Generator().manual_seed(seed)
    valid_parts = []
    for value in torch.unique(labels.long(), sorted=True):
        indices = torch.where(labels.long() == value)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        take = int(round(indices.numel() * valid_ratio))
        if indices.numel() > 1:
            take = min(max(take, 1), indices.numel() - 1)
        valid_parts.append(indices[:take])
    valid_idx = torch.cat(valid_parts) if valid_parts else torch.empty(0, dtype=torch.long)
    valid_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    valid_mask[valid_idx] = True
    train_idx = torch.where(~valid_mask)[0]
    return train_idx.sort().values, valid_idx.sort().values


def split_indices(cache, valid_ratio=0.2, seed=42):
    labels = cache['utility_label'].long().cpu()
    n = labels.numel()
    if n < 2:
        raise ValueError('Calibration cache must contain at least two samples.')
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError('valid_ratio must be between 0 and 1.')

    user_ids = cache.get('user_ids')
    if user_ids is None:
        train_idx, valid_idx = _stratified_sample_split(labels, valid_ratio, seed)
        return train_idx, valid_idx, 'deterministic_stratified_sample'

    user_ids = user_ids.long().cpu()
    if user_ids.numel() != n:
        raise ValueError('user_ids and utility_label have different lengths.')
    unique_users = torch.unique(user_ids)
    if unique_users.numel() < 2:
        train_idx, valid_idx = _stratified_sample_split(labels, valid_ratio, seed)
        return train_idx, valid_idx, 'deterministic_stratified_sample_single_user'

    groups = []
    for user_id in unique_users.tolist():
        idx = torch.where(user_ids == user_id)[0]
        groups.append((int(user_id), idx, int(labels[idx].sum().item())))
    random.Random(seed).shuffle(groups)

    target_n = n * valid_ratio
    target_pos = labels.sum().item() * valid_ratio
    valid_groups = []
    valid_n = 0
    valid_pos = 0
    remaining_groups = len(groups)
    for group in groups:
        remaining_groups -= 1
        _, idx, pos = group
        current_score = abs(valid_n - target_n) / n + abs(valid_pos - target_pos) / max(1, labels.sum().item())
        next_score = abs(valid_n + idx.numel() - target_n) / n + abs(valid_pos + pos - target_pos) / max(1, labels.sum().item())
        if next_score <= current_score and remaining_groups >= 1:
            valid_groups.append(group)
            valid_n += idx.numel()
            valid_pos += pos

    if not valid_groups:
        valid_groups = [groups[0]]
    if len(valid_groups) == len(groups):
        valid_groups.pop()

    valid_users = {group[0] for group in valid_groups}
    valid_mask = torch.tensor([int(user_id) in valid_users for user_id in user_ids], dtype=torch.bool)
    train_idx = torch.where(~valid_mask)[0]
    valid_idx = torch.where(valid_mask)[0]
    if train_idx.numel() == 0 or valid_idx.numel() == 0:
        raise RuntimeError('Unable to create non-empty user-disjoint calibration splits.')
    return train_idx, valid_idx, 'user_disjoint_greedy_stratified'


def subset_cache(cache, indices, split_name, source_path):
    n = cache['utility_label'].numel()
    result = {}
    for key, value in cache.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == n:
            result[key] = value[indices]
        else:
            result[key] = value
    result['split'] = split_name
    result['source_cache'] = str(Path(source_path).resolve())
    result['split_indices'] = indices.clone()
    return result


def cache_stats(cache):
    labels = cache['utility_label'].float()
    user_ids = cache.get('user_ids')
    return {
        'num_samples': int(labels.numel()),
        'num_users': int(torch.unique(user_ids).numel()) if user_ids is not None else None,
        'utility_positive_ratio': labels.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Create leakage-safe calibration train/valid caches.')
    parser.add_argument('--input_cache', required=True)
    parser.add_argument('--train_output', required=True)
    parser.add_argument('--valid_output', required=True)
    parser.add_argument('--index_output', default=None)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output_paths = [Path(args.train_output), Path(args.valid_output)]
    index_path = Path(args.index_output) if args.index_output else output_paths[0].parent / 'calibration_split_indices.pt'
    for path in output_paths + [index_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'Output exists; pass --overwrite to replace it: {path}')

    cache = torch.load(args.input_cache, map_location='cpu')
    train_idx, valid_idx, mode = split_indices(cache, args.valid_ratio, args.seed)
    train_cache = subset_cache(cache, train_idx, 'calibration_train', args.input_cache)
    valid_cache = subset_cache(cache, valid_idx, 'calibration_valid', args.input_cache)

    if 'user_ids' in cache:
        overlap = set(train_cache['user_ids'].tolist()).intersection(valid_cache['user_ids'].tolist())
        if mode.startswith('user_disjoint') and overlap:
            raise RuntimeError(f'User-disjoint split failed with {len(overlap)} overlapping users.')
    else:
        print('WARNING: input cache has no user_ids; using deterministic sample-level stratification.')

    for path in output_paths + [index_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, output_paths[0])
    torch.save(valid_cache, output_paths[1])
    manifest = {
        'source_cache': str(Path(args.input_cache).resolve()),
        'seed': args.seed,
        'valid_ratio': args.valid_ratio,
        'mode': mode,
        'train_indices': train_idx,
        'valid_indices': valid_idx,
        'train_stats': cache_stats(train_cache),
        'valid_stats': cache_stats(valid_cache),
    }
    torch.save(manifest, index_path)
    printable = {key: value for key, value in manifest.items() if not torch.is_tensor(value)}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f'Saved calibration train cache: {output_paths[0]}')
    print(f'Saved calibration valid cache: {output_paths[1]}')
    print(f'Saved split indices: {index_path}')


if __name__ == '__main__':
    main()

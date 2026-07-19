import torch

from split_calibration import split_indices, subset_cache


def test_calibration_users_do_not_overlap():
    user_ids = torch.repeat_interleave(torch.arange(20), 3)
    labels = (torch.arange(user_ids.numel()) % 3 == 0).long()
    cache = {
        'dataset': 'synthetic',
        'split': 'valid',
        'user_ids': user_ids,
        'utility_label': labels,
        'features': torch.randn(user_ids.numel(), 10),
    }

    train_idx, valid_idx, mode = split_indices(cache, valid_ratio=0.2, seed=42)
    train = subset_cache(cache, train_idx, 'calibration_train', 'synthetic.pt')
    valid = subset_cache(cache, valid_idx, 'calibration_valid', 'synthetic.pt')

    assert mode == 'user_disjoint_greedy_stratified'
    assert set(train['user_ids'].tolist()).isdisjoint(valid['user_ids'].tolist())
    assert train_idx.numel() + valid_idx.numel() == user_ids.numel()
    assert torch.equal(torch.cat([train_idx, valid_idx]).sort().values, torch.arange(user_ids.numel()))


def test_calibration_split_is_reproducible():
    cache = {
        'user_ids': torch.repeat_interleave(torch.arange(10), 2),
        'utility_label': torch.arange(20).remainder(2),
    }
    first = split_indices(cache, valid_ratio=0.3, seed=9)
    second = split_indices(cache, valid_ratio=0.3, seed=9)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])

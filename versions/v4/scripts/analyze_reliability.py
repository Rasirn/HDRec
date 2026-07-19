import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_cache, save_json
from cache_compatibility import validate_cache_compatibility
from reliability_fusion import FeatureStandardizer


def binary_metrics(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    acc = (preds == labels).float().mean().item()
    precision = (tp / (tp + fp).clamp_min(1)).item()
    recall = (tp / (tp + fn).clamp_min(1)).item()
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    brier = ((probs - labels.float()) ** 2).mean().item()
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'brier': brier}


def auc_score(logits, labels):
    labels = labels.long()
    positive_count = int((labels == 1).sum().item())
    negative_count = int((labels == 0).sum().item())
    if positive_count == 0 or negative_count == 0:
        return float('nan')
    order = torch.argsort(logits)
    sorted_scores = logits[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0).float()
    starts = ends - counts.float() + 1.0
    average_ranks = torch.repeat_interleave((starts + ends) / 2.0, counts)
    ranks = torch.empty_like(average_ranks)
    ranks[order] = average_ranks
    pos_ranks = ranks[labels == 1]
    auc = (
        pos_ranks.sum() - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return auc.item()


def fit_linear_predictor(x_train, y_train, x_valid, y_valid, args, seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.size(1), 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        logits = model(x_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        valid_logits = model(x_valid).squeeze(-1)
        return {
            'auc': auc_score(valid_logits, y_valid.long()),
            **binary_metrics(valid_logits, y_valid.long()),
            'train_positive_rate': y_train.float().mean().item(),
            'valid_positive_rate': y_valid.float().mean().item(),
            'num_train_samples': int(y_train.numel()),
            'num_valid_samples': int(y_valid.numel()),
            'linear_coefficients': model.weight.squeeze(0).tolist(),
            'bias': model.bias.item(),
        }


def subset_payload(cache, indices):
    n = cache['features'].size(0)
    return {
        key: value[indices] if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == n else value
        for key, value in cache.items()
    }


def main():
    parser = argparse.ArgumentParser(description='Report CE and rank utility predictability.')
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--valid_cache', default=None, help='Independent calibration-valid cache.')
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cache = load_cache(args.cache_path)
    x = cache['features'].float()
    y_ce = cache['utility_ce_label'].float()
    y_rank = cache['utility_rank_label'].float()
    n = x.size(0)
    if args.valid_cache:
        valid_cache = load_cache(args.valid_cache)
        validate_cache_compatibility(cache, valid_cache, args.cache_path, args.valid_cache)
        x_train, y_ce_train, y_rank_train = x, y_ce, y_rank
        x_valid = valid_cache['features'].float()
        y_ce_valid = valid_cache['utility_ce_label'].float()
        y_rank_valid = valid_cache['utility_rank_label'].float()
    else:
        perm = torch.randperm(n)
        cut = max(1, int(n * 0.8))
        train_idx, valid_idx = perm[:cut], perm[cut:]
        if valid_idx.numel() == 0:
            valid_idx = train_idx
        x_train, y_ce_train, y_rank_train = x[train_idx], y_ce[train_idx], y_rank[train_idx]
        x_valid = x[valid_idx]
        y_ce_valid, y_rank_valid = y_ce[valid_idx], y_rank[valid_idx]
        valid_cache = subset_payload(cache, valid_idx)
        cache = subset_payload(cache, train_idx)
        print('WARNING: no --valid_cache supplied; using a sample-level diagnostic split.')

    scaler = FeatureStandardizer().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)

    ce_result = fit_linear_predictor(x_train, y_ce_train, x_valid, y_ce_valid, args, args.seed)
    rank_result = fit_linear_predictor(x_train, y_rank_train, x_valid, y_rank_valid, args, args.seed + 1)

    train_non_tie = ~cache['utility_rank_tie'].bool()
    valid_non_tie = ~valid_cache['utility_rank_tie'].bool()
    rank_non_tie_result = None
    if (
        train_non_tie.sum() >= 4
        and valid_non_tie.sum() >= 4
        and torch.unique(y_rank_train[train_non_tie]).numel() == 2
        and torch.unique(y_rank_valid[valid_non_tie]).numel() == 2
    ):
        rank_non_tie_result = fit_linear_predictor(
            x_train[train_non_tie],
            y_rank_train[train_non_tie],
            x_valid[valid_non_tie],
            y_rank_valid[valid_non_tie],
            args,
            args.seed + 2,
        )

    rank_improved = valid_cache['utility_rank_label'].bool()
    rank_harmed = valid_cache['utility_rank_harm'].bool()
    rank_tie = valid_cache['utility_rank_tie'].bool()
    result = {
        'dataset': cache.get('dataset'),
        'split': cache.get('split'),
        'num_samples': int(y_ce_train.numel()),
        'num_valid_samples': int(y_ce_valid.numel()),
        'feature_names': cache['feature_names'],
        'ce_utility': ce_result,
        'rank_utility_improved_vs_non_improved': rank_result,
        'rank_utility_improved_vs_harmed': rank_non_tie_result,
        'ce_utility_auc': ce_result['auc'],
        'rank_utility_auc': rank_result['auc'],
        'ce_rank_label_agreement': (y_ce_valid.long() == y_rank_valid.long()).float().mean().item(),
        'rank_improved_rate': rank_improved.float().mean().item(),
        'rank_harmed_rate': rank_harmed.float().mean().item(),
        'rank_tie_rate': rank_tie.float().mean().item(),
    }

    out_path = args.output_json or str(Path(args.cache_path).with_suffix('.utility.json'))
    save_json(out_path, result)
    print(result)
    print(f'Saved utility analysis: {out_path}')


if __name__ == '__main__':
    main()

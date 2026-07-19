import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_cache, save_json
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
    pos = logits[labels == 1]
    neg = logits[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float('nan')
    scores = torch.cat([pos, neg])
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_ranks = ranks[:pos.numel()]
    auc = (pos_ranks.sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
    return auc.item()


def main():
    parser = argparse.ArgumentParser(description='Train a linear utility predictor and report predictability metrics.')
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
    y = cache['utility_label'].float()
    n = x.size(0)
    if args.valid_cache:
        valid_cache = load_cache(args.valid_cache)
        if cache.get('dataset') != valid_cache.get('dataset'):
            raise ValueError('Training and validation utility caches use different datasets.')
        if 'user_ids' in cache and 'user_ids' in valid_cache:
            overlap = set(cache['user_ids'].tolist()).intersection(valid_cache['user_ids'].tolist())
            if overlap:
                raise ValueError(f'Utility analysis user leakage: {len(overlap)} overlapping users.')
        x_train, y_train = x, y
        x_valid = valid_cache['features'].float()
        y_valid = valid_cache['utility_label'].float()
    else:
        perm = torch.randperm(n)
        cut = max(1, int(n * 0.8))
        train_idx, valid_idx = perm[:cut], perm[cut:]
        if valid_idx.numel() == 0:
            valid_idx = train_idx
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]
        print('WARNING: no --valid_cache supplied; using a sample-level diagnostic split.')

    scaler = FeatureStandardizer().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)

    model = torch.nn.Linear(x.size(1), 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        logits = model(x_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        valid_logits = model(x_valid).squeeze(-1)
        result = {
            'dataset': cache.get('dataset'),
            'split': cache.get('split'),
            'num_samples': int(n),
            'num_valid_samples': int(y_valid.numel()),
            'positive_rate': y.mean().item(),
            'auc': auc_score(valid_logits, y_valid.long()),
            **binary_metrics(valid_logits, y_valid.long()),
            'feature_names': cache['feature_names'],
            'linear_coefficients': model.weight.squeeze(0).tolist(),
            'bias': model.bias.item(),
        }

    out_path = args.output_json or str(Path(args.cache_path).with_suffix('.utility.json'))
    save_json(out_path, result)
    print(result)
    print(f'Saved utility analysis: {out_path}')


if __name__ == '__main__':
    main()

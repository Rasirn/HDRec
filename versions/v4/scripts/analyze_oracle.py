import argparse
from pathlib import Path

import torch

from common import load_cache, save_json
from ranking_metrics import ranking_metrics, target_ranks
from utility_label import fixed_text_fusion, id_confidence_residual


def oracle_select_scores(text_logits, fixed_logits, labels):
    text_ranks = target_ranks(text_logits, labels)
    fixed_ranks = target_ranks(fixed_logits, labels)
    use_fixed = fixed_ranks < text_ranks
    scores = text_logits.clone()
    scores[use_fixed] = fixed_logits[use_fixed]
    return scores, use_fixed


def oracle_alpha_scores(text_logits, id_logits, labels, alpha_grid, temperature):
    residual = id_confidence_residual(id_logits, temperature=temperature)
    best_scores = text_logits.clone()
    best_ranks = target_ranks(text_logits, labels)
    best_alpha = torch.zeros(labels.numel(), dtype=torch.float32)
    for alpha in alpha_grid:
        scores = text_logits + float(alpha) * residual
        ranks = target_ranks(scores, labels)
        better = ranks < best_ranks
        best_ranks[better] = ranks[better]
        best_scores[better] = scores[better]
        best_alpha[better] = float(alpha)
    return best_scores, best_alpha


def bucket_alpha(cache, best_alpha):
    out = {}
    for name in ['history_length', 'history_pop_mean', 'history_pop_std']:
        vals = cache.get(name)
        if vals is None:
            continue
        vals = vals.float()
        qs = torch.quantile(vals, torch.tensor([0.33, 0.66]))
        buckets = torch.bucketize(vals, qs)
        out[name] = {}
        for b in range(3):
            mask = buckets == b
            if mask.any():
                out[name][str(b)] = {
                    'count': int(mask.sum().item()),
                    'alpha_mean': best_alpha[mask].mean().item(),
                    'alpha0_rate': (best_alpha[mask] == 0).float().mean().item(),
                }
    return out


def main():
    parser = argparse.ArgumentParser(description='Run Oracle-Select and Oracle-Alpha diagnostics from cached logits.')
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--alpha_max', type=float, default=None)
    parser.add_argument('--alpha_step', type=float, default=0.1)
    args = parser.parse_args()

    cache = load_cache(args.cache_path)
    text = cache['logits_text'].float()
    ids = cache['logits_id'].float()
    labels = cache['labels'].long()
    alpha0 = float(cache.get('alpha0', 0.5))
    alpha_max = float(args.alpha_max if args.alpha_max is not None else max(1.0, alpha0))
    temp = float(cache.get('fusion_temperature', 1.0))

    fixed = fixed_text_fusion(text, ids, alpha=alpha0, temperature=temp)
    select_scores, select_use_fixed = oracle_select_scores(text, fixed, labels)
    grid = torch.arange(0.0, alpha_max + args.alpha_step / 2, args.alpha_step).tolist()
    alpha_scores, best_alpha = oracle_alpha_scores(text, ids, labels, grid, temp)

    result = {
        'dataset': cache.get('dataset'),
        'split': cache.get('split'),
        'alpha0': alpha0,
        'alpha_grid': grid,
        'text_only': ranking_metrics(text, labels),
        'v1_fixed': ranking_metrics(fixed, labels),
        'oracle_select': ranking_metrics(select_scores, labels),
        'oracle_alpha': ranking_metrics(alpha_scores, labels),
        'oracle_select_use_fixed_rate': select_use_fixed.float().mean().item(),
        'oracle_alpha_zero_rate': (best_alpha == 0).float().mean().item(),
        'oracle_alpha_mean': best_alpha.mean().item(),
        'oracle_alpha_distribution': {str(a): int((best_alpha == float(a)).sum().item()) for a in grid},
        'oracle_alpha_buckets': bucket_alpha(cache, best_alpha),
    }

    out_path = args.output_json or str(Path(args.cache_path).with_suffix('.oracle.json'))
    save_json(out_path, result)
    print(result)
    print(f'Saved oracle analysis: {out_path}')


if __name__ == '__main__':
    main()

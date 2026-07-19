import argparse
from pathlib import Path

import torch

from common import load_cache, save_json
from ranking_metrics import ranking_metrics, target_ranks
from utility_label import fixed_text_fusion, id_confidence_residual


CURVE_METRICS = ('Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10', 'MRR')


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
    best_alpha = torch.zeros(labels.numel(), dtype=torch.float32, device=text_logits.device)
    for alpha in alpha_grid:
        scores = text_logits + float(alpha) * residual
        ranks = target_ranks(scores, labels)
        better = ranks < best_ranks
        best_ranks[better] = ranks[better]
        best_scores[better] = scores[better]
        best_alpha[better] = float(alpha)
    return best_scores, best_alpha


def global_alpha_curve(text_logits, id_logits, labels, alpha_grid, temperature):
    residual = id_confidence_residual(id_logits, temperature=temperature)
    curve = []
    for alpha in alpha_grid:
        metrics = ranking_metrics(text_logits + float(alpha) * residual, labels)
        curve.append({'alpha': float(alpha), **{name: metrics[name] for name in CURVE_METRICS}})
    best = max(curve, key=lambda row: row['NDCG@10'])
    return curve, best


def alpha_bin_masks(best_alpha, alpha0, atol=1e-6):
    at_zero = torch.isclose(best_alpha, torch.zeros_like(best_alpha), atol=atol, rtol=0.0)
    at_alpha0 = torch.isclose(
        best_alpha, torch.full_like(best_alpha, alpha0), atol=atol, rtol=0.0
    )
    return {
        '0': at_zero,
        '(0, alpha0)': (best_alpha > atol) & (best_alpha < alpha0 - atol),
        'alpha0': at_alpha0,
        '(alpha0, alpha_max]': best_alpha > alpha0 + atol,
    }


def utility_alpha_cross_table(labels, best_alpha, alpha0):
    bins = alpha_bin_masks(best_alpha, alpha0)
    table = {}
    for label_value in (0, 1):
        label_mask = labels.long() == label_value
        count = int(label_mask.sum().item())
        row = {'count': count}
        for name, bin_mask in bins.items():
            bin_count = int((label_mask & bin_mask).sum().item())
            row[name] = {
                'count': bin_count,
                'rate': bin_count / count if count else float('nan'),
            }
        table[str(label_value)] = row
    return table


def direction_statistics(labels, best_alpha, alpha0):
    bins = alpha_bin_masks(best_alpha, alpha0)
    positive = labels.long() == 1
    negative = labels.long() == 0

    def rate(mask, condition):
        denominator = int(condition.sum().item())
        return (condition & mask).sum().item() / denominator if denominator else float('nan')

    positive_above = rate(bins['(alpha0, alpha_max]'], positive)
    positive_below = rate(bins['0'] | bins['(0, alpha0)'], positive)
    positive_equal = rate(bins['alpha0'], positive)
    negative_zero = rate(bins['0'], negative)
    negative_below = rate(bins['0'] | bins['(0, alpha0)'], negative)
    negative_above = rate(bins['(alpha0, alpha_max]'], negative)
    return {
        'P(best_alpha > alpha0 | utility=1)': positive_above,
        'P(best_alpha < alpha0 | utility=1)': positive_below,
        'P(best_alpha == alpha0 | utility=1)': positive_equal,
        'P(best_alpha == 0 | utility=0)': negative_zero,
        'P(best_alpha < alpha0 | utility=0)': negative_below,
        'P(best_alpha > alpha0 | utility=0)': negative_above,
        'positive_direction_supported': positive_above > max(positive_below, positive_equal),
        'negative_direction_supported': negative_below > negative_above,
    }


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
        for bucket in range(3):
            mask = buckets == bucket
            if mask.any():
                out[name][str(bucket)] = {
                    'count': int(mask.sum().item()),
                    'alpha_mean': best_alpha[mask].mean().item(),
                    'alpha_zero_rate': (best_alpha[mask] == 0).float().mean().item(),
                }
    return out


def main():
    parser = argparse.ArgumentParser(description='Run Oracle, global-alpha and direction diagnostics.')
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--alpha_max', type=float, default=None)
    parser.add_argument('--alpha_step', type=float, default=0.05)
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
    step_count = int(round(alpha_max / args.alpha_step))
    grid = [round(index * args.alpha_step, 10) for index in range(step_count + 1)]
    if grid[-1] < alpha_max:
        grid.append(alpha_max)
    alpha_scores, best_alpha = oracle_alpha_scores(text, ids, labels, grid, temp)
    curve, best_global = global_alpha_curve(text, ids, labels, grid, temp)

    ce_labels = cache.get('utility_ce_label', cache.get('utility_label')).long()
    rank_labels = cache['utility_rank_label'].long()
    result = {
        'dataset': cache.get('dataset'),
        'split': cache.get('split'),
        'alpha0': alpha0,
        'alpha_grid': grid,
        'text_only': ranking_metrics(text, labels),
        'v1_fixed': ranking_metrics(fixed, labels),
        'best_global_alpha': best_global['alpha'],
        'best_global_metrics': {key: best_global[key] for key in CURVE_METRICS},
        'global_alpha_curve': curve,
        'oracle_select': ranking_metrics(select_scores, labels),
        'oracle_alpha': ranking_metrics(alpha_scores, labels),
        'oracle_select_use_fixed_rate': select_use_fixed.float().mean().item(),
        'oracle_alpha_zero_rate': (best_alpha == 0).float().mean().item(),
        'oracle_alpha_mean': best_alpha.mean().item(),
        'best_alpha': best_alpha.tolist(),
        'oracle_alpha_distribution': {
            str(alpha): int(torch.isclose(best_alpha, torch.full_like(best_alpha, alpha), atol=1e-6, rtol=0.0).sum().item())
            for alpha in grid
        },
        'oracle_alpha_buckets': bucket_alpha(cache, best_alpha),
        'ce_utility_direction': direction_statistics(ce_labels, best_alpha, alpha0),
        'rank_utility_direction': direction_statistics(rank_labels, best_alpha, alpha0),
        'ce_utility_by_oracle_alpha_bin': utility_alpha_cross_table(ce_labels, best_alpha, alpha0),
        'rank_utility_by_oracle_alpha_bin': utility_alpha_cross_table(rank_labels, best_alpha, alpha0),
    }

    out_path = args.output_json or str(Path(args.cache_path).with_suffix('.oracle.json'))
    save_json(out_path, result)
    print({
        'dataset': result['dataset'],
        'split': result['split'],
        'alpha0': result['alpha0'],
        'text_only': result['text_only'],
        'v1_fixed': result['v1_fixed'],
        'best_global_alpha': result['best_global_alpha'],
        'best_global_metrics': result['best_global_metrics'],
        'oracle_alpha': result['oracle_alpha'],
        'ce_utility_direction': result['ce_utility_direction'],
        'rank_utility_direction': result['rank_utility_direction'],
    })
    print(f'Saved oracle analysis: {out_path}')


if __name__ == '__main__':
    main()

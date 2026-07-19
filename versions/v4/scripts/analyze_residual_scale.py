import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from common import load_cache, save_json
from alpha_targets import validate_alpha_target_source
from reliability_features import (
    RESIDUAL_SCALE_FEATURE_NAMES,
    RESIDUAL_SCALE_SCHEMA_VERSION,
    compute_residual_scale_features,
)
from utility_label import id_confidence_residual


def extract_scale_features(cache, device='cpu', chunk_size=256):
    chunks = []
    compute_device = torch.device(device)
    for start in range(0, cache['labels'].numel(), chunk_size):
        end = min(start + chunk_size, cache['labels'].numel())
        text = cache['logits_text'][start:end].to(compute_device)
        ids = cache['logits_id'][start:end].to(compute_device)
        residual = id_confidence_residual(ids.float(), cache['fusion_temperature'])
        chunks.append(compute_residual_scale_features(text, ids, residual).cpu())
    return torch.cat(chunks, dim=0)


def save_feature_sidecar(path, cache, source_path, features):
    payload = {
        'feature_schema': RESIDUAL_SCALE_SCHEMA_VERSION,
        'feature_names': RESIDUAL_SCALE_FEATURE_NAMES,
        'dataset': cache['dataset'],
        'split': cache['split'],
        'source_cache_path': str(Path(source_path).resolve()),
        'checkpoint_path': cache['checkpoint_path'],
        'fusion_temperature': cache['fusion_temperature'],
        'user_ids': cache['user_ids'],
        'features': features,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def target_variables(targets):
    rows = torch.arange(targets['best_alpha_rank_index'].numel())
    hard = targets['best_alpha_rank_index'].long()
    grid = targets['alpha_grid']
    alpha0_index = int(torch.argmin((grid - float(targets['alpha0'])).abs()).item())
    text_rank = targets['target_rank'][:, 0].float()
    fixed_rank = targets['target_rank'][:, alpha0_index].float()
    return {
        'oracle_best_alpha': targets['best_alpha_rank'].float(),
        'oracle_alpha_bin': targets['best_alpha_class'].float(),
        'oracle_gain_over_text': targets['dcg_at_10'][rows, hard] - targets['dcg_at_10'][:, 0],
        'fixed_gain_over_text': targets['dcg_at_10'][:, alpha0_index] - targets['dcg_at_10'][:, 0],
        'fixed_harm': (fixed_rank > text_rank).float(),
    }


def correlations(features, variables):
    rows = []
    for feature_index, feature_name in enumerate(RESIDUAL_SCALE_FEATURE_NAMES):
        values = features[:, feature_index].numpy()
        for variable_name, variable in variables.items():
            statistic = float(spearmanr(values, variable.numpy()).statistic)
            if not np.isfinite(statistic):
                statistic = 0.0
            rows.append({'feature': feature_name, 'variable': variable_name, 'spearman': statistic})
    return sorted(rows, key=lambda row: abs(row['spearman']), reverse=True)


def distribution_summary(features):
    result = {}
    for name in ('residual_mean', 'residual_std', 'residual_rms', 'residual_max', 'residual_to_text_std_ratio'):
        values = features[:, RESIDUAL_SCALE_FEATURE_NAMES.index(name)].float()
        quantiles = torch.quantile(values, torch.tensor([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]))
        result[name] = {
            'mean': values.mean().item(),
            'std': values.std(unbiased=False).item(),
            'quantiles': dict(zip(('min', 'p01', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99', 'max'), quantiles.tolist())),
            'p99_to_p01_ratio': (quantiles[7] / quantiles[1].clamp_min(1e-8)).item(),
        }
    return result


def bucket_report(features, targets, feature_name, boundaries):
    values = features[:, RESIDUAL_SCALE_FEATURE_NAMES.index(feature_name)]
    buckets = torch.bucketize(values, boundaries)
    rows = torch.arange(values.numel())
    hard = targets['best_alpha_rank_index'].long()
    alpha = targets['best_alpha_rank'].float()
    alpha0_index = int(torch.argmin((targets['alpha_grid'] - float(targets['alpha0'])).abs()).item())
    text_rank = targets['target_rank'][:, 0]
    fixed_rank = targets['target_rank'][:, alpha0_index]
    output = []
    for bucket in range(boundaries.numel() + 1):
        mask = buckets == bucket
        selected_rows = rows[mask]
        output.append({
            'bucket': bucket,
            'count': int(mask.sum().item()),
            'text_ndcg10': targets['dcg_at_10'][mask, 0].mean().item(),
            'fixed_ndcg10': targets['dcg_at_10'][mask, alpha0_index].mean().item(),
            'oracle_ndcg10': targets['dcg_at_10'][selected_rows, hard[mask]].mean().item(),
            'fixed_harm_rate': (fixed_rank[mask] > text_rank[mask]).float().mean().item(),
            'oracle_alpha_mean': alpha[mask].mean().item(),
            'oracle_alpha_median': alpha[mask].median().item(),
            'oracle_alpha_zero_rate': (alpha[mask] == 0).float().mean().item(),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description='Analyze raw ID residual scale without labels in features.')
    parser.add_argument('--train_cache', required=True)
    parser.add_argument('--valid_cache', required=True)
    parser.add_argument('--train_targets', required=True)
    parser.add_argument('--valid_targets', required=True)
    parser.add_argument('--train_features_output', required=True)
    parser.add_argument('--valid_features_output', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--num_buckets', type=int, default=5)
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train_cache, valid_cache = load_cache(args.train_cache), load_cache(args.valid_cache)
    train_targets = torch.load(args.train_targets, map_location='cpu')
    valid_targets = torch.load(args.valid_targets, map_location='cpu')
    validate_alpha_target_source(train_targets, train_cache, args.train_cache)
    validate_alpha_target_source(valid_targets, valid_cache, args.valid_cache)
    train_features = extract_scale_features(train_cache, args.device, args.chunk_size)
    valid_features = extract_scale_features(valid_cache, args.device, args.chunk_size)
    save_feature_sidecar(args.train_features_output, train_cache, args.train_cache, train_features)
    save_feature_sidecar(args.valid_features_output, valid_cache, args.valid_cache, valid_features)

    bucket_outputs = {}
    quantiles = torch.linspace(0, 1, args.num_buckets + 1)[1:-1]
    for feature_name in ('residual_rms', 'residual_to_text_std_ratio'):
        train_values = train_features[:, RESIDUAL_SCALE_FEATURE_NAMES.index(feature_name)]
        boundaries = torch.quantile(train_values, quantiles)
        bucket_outputs[feature_name] = {
            'train_boundaries': boundaries.tolist(),
            'train': bucket_report(train_features, train_targets, feature_name, boundaries),
            'valid': bucket_report(valid_features, valid_targets, feature_name, boundaries),
        }
    result = {
        'feature_schema': RESIDUAL_SCALE_SCHEMA_VERSION,
        'feature_names': RESIDUAL_SCALE_FEATURE_NAMES,
        'train_distribution': distribution_summary(train_features),
        'valid_distribution': distribution_summary(valid_features),
        'train_top_correlations': correlations(train_features, target_variables(train_targets))[:20],
        'valid_top_correlations': correlations(valid_features, target_variables(valid_targets))[:20],
        'buckets': bucket_outputs,
    }
    save_json(args.output_json, result)
    print({'valid_top_correlations': result['valid_top_correlations'][:10]})
    print(f'Saved residual scale analysis: {args.output_json}')


if __name__ == '__main__':
    main()

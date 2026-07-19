import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import load_cache, save_json
from alpha_targets import alpha_class_indices, compute_alpha_outcomes_from_residual, make_alpha_grid, rank_first_hard_target
from paired_bootstrap import paired_bootstrap
from ranking_metrics import target_ranks
from reliability_fusion import FeatureStandardizer
from residual_calibration import calibrate_residual, config_complexity, residual_config_grid


class ScaleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, features):
        return self.net(features.float())


def residual_for_cache(cache, config, device):
    return calibrate_residual(
        cache['logits_id'].to(device),
        transform=config['transform'],
        fusion_temperature=config.get('temperature') or cache['fusion_temperature'],
        topk=config.get('topk'),
    ).cpu()


def per_alpha_metrics(cache, residual, alpha_grid, device):
    text = cache['logits_text'].float().to(device)
    residual = residual.float().to(device)
    labels = cache['labels'].long().to(device)
    text_ranks = target_ranks(text, labels)
    curves = []
    per_sample = []
    for alpha in alpha_grid.to(device):
        scores = text + alpha * residual
        ranks = target_ranks(scores, labels)
        hit5, hit10 = (ranks < 5).float(), (ranks < 10).float()
        dcg = 1.0 / torch.log2(ranks + 2.0)
        curves.append({
            'alpha': float(alpha.item()),
            'NDCG@5': (dcg * hit5).mean().item(),
            'NDCG@10': (dcg * hit10).mean().item(),
            'Recall@5': hit5.mean().item(),
            'Recall@10': hit10.mean().item(),
            'MRR': (1.0 / (ranks + 1.0)).mean().item(),
            'harm_rate': (ranks > text_ranks).float().mean().item(),
            'benefit_rate': (ranks < text_ranks).float().mean().item(),
            'average_target_rank_gain': (text_ranks - ranks).mean().item(),
        })
        per_sample.append((dcg * hit10).cpu())
    return curves, torch.stack(per_sample, dim=-1)


def choose_train_configs(train_cache, alpha_grid, device):
    candidates = []
    per_transform = {}
    for config in residual_config_grid():
        residual = residual_for_cache(train_cache, config, device)
        curve, _ = per_alpha_metrics(train_cache, residual, alpha_grid, device)
        for metrics in curve:
            candidate = {**config, **metrics, 'complexity': config_complexity(config)}
            candidates.append(candidate)
            key = config['transform']
            current = per_transform.get(key)
            ordering = (-candidate['NDCG@10'], candidate['harm_rate'], candidate['alpha'], candidate['complexity'])
            if current is None or ordering < current[0]:
                per_transform[key] = (ordering, candidate)
    best = min(candidates, key=lambda row: (-row['NDCG@10'], row['harm_rate'], row['alpha'], row['complexity']))
    return best, {key: value[1] for key, value in per_transform.items()}


def evaluate_frozen_config(cache, selected, alpha_grid, device):
    residual = residual_for_cache(cache, selected, device)
    curve, sample_dcg = per_alpha_metrics(cache, residual, alpha_grid, device)
    alpha_index = int(torch.argmin((alpha_grid - float(selected['alpha'])).abs()).item())
    metrics = curve[alpha_index]
    scale = residual.float().square().mean(dim=-1).sqrt()
    metrics['residual_rms_mean'] = scale.mean().item()
    metrics['residual_rms_std'] = scale.std(unbiased=False).item()
    return metrics, sample_dcg[:, alpha_index], residual


def oracle_metrics(cache, residual, alpha_grid, device):
    outcomes = compute_alpha_outcomes_from_residual(cache['logits_text'], residual, cache['labels'], alpha_grid, device=device)
    hard = rank_first_hard_target(outcomes['target_rank'], outcomes['cross_entropy'])
    rows = torch.arange(hard.numel())
    return {
        'NDCG@10': outcomes['dcg_at_10'][rows, hard].mean().item(),
        'Recall@10': outcomes['hit_at_10'][rows, hard].mean().item(),
        'alpha_mean': alpha_grid[hard].mean().item(),
        'alpha_zero_rate': (hard == 0).float().mean().item(),
    }, outcomes, hard


def train_mlp(features, labels, valid_features, classes, args, seed):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = ScaleMLP(features.size(1), classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, y = features.to(device), labels.to(device)
    for _ in range(200):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        return torch.softmax(model(valid_features.to(device)), dim=-1).cpu()


def dynamic_metrics(cache, residual, predicted_alpha, oracle_alpha, fixed_ndcg, oracle_ndcg, device):
    text = cache['logits_text'].float().to(device)
    labels = cache['labels'].long().to(device)
    scores = text + predicted_alpha.to(device).unsqueeze(-1) * residual.to(device)
    ranks, text_ranks = target_ranks(scores, labels), target_ranks(text, labels)
    dcg = 1.0 / torch.log2(ranks + 2.0) * (ranks < 10).float()
    denominator = oracle_ndcg - fixed_ndcg
    rounded = torch.round(predicted_alpha * 20) / 20
    alpha_distribution = {
        str(float(value)): int((rounded == value).sum().item())
        for value in torch.unique(rounded, sorted=True)
    }
    return {
        'NDCG@10': dcg.mean().item(),
        'Recall@10': (ranks < 10).float().mean().item(),
        'harm_rate': (ranks > text_ranks).float().mean().item(),
        'benefit_rate': (ranks < text_ranks).float().mean().item(),
        'alpha_mean': predicted_alpha.mean().item(),
        'alpha_std': predicted_alpha.std(unbiased=False).item(),
        'alpha_zero_rate': (predicted_alpha <= 1e-6).float().mean().item(),
        'alpha_distribution': alpha_distribution,
        'alpha_mae': (predicted_alpha - oracle_alpha).abs().mean().item(),
        'oracle_recovery_ratio': (dcg.mean().item() - fixed_ndcg) / denominator if abs(denominator) > 1e-12 else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(description='Train-only residual calibration and scale-aware alpha diagnosis.')
    parser.add_argument('--train_cache', required=True); parser.add_argument('--valid_cache', required=True)
    parser.add_argument('--train_scale_features', required=True); parser.add_argument('--valid_scale_features', required=True)
    parser.add_argument('--output_json', required=True); parser.add_argument('--selected_config_json', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train, valid = load_cache(args.train_cache), load_cache(args.valid_cache)
    train_scale, valid_scale = torch.load(args.train_scale_features, map_location='cpu'), torch.load(args.valid_scale_features, map_location='cpu')
    if set(train['user_ids'].tolist()).intersection(valid['user_ids'].tolist()): raise ValueError('Calibration users overlap.')
    for sidecar, cache, path in (
        (train_scale, train, args.train_cache),
        (valid_scale, valid, args.valid_cache),
    ):
        if sidecar['dataset'] != cache['dataset'] or sidecar['split'] != cache['split']:
            raise ValueError('Residual-scale feature sidecar dataset/split mismatch.')
        if not torch.equal(sidecar['user_ids'], cache['user_ids']):
            raise ValueError('Residual-scale feature sidecar user_ids mismatch.')
        if sidecar['source_cache_path'] != str(Path(path).resolve()):
            raise ValueError('Residual-scale feature sidecar source path mismatch.')
    alpha_grid = make_alpha_grid(1.0, 0.05)

    selected, per_transform = choose_train_configs(train, alpha_grid, args.device)
    save_json(args.selected_config_json, {'selected_on': 'calibration_train', 'selected_config': selected, 'per_transform': per_transform})
    valid_results = {}
    frozen_details = {}
    for transform, config in per_transform.items():
        metrics, _, _ = evaluate_frozen_config(valid, config, alpha_grid, args.device)
        valid_results[transform] = metrics
        frozen_details[transform] = config
    best_metrics, best_sample_dcg, valid_residual = evaluate_frozen_config(valid, selected, alpha_grid, args.device)
    train_residual = residual_for_cache(train, selected, args.device)

    raw_fixed = {'transform': 'R0', 'temperature': float(train['fusion_temperature']), 'topk': None, 'alpha': 0.5}
    raw_fixed_metrics, raw_fixed_dcg, raw_residual = evaluate_frozen_config(valid, raw_fixed, alpha_grid, args.device)
    raw_train_selected = per_transform['R0']
    raw_selected_metrics, _, _ = evaluate_frozen_config(valid, raw_train_selected, alpha_grid, args.device)
    text_metrics, text_sample_dcg, _ = evaluate_frozen_config(valid, {**raw_fixed, 'alpha': 0.0}, alpha_grid, args.device)
    raw_oracle, _, _ = oracle_metrics(valid, raw_residual, alpha_grid, args.device)
    calibrated_oracle, train_dummy, valid_hard = oracle_metrics(valid, valid_residual, alpha_grid, args.device)

    train_outcomes = compute_alpha_outcomes_from_residual(train['logits_text'], train_residual, train['labels'], alpha_grid, device=args.device)
    train_hard = rank_first_hard_target(train_outcomes['target_rank'], train_outcomes['cross_entropy'])
    valid_outcomes = train_dummy
    train_features = torch.cat([train['features'].float(), train_scale['features'].float()], dim=-1)
    valid_features = torch.cat([valid['features'].float(), valid_scale['features'].float()], dim=-1)
    scaler = FeatureStandardizer().fit(train_features); x_train, x_valid = scaler.transform(train_features), scaler.transform(valid_features)
    y4_train, y4_valid = alpha_class_indices(alpha_grid[train_hard], train['alpha0']), alpha_class_indices(alpha_grid[valid_hard], valid['alpha0'])

    log4 = LogisticRegression(max_iter=1000, random_state=args.seed).fit(x_train.numpy(), y4_train.numpy())
    log21 = LogisticRegression(max_iter=1000, random_state=args.seed).fit(x_train.numpy(), train_hard.numpy())
    mlp4, mlp21 = train_mlp(x_train, y4_train, x_valid, 4, args, args.seed), train_mlp(x_train, train_hard, x_valid, 21, args, args.seed + 1)
    reps = torch.tensor([0.0, 0.25, 0.5, 0.75])
    predictions = {
        'logistic_four': reps[torch.from_numpy(log4.predict(x_valid.numpy()))],
        'logistic_21': alpha_grid[torch.from_numpy(log21.predict(x_valid.numpy()))],
        'mlp_four': reps[mlp4.argmax(dim=-1)],
        'mlp_21': alpha_grid[mlp21.argmax(dim=-1)],
    }
    dynamic = {name: dynamic_metrics(valid, valid_residual, alpha, alpha_grid[valid_hard], best_metrics['NDCG@10'], calibrated_oracle['NDCG@10'], args.device) for name, alpha in predictions.items()}
    classification = {
        'logistic_four': {'accuracy': accuracy_score(y4_valid, log4.predict(x_valid.numpy())), 'macro_f1': f1_score(y4_valid, log4.predict(x_valid.numpy()), average='macro', zero_division=0)},
        'logistic_21': {'accuracy': accuracy_score(valid_hard, log21.predict(x_valid.numpy())), 'macro_f1': f1_score(valid_hard, log21.predict(x_valid.numpy()), average='macro', zero_division=0)},
    }
    bootstrap = {
        'vs_text': paired_bootstrap(best_sample_dcg - text_sample_dcg, seed=args.seed),
        'vs_v1_fixed': paired_bootstrap(best_sample_dcg - raw_fixed_dcg, seed=args.seed),
    }
    go_a = best_metrics['NDCG@10'] > text_metrics['NDCG@10'] and best_metrics['NDCG@10'] > raw_fixed_metrics['NDCG@10'] and best_metrics['harm_rate'] < raw_fixed_metrics['harm_rate'] and bootstrap['vs_text']['ci95_high'] > 0
    best_dynamic_name = max(dynamic, key=lambda name: dynamic[name]['NDCG@10'])
    go_b = dynamic[best_dynamic_name]['NDCG@10'] > best_metrics['NDCG@10']
    result = {
        'selection_protocol': 'configuration and alpha selected only on calibration_train',
        'selected_config': selected, 'per_transform_train_selection': frozen_details,
        'text_only': text_metrics, 'v1_raw_fixed': raw_fixed_metrics,
        'raw_train_selected': {'config': raw_train_selected, 'valid_metrics': raw_selected_metrics},
        'calibrated_per_transform_valid': valid_results,
        'best_calibrated_valid': best_metrics,
        'raw_oracle_valid': raw_oracle, 'calibrated_oracle_valid': calibrated_oracle,
        'bootstrap': bootstrap, 'scale_aware_classification': classification,
        'scale_aware_dynamic': dynamic, 'best_dynamic_model': best_dynamic_name,
        'go_a': go_a, 'go_b': go_b, 'decision': 'Go-A' if go_a else ('Go-B' if go_b else 'No-Go'),
    }
    save_json(args.output_json, result)
    print({'selected_config': selected, 'text_ndcg10': text_metrics['NDCG@10'], 'best_calibrated': best_metrics, 'best_dynamic': {best_dynamic_name: dynamic[best_dynamic_name]}, 'decision': result['decision']})
    print(f'Saved residual calibration analysis: {args.output_json}')


if __name__ == '__main__': main()

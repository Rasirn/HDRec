import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from common import load_cache, save_json
from alpha_targets import validate_alpha_target_source
from ranking_metrics import ranking_metrics, target_ranks
from reliability_fusion import FeatureStandardizer
from utility_label import id_confidence_residual


class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.net(features.float())


def classification_metrics(labels, predictions, num_classes, probabilities=None):
    result = {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, labels=list(range(num_classes)), average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(labels, predictions, labels=list(range(num_classes))).tolist(),
        'class_distribution': dict(sorted(Counter(int(value) for value in labels).items())),
        'predicted_class_distribution': dict(sorted(Counter(int(value) for value in predictions).items())),
    }
    if probabilities is not None:
        topk = min(3, probabilities.shape[1])
        top_indices = np.argpartition(probabilities, -topk, axis=1)[:, -topk:]
        result['top3_accuracy'] = float(np.mean([label in row for label, row in zip(labels, top_indices)]))
    return result


def representative_alpha_by_class(alpha, classes, alpha_grid, num_classes=4):
    representatives = []
    for class_index in range(num_classes):
        values = alpha[classes == class_index]
        if values.numel():
            indices = torch.argmin((values.unsqueeze(-1) - alpha_grid.unsqueeze(0)).abs(), dim=-1)
            mode_index = int(torch.bincount(indices, minlength=alpha_grid.numel()).argmax().item())
            representatives.append(float(alpha_grid[mode_index].item()))
        else:
            representatives.append(float((0.0, 0.25, 0.5, 0.75)[class_index]))
    return torch.tensor(representatives)


def alpha_distribution(alpha, alpha_grid):
    nearest = torch.argmin((alpha.unsqueeze(-1) - alpha_grid.unsqueeze(0)).abs(), dim=-1)
    counts = torch.bincount(nearest, minlength=alpha_grid.numel())
    return {str(float(value)): int(count) for value, count in zip(alpha_grid.tolist(), counts.tolist())}


def evaluate_predicted_alpha(cache, predicted_alpha, oracle_alpha, alpha_grid, best_global_ndcg, oracle_ndcg,
                             device):
    compute_device = torch.device(device)
    text = cache['logits_text'].float().to(compute_device)
    ids = cache['logits_id'].float().to(compute_device)
    labels = cache['labels'].long().to(compute_device)
    alpha = predicted_alpha.float().to(compute_device)
    residual = id_confidence_residual(ids, temperature=cache['fusion_temperature'])
    final = text + alpha.unsqueeze(-1) * residual
    metrics = ranking_metrics(final, labels)
    text_ranks = target_ranks(text, labels)
    dynamic_ranks = target_ranks(final, labels)
    denominator = oracle_ndcg - best_global_ndcg
    recovery = (metrics['NDCG@10'] - best_global_ndcg) / denominator if abs(denominator) > 1e-12 else float('nan')
    return {
        **metrics,
        'predicted_alpha_mean': alpha.mean().item(),
        'predicted_alpha_std': alpha.std(unbiased=False).item(),
        'predicted_alpha_zero_rate': (alpha <= 1e-6).float().mean().item(),
        'predicted_alpha_above_alpha0_rate': (alpha > float(cache['alpha0']) + 1e-6).float().mean().item(),
        'mean_absolute_alpha_error': (alpha.cpu() - oracle_alpha.float()).abs().mean().item(),
        'predicted_alpha_distribution': alpha_distribution(alpha.cpu(), alpha_grid),
        'harm_rate_vs_text': (dynamic_ranks > text_ranks).float().mean().item(),
        'benefit_rate_vs_text': (dynamic_ranks < text_ranks).float().mean().item(),
        'oracle_recovery_ratio': recovery,
    }


def train_mlp(x_train, y_train, x_valid, num_classes, args, seed):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = BaselineMLP(x_train.size(1), num_classes, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = x_train.to(device)
    y_train = y_train.long().to(device)
    for _ in range(args.epochs):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x_valid.to(device)), dim=-1).cpu()
    return probabilities


def oracle_summary(alpha, alpha0):
    quantiles = torch.quantile(alpha.float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    return {
        'alpha_zero_rate': (alpha == 0).float().mean().item(),
        'alpha_between_zero_and_alpha0_rate': ((alpha > 0) & (alpha < alpha0)).float().mean().item(),
        'alpha_at_alpha0_rate': torch.isclose(alpha, torch.full_like(alpha, alpha0), atol=1e-6).float().mean().item(),
        'alpha_above_alpha0_rate': (alpha > alpha0).float().mean().item(),
        'alpha_mean': alpha.float().mean().item(),
        'alpha_median': alpha.float().median().item(),
        'alpha_p10': quantiles[0].item(),
        'alpha_p25': quantiles[1].item(),
        'alpha_p50': quantiles[2].item(),
        'alpha_p75': quantiles[3].item(),
        'alpha_p90': quantiles[4].item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze oracle alpha targets and simple predictability baselines.')
    parser.add_argument('--train_cache', required=True)
    parser.add_argument('--valid_cache', required=True)
    parser.add_argument('--train_targets', required=True)
    parser.add_argument('--valid_targets', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train_cache = load_cache(args.train_cache)
    valid_cache = load_cache(args.valid_cache)
    train_targets = torch.load(args.train_targets, map_location='cpu')
    valid_targets = torch.load(args.valid_targets, map_location='cpu')
    validate_alpha_target_source(train_targets, train_cache, args.train_cache)
    validate_alpha_target_source(valid_targets, valid_cache, args.valid_cache)
    if set(train_cache['user_ids'].tolist()).intersection(valid_cache['user_ids'].tolist()):
        raise ValueError('Calibration train/valid users overlap.')
    if not torch.equal(train_targets['alpha_grid'], valid_targets['alpha_grid']):
        raise ValueError('Train/valid alpha grids differ.')

    alpha_grid = train_targets['alpha_grid'].float()
    y21_train = train_targets['best_alpha_rank_index'].long()
    y21_valid = valid_targets['best_alpha_rank_index'].long()
    y4_train = train_targets['best_alpha_class'].long()
    y4_valid = valid_targets['best_alpha_class'].long()
    oracle_alpha_train = train_targets['best_alpha_rank'].float()
    oracle_alpha_valid = valid_targets['best_alpha_rank'].float()

    scaler = FeatureStandardizer().fit(train_cache['features'])
    x_train = scaler.transform(train_cache['features'])
    x_valid = scaler.transform(valid_cache['features'])
    x_train_np, x_valid_np = x_train.numpy(), x_valid.numpy()

    mean_dcg10 = valid_targets['dcg_at_10'].mean(dim=0)
    best_global_index = int(mean_dcg10.argmax().item())
    best_global_alpha = float(alpha_grid[best_global_index].item())
    best_global_ndcg = float(mean_dcg10[best_global_index].item())
    oracle_ndcg = float(valid_targets['dcg_at_10'][torch.arange(y21_valid.numel()), y21_valid].mean().item())
    representatives = representative_alpha_by_class(oracle_alpha_train, y4_train, alpha_grid)

    majority4 = int(torch.bincount(y4_train, minlength=4).argmax().item())
    majority21 = int(torch.bincount(y21_train, minlength=alpha_grid.numel()).argmax().item())

    logistic4 = LogisticRegression(max_iter=1000, random_state=args.seed).fit(x_train_np, y4_train.numpy())
    logistic21 = LogisticRegression(max_iter=1000, random_state=args.seed).fit(x_train_np, y21_train.numpy())
    logistic4_prob = logistic4.predict_proba(x_valid_np)
    logistic21_observed_prob = logistic21.predict_proba(x_valid_np)
    logistic21_prob = np.zeros((x_valid_np.shape[0], alpha_grid.numel()), dtype=np.float64)
    logistic21_prob[:, logistic21.classes_.astype(int)] = logistic21_observed_prob
    logistic4_pred = logistic4.predict(x_valid_np)
    logistic21_pred = logistic21.predict(x_valid_np)

    mlp4_prob = train_mlp(x_train, y4_train, x_valid, 4, args, args.seed)
    mlp21_prob = train_mlp(x_train, y21_train, x_valid, alpha_grid.numel(), args, args.seed + 1)
    mlp4_pred = mlp4_prob.argmax(dim=-1).numpy()
    mlp21_pred = mlp21_prob.argmax(dim=-1).numpy()

    prediction_sets = {
        'text_only': torch.zeros_like(oracle_alpha_valid),
        'v1_fixed_alpha0': torch.full_like(oracle_alpha_valid, float(valid_cache['alpha0'])),
        'best_global_alpha': torch.full_like(oracle_alpha_valid, best_global_alpha),
        'majority_four_class': torch.full_like(oracle_alpha_valid, representatives[majority4]),
        'majority_21_class': torch.full_like(oracle_alpha_valid, alpha_grid[majority21]),
        'logistic_four_class': representatives[torch.from_numpy(logistic4_pred)],
        'logistic_21_class': alpha_grid[torch.from_numpy(logistic21_pred)],
        'mlp_four_class': representatives[torch.from_numpy(mlp4_pred)],
        'mlp_21_class': alpha_grid[torch.from_numpy(mlp21_pred)],
    }
    recommendation_results = {
        name: evaluate_predicted_alpha(
            valid_cache,
            predicted,
            oracle_alpha_valid,
            alpha_grid,
            best_global_ndcg,
            oracle_ndcg,
            args.device,
        )
        for name, predicted in prediction_sets.items()
    }

    feature_names = train_cache['feature_names']
    feature_means = {}
    for class_index in range(4):
        mask = y4_train == class_index
        feature_means[str(class_index)] = {
            name: train_cache['features'][mask, index].mean().item()
            for index, name in enumerate(feature_names)
        }
    correlations = {}
    for index, name in enumerate(feature_names):
        statistic = float(spearmanr(
            valid_cache['features'][:, index].numpy(), oracle_alpha_valid.numpy()
        ).statistic)
        correlations[name] = statistic if np.isfinite(statistic) else 0.0
    logistic4_importance = {
        name: float(np.abs(logistic4.coef_[:, index]).mean())
        for index, name in enumerate(feature_names)
    }
    sorted_importance = sorted(logistic4_importance, key=logistic4_importance.get, reverse=True)
    popularity_only_warning = (
        sorted_importance[0] == 'history_pop_mean'
        and abs(correlations['history_pop_mean']) >= 0.3
    )

    result = {
        'dataset': train_cache['dataset'],
        'alpha_grid': alpha_grid.tolist(),
        'target_metadata': {
            key: train_targets[key]
            for key in ('alpha_step', 'alpha_max', 'fusion_temperature', 'residual_definition', 'tau_ce', 'tau_metric', 'beta_rr', 'beta_ce')
        },
        'oracle_alpha_distribution': oracle_summary(oracle_alpha_valid, valid_cache['alpha0']),
        'best_global_alpha': best_global_alpha,
        'best_global_ndcg10': best_global_ndcg,
        'oracle_ndcg10': oracle_ndcg,
        'four_class_representative_alpha': representatives.tolist(),
        'classification': {
            'majority_four_class': classification_metrics(y4_valid.numpy(), np.full(y4_valid.numel(), majority4), 4),
            'logistic_four_class': classification_metrics(y4_valid.numpy(), logistic4_pred, 4, logistic4_prob),
            'mlp_four_class': classification_metrics(y4_valid.numpy(), mlp4_pred, 4, mlp4_prob.numpy()),
            'majority_21_class': classification_metrics(y21_valid.numpy(), np.full(y21_valid.numel(), majority21), alpha_grid.numel()),
            'logistic_21_class': classification_metrics(y21_valid.numpy(), logistic21_pred, alpha_grid.numel(), logistic21_prob),
            'mlp_21_class': classification_metrics(y21_valid.numpy(), mlp21_pred, alpha_grid.numel(), mlp21_prob.numpy()),
        },
        'recommendation_results': recommendation_results,
        'feature_means_by_four_class': feature_means,
        'feature_spearman_with_oracle_alpha': correlations,
        'logistic_four_class_feature_importance': logistic4_importance,
        'feature_importance_order': sorted_importance,
        'popularity_only_warning': popularity_only_warning,
        'go_models': [
            name for name, metrics in recommendation_results.items()
            if name not in (
                'text_only', 'v1_fixed_alpha0', 'best_global_alpha',
                'majority_four_class', 'majority_21_class'
            )
            and metrics['NDCG@10'] > best_global_ndcg
        ],
    }
    result['alpha_predictability_go'] = bool(result['go_models'])
    output_path = Path(args.output_json)
    save_json(output_path, result)
    print({
        'best_global_alpha': best_global_alpha,
        'best_global_ndcg10': best_global_ndcg,
        'oracle_ndcg10': oracle_ndcg,
        'recommendation_ndcg10': {name: value['NDCG@10'] for name, value in recommendation_results.items()},
        'go_models': result['go_models'],
        'alpha_predictability_go': result['alpha_predictability_go'],
        'popularity_only_warning': popularity_only_warning,
    })
    print(f'Saved alpha target analysis: {output_path}')


if __name__ == '__main__':
    main()

from pathlib import Path

import torch
import torch.nn.functional as F

from ranking_metrics import target_ranks
from utility_label import id_confidence_residual


ALPHA_TARGET_SCHEMA = 'v4_alpha_targets_v1'
RESIDUAL_DEFINITION = (
    '(logits_id - min(logits_id) + 1e-8) * '
    'sigmoid((logits_id - mean(logits_id)) / fusion_temperature)'
)


def make_alpha_grid(alpha_max=1.0, alpha_step=0.05):
    if alpha_max <= 0 or alpha_step <= 0:
        raise ValueError('alpha_max and alpha_step must be positive.')
    steps = int(round(alpha_max / alpha_step))
    grid = torch.arange(steps + 1, dtype=torch.float32) * float(alpha_step)
    if not torch.isclose(grid[-1], torch.tensor(float(alpha_max)), atol=1e-7, rtol=0.0):
        grid = torch.cat([grid, torch.tensor([float(alpha_max)])])
    grid[-1] = float(alpha_max)
    return grid


def alpha_class_indices(alpha, alpha0, atol=1e-6):
    alpha = alpha.float()
    classes = torch.empty_like(alpha, dtype=torch.long)
    at_zero = torch.isclose(alpha, torch.zeros_like(alpha), atol=atol, rtol=0.0)
    at_alpha0 = torch.isclose(alpha, torch.full_like(alpha, float(alpha0)), atol=atol, rtol=0.0)
    classes[at_zero] = 0
    classes[(alpha > atol) & (alpha < float(alpha0) - atol)] = 1
    classes[at_alpha0] = 2
    classes[alpha > float(alpha0) + atol] = 3
    return classes


def rank_first_hard_target(rank_per_alpha, ce_per_alpha):
    min_rank = rank_per_alpha.min(dim=-1, keepdim=True).values
    eligible = rank_per_alpha == min_rank
    masked_ce = ce_per_alpha.masked_fill(~eligible, float('inf'))
    return masked_ce.argmin(dim=-1)


def soft_targets(ce_per_alpha, dcg10_per_alpha, reciprocal_rank_per_alpha, tau_ce=0.5,
                 tau_metric=0.1, beta_rr=0.1, beta_ce=0.05, eps=1e-6):
    if tau_ce <= 0 or tau_metric <= 0:
        raise ValueError('Soft-target temperatures must be positive.')
    ce_distribution = torch.softmax(-ce_per_alpha / float(tau_ce), dim=-1)
    ce_mean = ce_per_alpha.mean(dim=-1, keepdim=True)
    ce_std = ce_per_alpha.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    normalized_ce = (ce_per_alpha - ce_mean) / ce_std
    reward = (
        dcg10_per_alpha
        + float(beta_rr) * reciprocal_rank_per_alpha
        - float(beta_ce) * normalized_ce
    )
    metric_distribution = torch.softmax(reward / float(tau_metric), dim=-1)
    return ce_distribution, metric_distribution, reward


def compute_alpha_outcomes(logits_text, logits_id, labels, alpha_grid, fusion_temperature,
                           device='cpu', chunk_size=256):
    n = labels.numel()
    num_alpha = alpha_grid.numel()
    outcomes = {
        name: torch.empty(n, num_alpha, dtype=dtype)
        for name, dtype in (
            ('cross_entropy', torch.float32),
            ('target_rank', torch.int32),
            ('reciprocal_rank', torch.float32),
            ('dcg_at_5', torch.float32),
            ('dcg_at_10', torch.float32),
            ('hit_at_5', torch.float32),
            ('hit_at_10', torch.float32),
        )
    }
    compute_device = torch.device(device)
    grid_device = alpha_grid.to(compute_device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        text = logits_text[start:end].float().to(compute_device)
        ids = logits_id[start:end].float().to(compute_device)
        target = labels[start:end].long().to(compute_device)
        residual = id_confidence_residual(ids, temperature=fusion_temperature)
        for alpha_index, alpha in enumerate(grid_device):
            scores = text + alpha * residual
            ce = F.cross_entropy(scores, target, reduction='none')
            rank_zero_based = target_ranks(scores, target)
            reciprocal_rank = 1.0 / (rank_zero_based + 1.0)
            discount = 1.0 / torch.log2(rank_zero_based + 2.0)
            values = {
                'cross_entropy': ce,
                'target_rank': rank_zero_based.to(torch.int32),
                'reciprocal_rank': reciprocal_rank,
                'dcg_at_5': discount * (rank_zero_based < 5).float(),
                'dcg_at_10': discount * (rank_zero_based < 10).float(),
                'hit_at_5': (rank_zero_based < 5).float(),
                'hit_at_10': (rank_zero_based < 10).float(),
            }
            for name, value in values.items():
                outcomes[name][start:end, alpha_index] = value.detach().cpu()
    return outcomes


def build_alpha_target_payload(cache, source_cache_path, alpha_grid, outcomes, tau_ce=0.5,
                               tau_metric=0.1, beta_rr=0.1, beta_ce=0.05):
    hard_index = rank_first_hard_target(outcomes['target_rank'], outcomes['cross_entropy'])
    ce_distribution, metric_distribution, reward = soft_targets(
        outcomes['cross_entropy'],
        outcomes['dcg_at_10'],
        outcomes['reciprocal_rank'],
        tau_ce=tau_ce,
        tau_metric=tau_metric,
        beta_rr=beta_rr,
        beta_ce=beta_ce,
    )
    best_alpha = alpha_grid[hard_index]
    return {
        'target_schema': ALPHA_TARGET_SCHEMA,
        'dataset': cache['dataset'],
        'split': cache['split'],
        'source_cache_path': str(Path(source_cache_path).resolve()),
        'source_cache_version': cache['version'],
        'feature_schema': cache['feature_schema'],
        'feature_names': cache['feature_names'],
        'checkpoint_path': cache['checkpoint_path'],
        'fusion_temperature': float(cache['fusion_temperature']),
        'residual_definition': RESIDUAL_DEFINITION,
        'alpha0': float(cache['alpha0']),
        'alpha_grid': alpha_grid,
        'alpha_step': float((alpha_grid[1] - alpha_grid[0]).item()),
        'alpha_max': float(alpha_grid[-1].item()),
        'tau_ce': float(tau_ce),
        'tau_metric': float(tau_metric),
        'beta_rr': float(beta_rr),
        'beta_ce': float(beta_ce),
        'user_ids': cache['user_ids'].clone(),
        **outcomes,
        'best_alpha_rank': best_alpha,
        'best_alpha_rank_index': hard_index,
        'best_alpha_class': alpha_class_indices(best_alpha, cache['alpha0']),
        'alpha_target_ce_distribution': ce_distribution,
        'alpha_target_metric_distribution': metric_distribution,
        'metric_reward': reward,
    }


def validate_alpha_target_source(targets, cache, source_path=None):
    fields = ('dataset', 'feature_schema', 'feature_names', 'checkpoint_path', 'fusion_temperature')
    required_target_fields = fields + (
        'target_schema', 'source_cache_path', 'user_ids', 'alpha_grid',
        'best_alpha_rank', 'best_alpha_rank_index', 'alpha_target_ce_distribution',
        'alpha_target_metric_distribution',
    )
    missing = [field for field in required_target_fields if field not in targets]
    if missing:
        raise ValueError(f'Incompatible legacy alpha target cache; missing fields: {missing}')
    for field in fields:
        if field not in cache:
            raise ValueError(f'Incompatible source cache; missing field: {field}')
        if targets[field] != cache[field]:
            raise ValueError(f'Alpha target/cache mismatch for {field}.')
    if not torch.equal(targets['user_ids'], cache['user_ids']):
        raise ValueError('Alpha target/cache user_ids differ.')
    if source_path is not None and str(Path(source_path).resolve()) != targets['source_cache_path']:
        raise ValueError('Alpha target references a different source cache path.')

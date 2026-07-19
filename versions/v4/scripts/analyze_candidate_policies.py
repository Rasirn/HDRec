import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_cache, save_json
from candidate_features import CANDIDATE_FEATURE_NAMES, rank_positions
from candidate_policies import (
    POLICY_ALPHA_GRID,
    candidate_policy_configs,
    policy_complexity,
    policy_gate,
    policy_key,
)
from paired_bootstrap import paired_bootstrap
from utility_label import id_confidence_residual


def load_item_popularity(data_root, dataset, item_num):
    with open(Path(data_root) / dataset / 'train.json') as handle:
        train = json.load(handle)
    popularity = torch.zeros(item_num, dtype=torch.float32)
    for sequence in train.values():
        indices = torch.as_tensor(sequence, dtype=torch.long)
        indices = indices[(indices >= 0) & (indices < item_num)]
        popularity.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
    return popularity


def target_ranks_for_scores(scores, labels):
    # Supports [alpha, batch, item] without materializing candidate features.
    if scores.ndim == 2:
        target = scores.gather(1, labels[:, None])
        return (scores > target).sum(dim=-1).float()
    target = scores.gather(2, labels[None, :, None].expand(scores.size(0), -1, 1))
    return (scores > target).sum(dim=-1).float()


def per_sample_dcg10(ranks):
    return (ranks < 10).float() / torch.log2(ranks + 2.0)


def metrics_from_ranks(ranks, text_ranks):
    dcg = 1.0 / torch.log2(ranks + 2.0)
    return {
        'NDCG@5': (dcg * (ranks < 5)).mean().item(),
        'NDCG@10': (dcg * (ranks < 10)).mean().item(),
        'Recall@5': (ranks < 5).float().mean().item(),
        'Recall@10': (ranks < 10).float().mean().item(),
        'MRR': (1.0 / (ranks + 1.0)).mean().item(),
        'harm_rate': (ranks > text_ranks).float().mean().item(),
        'benefit_rate': (ranks < text_ranks).float().mean().item(),
        'average_target_rank_gain': (text_ranks - ranks).mean().item(),
    }


def evaluate_policy_grid(cache, configs, alpha_grid, batch_size, device):
    n = cache['labels'].numel()
    all_ranks = torch.empty(n, len(configs), len(alpha_grid), dtype=torch.float32)
    text_ranks = torch.empty(n, dtype=torch.float32)
    fixed_ranks = torch.empty(n, dtype=torch.float32)
    temperature = float(cache['fusion_temperature'])
    alpha = torch.tensor(alpha_grid, device=device, dtype=torch.float32)[:, None, None]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        text = cache['logits_text'][start:end].float().to(device)
        ids = cache['logits_id'][start:end].float().to(device)
        labels = cache['labels'][start:end].long().to(device)
        residual = id_confidence_residual(ids, temperature)
        text_rank, id_rank = rank_positions(text), rank_positions(ids)
        text_prob, id_prob = F.softmax(text, -1), F.softmax(ids, -1)
        text_ranks[start:end] = target_ranks_for_scores(text, labels).cpu()
        fixed_ranks[start:end] = target_ranks_for_scores(text + 0.5 * residual, labels).cpu()
        for config_index, config in enumerate(configs):
            gate = policy_gate(config, text_rank, id_rank, text_prob, id_prob)
            scores = text.unsqueeze(0) + alpha * gate.unsqueeze(0) * residual.unsqueeze(0)
            all_ranks[start:end, config_index] = target_ranks_for_scores(scores, labels).transpose(0, 1).cpu()
    return all_ranks, text_ranks, fixed_ranks


def select_train_configurations(configs, alpha_grid, ranks, text_ranks):
    selected_by_config = {}
    selected_by_family = {}
    all_candidates = []
    for config_index, config in enumerate(configs):
        key = policy_key(config)
        rows = []
        for alpha_index, alpha in enumerate(alpha_grid):
            row = {
                **config,
                'key': key,
                'max_alpha': float(alpha),
                **metrics_from_ranks(ranks[:, config_index, alpha_index], text_ranks),
                'complexity': policy_complexity(config),
            }
            rows.append(row)
            all_candidates.append(row)
        best = min(rows, key=lambda x: (-x['NDCG@10'], x['harm_rate'], x['max_alpha'], x['complexity']))
        selected_by_config[key] = best
        family = config['policy']
        current = selected_by_family.get(family)
        if current is None or (-best['NDCG@10'], best['harm_rate'], best['max_alpha'], best['complexity']) < (
            -current['NDCG@10'], current['harm_rate'], current['max_alpha'], current['complexity']
        ):
            selected_by_family[family] = best
    selected = min(all_candidates, key=lambda x: (-x['NDCG@10'], x['harm_rate'], x['max_alpha'], x['complexity']))
    return selected, selected_by_config, selected_by_family


def frozen_config_list(selected_by_config):
    return [selected_by_config[key] for key in sorted(selected_by_config)]


def evaluate_frozen_configs(cache, configs, batch_size, device):
    alpha_grid = [float(config['max_alpha']) for config in configs]
    # Each mask has one frozen alpha; evaluate together then take the diagonal.
    ranks_cube, text_ranks, fixed_ranks = evaluate_policy_grid(cache, configs, alpha_grid, batch_size, device)
    diagonal = torch.stack([ranks_cube[:, index, index] for index in range(len(configs))], dim=-1)
    return diagonal, text_ranks, fixed_ranks


def gate_statistics(cache, config, batch_size, device):
    values, active_counts = [], []
    n = cache['labels'].numel()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        text = cache['logits_text'][start:end].float().to(device)
        ids = cache['logits_id'][start:end].float().to(device)
        gate = policy_gate(config, rank_positions(text), rank_positions(ids), F.softmax(text, -1), F.softmax(ids, -1))
        values.append(gate.cpu().flatten())
        active_counts.append((gate > 1e-6).float().sum(dim=-1).cpu())
    gate, counts = torch.cat(values), torch.cat(active_counts)
    return {
        'gate_mean': gate.mean().item(), 'gate_std': gate.std(unbiased=False).item(),
        'gate_near_zero_rate': (gate <= 1e-6).float().mean().item(),
        'gate_gt_half_rate': (gate > 0.5).float().mean().item(),
        'active_candidates_mean': counts.mean().item(),
        'active_candidate_ratio': (counts / cache['logits_text'].size(1)).mean().item(),
    }


def oracle_from_columns(columns, text_ranks):
    dcg = per_sample_dcg10(columns)
    best_dcg, choice = dcg.max(dim=-1)
    chosen_ranks = columns[torch.arange(columns.size(0)), choice]
    result = metrics_from_ranks(chosen_ranks, text_ranks)
    result['NDCG@10'] = best_dcg.mean().item()
    result['num_policies'] = columns.size(1)
    return result


def main():
    parser = argparse.ArgumentParser(description='Train-only candidate policy selection and independent validation.')
    parser.add_argument('--train_cache', required=True)
    parser.add_argument('--valid_cache', required=True)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--selected_json', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--candidate_chunk', type=int, default=1024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train, valid = load_cache(args.train_cache), load_cache(args.valid_cache)
    if set(train['user_ids'].tolist()).intersection(valid['user_ids'].tolist()):
        raise ValueError('Calibration train/valid users overlap.')
    if train['dataset'] != 'Industrial_and_Scientific' or valid['dataset'] != train['dataset']:
        raise ValueError('This diagnostic is restricted to Industrial_and_Scientific.')

    item_num = train['logits_text'].size(1)
    popularity = load_item_popularity(args.data_root, train['dataset'], item_num)
    configs, alpha_grid = candidate_policy_configs(), POLICY_ALPHA_GRID
    train_ranks, train_text_ranks, train_fixed_ranks = evaluate_policy_grid(
        train, configs, alpha_grid, args.batch_size, args.device
    )
    selected, selected_by_config, selected_by_family = select_train_configurations(
        configs, alpha_grid, train_ranks, train_text_ranks
    )
    save_json(args.selected_json, {
        'selected_on': 'calibration_train', 'selected_rule': selected,
        'selected_by_config': selected_by_config, 'selected_by_family': selected_by_family,
    })

    frozen = frozen_config_list(selected_by_config)
    valid_ranks, valid_text_ranks, valid_fixed_ranks = evaluate_frozen_configs(
        valid, frozen, args.batch_size, args.device
    )
    key_to_index = {config['key']: index for index, config in enumerate(frozen)}
    selected_index = key_to_index[selected['key']]
    best_valid_ranks = valid_ranks[:, selected_index]
    text_metrics = metrics_from_ranks(valid_text_ranks, valid_text_ranks)
    fixed_metrics = metrics_from_ranks(valid_fixed_ranks, valid_text_ranks)
    best_metrics = metrics_from_ranks(best_valid_ranks, valid_text_ranks)
    best_metrics.update(gate_statistics(valid, selected, args.batch_size, args.device))
    bootstrap = paired_bootstrap(per_sample_dcg10(best_valid_ranks) - per_sample_dcg10(valid_text_ranks), seed=args.seed)

    family_keys = ['P2_id_topk', 'P3_intersection', 'P4_union', 'P5_rank_agreement', 'P6_soft_consensus']
    family_columns = [valid_text_ranks, valid_fixed_ranks] + [
        valid_ranks[:, key_to_index[selected_by_family[key]['key']]] for key in family_keys
    ]
    policy_oracle = oracle_from_columns(torch.stack(family_columns, dim=-1), valid_text_ranks)
    constrained_keys = [
        'P2_id_topk_k10', 'P2_id_topk_k50', 'P2_id_topk_k100',
        'P3_intersection_t10_i10', 'P3_intersection_t50_i50', 'P3_intersection_t100_i100',
    ]
    constrained = oracle_from_columns(torch.stack(
        [valid_text_ranks] + [valid_ranks[:, key_to_index[key]] for key in constrained_keys], dim=-1
    ), valid_text_ranks)
    fixed_rule_go = (
        best_metrics['NDCG@10'] > text_metrics['NDCG@10']
        and bootstrap['mean_difference'] > 0
        and bootstrap['probability_improvement_gt_zero'] >= 0.90
        and best_metrics['harm_rate'] < fixed_metrics['harm_rate']
    )
    oracle_space = policy_oracle['NDCG@10'] - text_metrics['NDCG@10']
    go_no_go_1 = fixed_rule_go or oracle_space >= 0.0005
    feature_dim = len(CANDIDATE_FEATURE_NAMES)
    result = {
        'protocol': 'all rule configurations selected on calibration_train; calibration_valid evaluated once',
        'dataset': train['dataset'], 'seed': args.seed,
        'cache_audit': {
            'train_samples': train['labels'].numel(), 'valid_samples': valid['labels'].numel(),
            'num_items': item_num, 'candidate_feature_dim': feature_dim,
            'full_train_feature_bytes': train['labels'].numel() * item_num * feature_dim * 4,
            'batch_chunk_feature_bytes': args.batch_size * args.candidate_chunk * feature_dim * 4,
            'sample_batch_size': args.batch_size, 'candidate_chunk': args.candidate_chunk,
            'item_popularity_nonzero': int((popularity > 0).sum().item()),
        },
        'selected_rule_train': selected,
        'text_only_valid': text_metrics, 'v1_fixed_valid': fixed_metrics,
        'best_candidate_rule_valid': best_metrics,
        'bootstrap_best_rule_vs_text': bootstrap,
        'policy_family_oracle_valid': policy_oracle,
        'constrained_candidate_policy_oracle_valid': constrained,
        'fixed_rule_go': fixed_rule_go, 'policy_family_oracle_gain': oracle_space,
        'go_no_go_1': go_no_go_1,
        'decision': 'enter_linear_candidate_gate' if go_no_go_1 else 'candidate_fusion_no_go',
    }
    save_json(args.output_json, result)
    print({
        'selected': selected, 'text_ndcg10': text_metrics['NDCG@10'],
        'rule_ndcg10': best_metrics['NDCG@10'], 'oracle_ndcg10': policy_oracle['NDCG@10'],
        'bootstrap': bootstrap, 'decision': result['decision'],
    })
    print(f'Saved candidate policy analysis: {args.output_json}')


if __name__ == '__main__':
    main()

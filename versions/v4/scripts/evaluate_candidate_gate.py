import argparse
import json

import torch

from analyze_candidate_policies import load_item_popularity
from candidate_features import CANDIDATE_FEATURE_NAMES, CANDIDATE_FEATURE_SCHEMA_VERSION, rank_positions
from candidate_gate import load_candidate_gate_checkpoint
from candidate_gate_common import evaluate_gate
from common import load_cache, save_json
from paired_bootstrap import paired_bootstrap


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


@torch.no_grad()
def rank_disagreement(cache, batch_size, device):
    values = []
    denominator = max(1, cache['logits_text'].size(1) - 1)
    for start in range(0, cache['labels'].numel(), batch_size):
        end = min(start + batch_size, cache['labels'].numel())
        text_rank = rank_positions(cache['logits_text'][start:end].float().to(device))
        id_rank = rank_positions(cache['logits_id'][start:end].float().to(device))
        values.append(((text_rank.float() - id_rank.float()).abs().mean(dim=-1) / denominator).cpu())
    return torch.cat(values)


def grouped_analysis(train_values, valid_values, final_ranks, text_ranks):
    boundaries = torch.quantile(train_values.float(), torch.tensor([1.0 / 3.0, 2.0 / 3.0])).unique()
    buckets = torch.bucketize(valid_values.float().contiguous(), boundaries)
    final_dcg = (final_ranks < 10).float() / torch.log2(final_ranks + 2.0)
    text_dcg = (text_ranks < 10).float() / torch.log2(text_ranks + 2.0)
    rows = []
    for bucket in range(boundaries.numel() + 1):
        mask = buckets == bucket
        if not mask.any():
            continue
        rows.append({
            'bucket': bucket, 'num_samples': int(mask.sum().item()),
            'text_NDCG@10': text_dcg[mask].mean().item(),
            'gate_NDCG@10': final_dcg[mask].mean().item(),
            'mean_difference': (final_dcg[mask] - text_dcg[mask]).mean().item(),
            'harm_rate': (final_ranks[mask] > text_ranks[mask]).float().mean().item(),
            'benefit_rate': (final_ranks[mask] < text_ranks[mask]).float().mean().item(),
        })
    return {'train_boundaries': boundaries.tolist(), 'buckets': rows}


def main():
    parser = argparse.ArgumentParser(description='One-shot independent validation of frozen candidate gates.')
    parser.add_argument('--train_cache', required=True, help='Used only for training-derived group boundaries.')
    parser.add_argument('--valid_cache', required=True)
    parser.add_argument('--training_summary', required=True)
    parser.add_argument('--policy_results', required=True)
    parser.add_argument('--residual_results', required=True)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--candidate_chunk', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train, valid = load_cache(args.train_cache), load_cache(args.valid_cache)
    if set(train['user_ids'].tolist()).intersection(valid['user_ids'].tolist()):
        raise ValueError('Independent calibration-valid users overlap calibration-train.')
    summary, policy, residual = load_json(args.training_summary), load_json(args.policy_results), load_json(args.residual_results)
    if summary['dataset'] != valid['dataset'] or valid['split'] != 'calibration_valid':
        raise ValueError('Candidate checkpoint dataset/split mismatch.')
    popularity = load_item_popularity(args.data_root, valid['dataset'], valid['logits_text'].size(1)).to(args.device)
    all_valid = torch.arange(valid['labels'].numel())
    evaluations, raw_outputs = {}, {}
    for architecture, checkpoint_path in summary['checkpoints'].items():
        model, normalizer, payload = load_candidate_gate_checkpoint(checkpoint_path)
        if payload['dataset'] != valid['dataset']:
            raise ValueError('Candidate gate checkpoint dataset mismatch.')
        if payload['feature_schema'] != CANDIDATE_FEATURE_SCHEMA_VERSION or payload['feature_names'] != CANDIDATE_FEATURE_NAMES:
            raise ValueError('Candidate feature schema mismatch.')
        model.to(args.device)
        metrics, dcg, ranks, text_ranks = evaluate_gate(
            valid, all_valid, popularity, model, normalizer, payload['max_alpha'],
            args.batch_size, args.candidate_chunk, args.device,
        )
        text_dcg = (text_ranks < 10).float() / torch.log2(text_ranks + 2.0)
        bootstrap = paired_bootstrap(dcg - text_dcg, seed=args.seed)
        evaluations[architecture] = {
            **metrics, 'bootstrap_vs_text': bootstrap,
            'max_alpha': payload['max_alpha'], 'parameter_count': payload['parameter_count'],
            'extra_flops_per_candidate_approx': 2 * payload['input_dim'] if architecture == 'linear' else 2 * (payload['input_dim'] * payload['hidden_dim'] + payload['hidden_dim']),
        }
        raw_outputs[architecture] = (ranks, text_ranks)

    selected_name = summary['selected_architecture_on_gate_dev']
    selected = evaluations[selected_name]
    selected_ranks, text_ranks = raw_outputs[selected_name]
    train_popularity = popularity.cpu()[train['labels'].long()]
    valid_popularity = popularity.cpu()[valid['labels'].long()]
    train_disagreement = rank_disagreement(train, args.batch_size, args.device)
    valid_disagreement = rank_disagreement(valid, args.batch_size, args.device)
    dimensions = {
        'history_length': (train['history_length'], valid['history_length']),
        'target_item_popularity': (train_popularity, valid_popularity),
        'text_entropy': (train['features'][:, 0], valid['features'][:, 0]),
        'id_entropy': (train['features'][:, 1], valid['features'][:, 1]),
        'branch_jsd': (train['features'][:, 4], valid['features'][:, 4]),
        'text_id_rank_disagreement': (train_disagreement, valid_disagreement),
    }
    groups = {
        name: grouped_analysis(train_value, valid_value, selected_ranks, text_ranks)
        for name, (train_value, valid_value) in dimensions.items()
    }
    positive_dimensions = sum(
        any(row['mean_difference'] > 0 for row in report['buckets']) for report in groups.values()
    )
    text_ndcg = policy['text_only_valid']['NDCG@10']
    fixed = policy['v1_fixed_valid']
    not_collapsed = (
        selected['gate_std'] > 1e-6
        and selected['gate_near_zero_rate'] < 0.999
        and selected['gate_gt_half_rate'] < 0.999
    )
    go = (
        selected['NDCG@10'] > text_ndcg
        and selected['NDCG@10'] > fixed['NDCG@10']
        and selected['bootstrap_vs_text']['probability_improvement_gt_zero'] >= 0.90
        and selected['harm_rate'] < fixed['harm_rate']
        and not_collapsed
        and positive_dimensions >= 2
        and selected['NDCG@10'] - text_ndcg >= 0.0005
    )
    strong_go = go and selected['bootstrap_vs_text']['ci95_low'] > 0
    result = {
        'protocol': 'architecture/config selected on gate_dev; each frozen architecture evaluated once on calibration_valid',
        'selected_architecture_on_gate_dev': selected_name,
        'text_only': policy['text_only_valid'], 'v1_fixed': fixed,
        'raw_best_global_alpha_zero': policy['text_only_valid'],
        'r5_calibrated_alpha_point_one': residual['best_calibrated_valid'],
        'best_candidate_rule': policy['best_candidate_rule_valid'],
        'candidate_gates': evaluations,
        'constrained_candidate_policy_oracle': policy['constrained_candidate_policy_oracle_valid'],
        'raw_sequence_oracle_alpha': residual['raw_oracle_valid'],
        'group_analysis': groups, 'positive_group_dimensions': positive_dimensions,
        'selected_gate_not_collapsed': not_collapsed,
        'go': go, 'strong_go': strong_go,
        'decision': 'Strong Go' if strong_go else ('Go' if go else 'No-Go'),
    }
    save_json(args.output_json, result)
    print({
        'selected_architecture': selected_name, 'selected_gate': selected,
        'positive_group_dimensions': positive_dimensions, 'decision': result['decision'],
    })
    print(f'Saved independent candidate gate evaluation: {args.output_json}')


if __name__ == '__main__':
    main()

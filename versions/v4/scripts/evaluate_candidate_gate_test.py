import argparse
import json
from pathlib import Path

import torch

from analyze_candidate_policies import load_item_popularity
from candidate_features import CANDIDATE_FEATURE_NAMES, CANDIDATE_FEATURE_SCHEMA_VERSION
from candidate_gate import load_candidate_gate_checkpoint
from candidate_gate_common import evaluate_gate
from common import load_cache, save_json
from paired_bootstrap import paired_bootstrap
from ranking_metrics import ranking_metrics, target_ranks
from utility_label import fixed_text_fusion
from provenance import sha256, validate_provenance


def sample_dcg10(ranks):
    return (ranks < 10).float() / torch.log2(ranks + 2.0)


def comparison_metrics(scores, text, labels):
    ranks, text_ranks = target_ranks(scores, labels), target_ranks(text, labels)
    metrics = ranking_metrics(scores, labels)
    metrics.update({
        'harm_rate': (ranks > text_ranks).float().mean().item(),
        'benefit_rate': (ranks < text_ranks).float().mean().item(),
        'average_target_rank_gain': (text_ranks - ranks).mean().item(),
    })
    return metrics, ranks, text_ranks


def write_markdown(path, result):
    rows = [
        ('Text-only', result['text_only']),
        ('ID-only', result['id_only']),
        ('v1 fixed', result['v1_fixed']),
        ('v4 Candidate Gate', result['v4_candidate_gate']),
        ('Best global alpha', result['test_oracle']['best_global_metrics']),
        ('Oracle-Alpha', result['test_oracle']['oracle_alpha']),
    ]
    lines = [
        f"# {result['dataset']} v4 Final Test", '',
        '| Method | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | MRR |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['NDCG@5']:.6f} | {metrics['NDCG@10']:.6f} | "
            f"{metrics['Recall@5']:.6f} | {metrics['Recall@10']:.6f} | {metrics['MRR']:.6f} |"
        )
    lines.extend([
        '', f"- Best global alpha: {result['test_oracle']['best_global_alpha']}",
        f"- v4 vs text NDCG@10 difference: {result['v4_vs_text']['mean_difference']:.6f}",
        f"- v4 vs text bootstrap: {result['bootstrap_v4_vs_text']}",
        '- Test was used once for frozen-model evaluation and not for model selection.',
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='One-shot Industrial test evaluation of the final Candidate Gate.')
    parser.add_argument('--test_cache', required=True)
    parser.add_argument('--candidate_checkpoint', required=True)
    parser.add_argument('--oracle_json', required=True)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_markdown', default=None)
    parser.add_argument('--diagnostics_json', default=None)
    parser.add_argument('--official-only', action='store_true')
    parser.add_argument('--official_log', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--candidate_chunk', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cache = load_cache(args.test_cache)
    if cache['split'] != 'test':
        raise ValueError('Final evaluation requires a test cache.')
    model, normalizer, checkpoint = load_candidate_gate_checkpoint(args.candidate_checkpoint)
    if 'v1_provenance' not in cache or 'v1_provenance' not in checkpoint:
        raise ValueError('Official Candidate Gate evaluation requires v1 provenance in cache and gate checkpoint.')
    validate_provenance(cache['v1_provenance'], checkpoint['v1_provenance'])
    if checkpoint['dataset'] != cache['dataset']:
        raise ValueError('Candidate checkpoint and test cache datasets differ.')
    if checkpoint['feature_schema'] != CANDIDATE_FEATURE_SCHEMA_VERSION or checkpoint['feature_names'] != CANDIDATE_FEATURE_NAMES:
        raise ValueError('Candidate feature schema mismatch.')
    with open(args.oracle_json) as handle:
        oracle = json.load(handle)
    if oracle['split'] != 'test' or oracle['dataset'] != cache['dataset']:
        raise ValueError('Oracle report does not belong to this test cache.')

    popularity = load_item_popularity(args.data_root, cache['dataset'], cache['logits_text'].size(1)).to(args.device)
    model.to(args.device)
    indices = torch.arange(cache['labels'].numel())
    gate_metrics, gate_dcg, gate_ranks, text_ranks = evaluate_gate(
        cache, indices, popularity, model, normalizer, checkpoint['max_alpha'],
        args.batch_size, args.candidate_chunk, args.device,
    )
    text, ids, labels = cache['logits_text'].float(), cache['logits_id'].float(), cache['labels'].long()
    fixed = fixed_text_fusion(text, ids, alpha=float(cache['alpha0']), temperature=float(cache['fusion_temperature']))
    fixed_metrics, fixed_ranks, _ = comparison_metrics(fixed, text, labels)
    text_metrics = ranking_metrics(text, labels)
    id_metrics = ranking_metrics(ids, labels)
    text_dcg = sample_dcg10(text_ranks)
    result = {
        'protocol': 'frozen final Candidate Gate evaluated once on test; test not used for selection',
        'dataset': cache['dataset'], 'num_samples': labels.numel(),
        'checkpoint': str(Path(args.candidate_checkpoint).resolve()),
        'text_only': text_metrics, 'id_only': id_metrics,
        'v1_fixed': fixed_metrics, 'v4_candidate_gate': gate_metrics,
        'test_oracle': {
            'best_global_alpha': oracle['best_global_alpha'],
            'best_global_metrics': oracle['best_global_metrics'],
            'oracle_alpha': oracle['oracle_alpha'],
        },
        'v4_vs_text': {
            'mean_difference': gate_metrics['NDCG@10'] - text_metrics['NDCG@10'],
            'relative_difference': (gate_metrics['NDCG@10'] / text_metrics['NDCG@10'] - 1.0),
        },
        'v4_vs_v1_fixed': {
            'mean_difference': gate_metrics['NDCG@10'] - fixed_metrics['NDCG@10'],
            'relative_difference': (gate_metrics['NDCG@10'] / fixed_metrics['NDCG@10'] - 1.0),
        },
        'bootstrap_v4_vs_text': paired_bootstrap(gate_dcg - text_dcg, seed=args.seed),
        'bootstrap_v4_vs_v1_fixed': paired_bootstrap(gate_dcg - sample_dcg10(fixed_ranks), seed=args.seed),
    }
    if args.official_only:
        official = {
            'method': 'v4_candidate_gate', 'uses_candidate_gate': True,
            'dataset': cache['dataset'], 'num_samples': labels.numel(),
            'metrics': gate_metrics,
            'candidate_gate_checkpoint': str(Path(args.candidate_checkpoint).resolve()),
            'candidate_gate_checkpoint_sha256': sha256(args.candidate_checkpoint),
            'v1_provenance': cache['v1_provenance'],
            'feature_schema': checkpoint['feature_schema'],
            'gate_config': {
                key: checkpoint[key] for key in ('architecture', 'hidden_dim', 'dropout', 'max_alpha', 'selected_epoch') if key in checkpoint
            },
        }
        save_json(args.output_json, official)
        if not args.diagnostics_json:
            raise ValueError('--official-only requires --diagnostics_json.')
        save_json(args.diagnostics_json, result)
        lines = ['==Test set (v4 Candidate Gate)==']
        for name in ('NDCG@5', 'NDCG@10', 'Recall@5', 'Recall@10', 'MRR'):
            lines.append(f'{name}: {gate_metrics[name]:.6f}')
        lines.extend([
            f'dataset: {cache["dataset"]}',
            f'v1 checkpoint: {cache["v1_provenance"]["v1_checkpoint_path"]}',
            f'v1 checkpoint sha256: {cache["v1_provenance"]["v1_checkpoint_sha256"]}',
            f'v1 profile: {cache["v1_provenance"]["v1_profile"]}',
            f'candidate gate checkpoint: {Path(args.candidate_checkpoint).resolve()}',
            f'candidate gate sha256: {sha256(args.candidate_checkpoint)}',
            f'feature schema: {checkpoint["feature_schema"]}',
        ])
        text = '\n'.join(lines) + '\n'
        if args.official_log:
            Path(args.official_log).parent.mkdir(parents=True, exist_ok=True)
            Path(args.official_log).write_text(text)
        print(text, end='')
        return
    save_json(args.output_json, result)
    if args.output_markdown:
        write_markdown(args.output_markdown, result)
    print(result)
    print(f'Saved final test report: {args.output_json}')


if __name__ == '__main__':
    main()

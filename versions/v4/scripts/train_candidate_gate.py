import argparse
import copy
from pathlib import Path

import torch

from analyze_candidate_policies import load_item_popularity
from candidate_features import CANDIDATE_FEATURE_NAMES, CANDIDATE_FEATURE_SCHEMA_VERSION
from candidate_gate import CandidateGate, candidate_gate_checkpoint
from candidate_gate_common import (
    evaluate_gate,
    fit_candidate_normalizer,
    split_gate_users,
    train_epoch,
)
from common import load_cache, save_json, set_seed
from ranking_metrics import target_ranks


def text_ndcg10(cache, indices):
    ranks = target_ranks(cache['logits_text'][indices].float(), cache['labels'][indices])
    return ((ranks < 10).float() / torch.log2(ranks + 2.0)).mean().item()


def gate_config_grid():
    return [
        {'max_alpha': alpha, 'lambda_sparse': sparse, 'lambda_safe': safe}
        for alpha in (0.25, 0.5, 1.0)
        for sparse in (0.001, 0.01)
        for safe in (0.0, 0.001)
    ]


def select_architecture_config(cache, train_indices, dev_indices, popularity, architecture, args):
    # Deliberately has no independent validation cache argument.
    normalizer = fit_candidate_normalizer(
        cache, train_indices, popularity, args.batch_size, args.candidate_chunk, args.device
    )
    trials = []
    for trial_index, config in enumerate(gate_config_grid()):
        set_seed(args.seed)
        model = CandidateGate(
            len(CANDIDATE_FEATURE_NAMES), architecture=architecture,
            initial_probability=args.initial_gate_probability,
        ).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best = None
        stale = 0
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(
                cache, train_indices, popularity, model, normalizer, optimizer,
                config['max_alpha'], config['lambda_sparse'], config['lambda_safe'],
                args.batch_size, args.candidate_chunk, args.device,
                args.seed + trial_index * 100 + epoch,
            )
            metrics, _, _, _ = evaluate_gate(
                cache, dev_indices, popularity, model, normalizer, config['max_alpha'],
                args.batch_size, args.candidate_chunk, args.device,
            )
            row = {**config, 'architecture': architecture, 'epoch': epoch, 'train_loss': loss, **metrics}
            ordering = (-row['NDCG@10'], row['harm_rate'], row['max_alpha'], row['epoch'])
            if best is None or ordering < best[0]:
                best = (ordering, row, copy.deepcopy(model.state_dict()))
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        trials.append(best[1])
        print({'architecture': architecture, 'trial': trial_index + 1, 'best_dev': best[1]})
    selected = min(trials, key=lambda row: (-row['NDCG@10'], row['harm_rate'], row['max_alpha'], row['lambda_sparse'], row['lambda_safe']))
    return selected, trials


def retrain_selected(cache, all_indices, popularity, architecture, selected, output_path, args):
    normalizer = fit_candidate_normalizer(
        cache, all_indices, popularity, args.batch_size, args.candidate_chunk, args.device
    )
    set_seed(args.seed)
    model = CandidateGate(
        len(CANDIDATE_FEATURE_NAMES), architecture=architecture,
        initial_probability=args.initial_gate_probability,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    for epoch in range(1, int(selected['epoch']) + 1):
        losses.append(train_epoch(
            cache, all_indices, popularity, model, normalizer, optimizer,
            selected['max_alpha'], selected['lambda_sparse'], selected['lambda_safe'],
            args.batch_size, args.candidate_chunk, args.device, args.seed + epoch,
        ))
    checkpoint = candidate_gate_checkpoint(model, normalizer, {
        'feature_schema': CANDIDATE_FEATURE_SCHEMA_VERSION,
        'feature_names': CANDIDATE_FEATURE_NAMES,
        'dataset': cache['dataset'], 'seed': args.seed,
        'max_alpha': selected['max_alpha'],
        'lambda_sparse': selected['lambda_sparse'], 'lambda_safe': selected['lambda_safe'],
        'selected_epoch': int(selected['epoch']), 'train_losses': losses,
        'selection_split': 'gate_dev within calibration_train',
        'source_checkpoint': cache['checkpoint_path'],
        'parameter_count': model.parameter_count,
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train low-capacity candidate gates without independent-valid access.')
    parser.add_argument('--train_cache', required=True)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--candidate_chunk', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--initial_gate_probability', type=float, default=0.01, choices=[0.01, 0.05])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cache = load_cache(args.train_cache)
    if cache['dataset'] != 'Industrial_and_Scientific' or cache['split'] != 'calibration_train':
        raise ValueError('Candidate gate training requires Industrial calibration_train only.')
    train_indices, dev_indices = split_gate_users(cache, seed=args.seed)
    if set(cache['user_ids'][train_indices].tolist()).intersection(cache['user_ids'][dev_indices].tolist()):
        raise RuntimeError('gate_train and gate_dev users overlap.')
    popularity = load_item_popularity(args.data_root, cache['dataset'], cache['logits_text'].size(1)).to(args.device)
    dev_text = text_ndcg10(cache, dev_indices)
    output_dir = Path(args.output_dir)

    linear_selected, linear_trials = select_architecture_config(
        cache, train_indices, dev_indices, popularity, 'linear', args
    )
    train_tiny = linear_selected['NDCG@10'] > dev_text
    architectures = {'linear': {'selected': linear_selected, 'trials': linear_trials}}
    if train_tiny:
        tiny_selected, tiny_trials = select_architecture_config(
            cache, train_indices, dev_indices, popularity, 'tiny_mlp', args
        )
        architectures['tiny_mlp'] = {'selected': tiny_selected, 'trials': tiny_trials}

    selected_architecture = min(
        architectures,
        key=lambda name: (
            -architectures[name]['selected']['NDCG@10'],
            architectures[name]['selected']['harm_rate'],
            0 if name == 'linear' else 1,
        ),
    )
    all_indices = torch.arange(cache['labels'].numel())
    checkpoints = {}
    for architecture, details in architectures.items():
        path = output_dir / f'{architecture}_candidate_gate.pt'
        payload = retrain_selected(cache, all_indices, popularity, architecture, details['selected'], path, args)
        checkpoints[architecture] = str(path.resolve())
        details['parameter_count'] = payload['parameter_count']
    summary = {
        'protocol': 'gate_train/gate_dev user-disjoint selection within calibration_train; no calibration_valid access',
        'dataset': cache['dataset'], 'seed': args.seed,
        'gate_train_samples': train_indices.numel(), 'gate_dev_samples': dev_indices.numel(),
        'gate_train_users': torch.unique(cache['user_ids'][train_indices]).numel(),
        'gate_dev_users': torch.unique(cache['user_ids'][dev_indices]).numel(),
        'gate_dev_text_ndcg10': dev_text,
        'linear_positive_trend': train_tiny,
        'tiny_mlp_trained': train_tiny,
        'architectures': architectures,
        'selected_architecture_on_gate_dev': selected_architecture,
        'checkpoints': checkpoints,
    }
    save_json(args.summary_json, summary)
    print({
        'dev_text': dev_text, 'linear_selected': linear_selected,
        'tiny_mlp_trained': train_tiny, 'selected_architecture': selected_architecture,
    })
    print(f'Saved candidate gate training summary: {args.summary_json}')


if __name__ == '__main__':
    main()

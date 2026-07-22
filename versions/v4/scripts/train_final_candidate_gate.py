import argparse
import json
from pathlib import Path

import torch

from analyze_candidate_policies import load_item_popularity
from common import load_cache, save_json
from train_candidate_gate import retrain_selected
from provenance import validate_provenance


def merge_calibration_caches(train, valid):
    if train['dataset'] != valid['dataset']:
        raise ValueError('Calibration datasets do not match.')
    if train['feature_schema'] != valid['feature_schema']:
        raise ValueError('Calibration feature schemas do not match.')
    overlap = set(train['user_ids'].tolist()).intersection(valid['user_ids'].tolist())
    if overlap:
        raise ValueError(f'Calibration users overlap: {len(overlap)}')
    train_n, valid_n = train['labels'].numel(), valid['labels'].numel()
    merged = {}
    for key, value in train.items():
        other = valid.get(key)
        if (
            torch.is_tensor(value) and torch.is_tensor(other)
            and value.ndim > 0 and other.ndim > 0
            and value.size(0) == train_n and other.size(0) == valid_n
            and value.shape[1:] == other.shape[1:]
        ):
            merged[key] = torch.cat([value, other], dim=0)
        else:
            merged[key] = value
    merged['split'] = 'final_fuser_train'
    return merged


def main():
    parser = argparse.ArgumentParser(description='Retrain the frozen-config Candidate Gate on all calibration data.')
    parser.add_argument('--calibration_train', required=True)
    parser.add_argument('--calibration_valid', required=True)
    parser.add_argument('--training_summary', required=True)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--output_checkpoint', required=True)
    parser.add_argument('--output_manifest', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--candidate_chunk', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--initial_gate_probability', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train = load_cache(args.calibration_train)
    valid = load_cache(args.calibration_valid)
    merged = merge_calibration_caches(train, valid)
    with open(args.training_summary) as handle:
        summary = json.load(handle)
    if 'v1_provenance' not in merged or 'v1_provenance' not in summary:
        raise ValueError('Final Candidate Gate requires v1 provenance in caches and training summary.')
    validate_provenance(merged['v1_provenance'], summary['v1_provenance'])
    architecture = summary['selected_architecture_on_gate_dev']
    selected = summary['architectures'][architecture]['selected']
    popularity = load_item_popularity(
        args.data_root, merged['dataset'], merged['logits_text'].size(1)
    ).to(args.device)
    all_indices = torch.arange(merged['labels'].numel())
    payload = retrain_selected(
        merged, all_indices, popularity, architecture, selected,
        args.output_checkpoint, args,
    )
    manifest = {
        'protocol': 'frozen gate-dev configuration retrained on calibration_train + calibration_valid; no test access',
        'dataset': merged['dataset'], 'architecture': architecture,
        'frozen_config': selected,
        'num_samples': merged['labels'].numel(),
        'num_users': torch.unique(merged['user_ids']).numel(),
        'calibration_train_path': str(Path(args.calibration_train).resolve()),
        'calibration_valid_path': str(Path(args.calibration_valid).resolve()),
        'output_checkpoint': str(Path(args.output_checkpoint).resolve()),
        'parameter_count': payload['parameter_count'], 'seed': args.seed,
        'v1_provenance': merged['v1_provenance'],
    }
    save_json(args.output_manifest, manifest)
    print(manifest)
    print(f'Saved final Candidate Gate: {args.output_checkpoint}')


if __name__ == '__main__':
    main()

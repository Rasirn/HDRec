import argparse
from pathlib import Path

import torch

from common import load_cache
from alpha_targets import build_alpha_target_payload, compute_alpha_outcomes, make_alpha_grid


def main():
    parser = argparse.ArgumentParser(description='Generate multi-alpha supervision from frozen logits.')
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--alpha_max', type=float, default=1.0)
    parser.add_argument('--alpha_step', type=float, default=0.05)
    parser.add_argument('--tau_ce', type=float, default=0.5)
    parser.add_argument('--tau_metric', type=float, default=0.1)
    parser.add_argument('--beta_rr', type=float, default=0.1)
    parser.add_argument('--beta_ce', type=float, default=0.05)
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        print(f'Alpha target cache exists, reusing: {output_path}')
        return
    cache = load_cache(args.cache_path)
    alpha_grid = make_alpha_grid(args.alpha_max, args.alpha_step)
    outcomes = compute_alpha_outcomes(
        cache['logits_text'],
        cache['logits_id'],
        cache['labels'],
        alpha_grid,
        cache['fusion_temperature'],
        device=args.device,
        chunk_size=args.chunk_size,
    )
    payload = build_alpha_target_payload(
        cache,
        args.cache_path,
        alpha_grid,
        outcomes,
        tau_ce=args.tau_ce,
        tau_metric=args.tau_metric,
        beta_rr=args.beta_rr,
        beta_ce=args.beta_ce,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    counts = torch.bincount(payload['best_alpha_rank_index'], minlength=alpha_grid.numel())
    print({
        'output_path': str(output_path),
        'dataset': payload['dataset'],
        'split': payload['split'],
        'num_samples': int(cache['labels'].numel()),
        'alpha_grid': alpha_grid.tolist(),
        'hard_alpha_counts': counts.tolist(),
    })


if __name__ == '__main__':
    main()

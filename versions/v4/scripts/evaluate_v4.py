import argparse
from pathlib import Path

import torch

from analyze_oracle import oracle_alpha_scores, oracle_select_scores
from common import load_cache, save_json
from ranking_metrics import ranking_metrics, summarize_alpha, target_ranks
from reliability_fusion import load_fuser
from utility_label import fixed_text_fusion, id_confidence_residual


def harm_benefit(text, candidate, labels):
    rt = target_ranks(text, labels)
    rc = target_ranks(candidate, labels)
    return {
        'harm_rate': (rc > rt).float().mean().item(),
        'benefit_rate': (rc < rt).float().mean().item(),
        'average_target_rank_gain': (rt - rc).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate v4 fuser and baselines from cached logits.')
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--fuser_path', required=True)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--alpha_max', type=float, default=None)
    parser.add_argument('--alpha_step', type=float, default=0.1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cache = load_cache(args.cache_path)
    text = cache['logits_text'].float()
    ids = cache['logits_id'].float()
    labels = cache['labels'].long()
    alpha0 = float(cache.get('alpha0', 0.5))
    temp = float(cache.get('fusion_temperature', 1.0))

    fixed = fixed_text_fusion(text, ids, alpha=alpha0, temperature=temp)
    select_scores, select_use_fixed = oracle_select_scores(text, fixed, labels)
    grid_max = float(args.alpha_max if args.alpha_max is not None else max(1.0, alpha0))
    grid = torch.arange(0.0, grid_max + args.alpha_step / 2, args.alpha_step).tolist()
    alpha_scores, best_alpha = oracle_alpha_scores(text, ids, labels, grid, temp)

    fuser, scaler, payload = load_fuser(args.fuser_path, hidden_dim=args.hidden_dim, dropout=args.dropout, map_location=args.device)
    device = torch.device(args.device)
    fuser.to(device).eval()
    features = scaler.transform(cache['features']).to(device)
    residual = id_confidence_residual(ids.to(device), temperature=temp)
    with torch.no_grad():
        v4_scores, alpha = fuser(text.to(device), residual, features)
    v4_scores = v4_scores.cpu()
    alpha = alpha.cpu()

    result = {
        'dataset': cache.get('dataset'),
        'split': cache.get('split'),
        'fuser_path': args.fuser_path,
        'text_only': ranking_metrics(text, labels),
        'id_only': ranking_metrics(ids, labels),
        'v1_fixed': ranking_metrics(fixed, labels),
        'v4_context': ranking_metrics(v4_scores, labels),
        'oracle_select': ranking_metrics(select_scores, labels),
        'oracle_alpha': ranking_metrics(alpha_scores, labels),
        'v4_harm_benefit': harm_benefit(text, v4_scores, labels),
        'fixed_harm_benefit': harm_benefit(text, fixed, labels),
        'oracle_select_use_fixed_rate': select_use_fixed.float().mean().item(),
        'oracle_alpha_zero_rate': (best_alpha == 0).float().mean().item(),
        **summarize_alpha(alpha),
        'fuser_extra': payload.get('extra', {}),
    }
    out_path = args.output_json or str(Path(args.cache_path).with_suffix('.v4_eval.json'))
    save_json(out_path, result)
    print(result)
    print(f'Saved v4 evaluation: {out_path}')


if __name__ == '__main__':
    main()

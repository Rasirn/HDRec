import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_cache, save_json
from ranking_metrics import ranking_metrics, summarize_alpha
from reliability_fusion import FeatureStandardizer, ReliabilityFuser, save_fuser
from utility_label import cross_entropy_per_sample, id_confidence_residual


def evaluate_fuser(fuser, scaler, cache, temperature, device):
    fuser.eval()
    text = cache['logits_text'].float().to(device)
    ids = cache['logits_id'].float().to(device)
    labels = cache['labels'].long().to(device)
    features = scaler.transform(cache['features']).to(device)
    residual = id_confidence_residual(ids, temperature=temperature)
    with torch.no_grad():
        final, alpha = fuser(text, residual, features)
        metrics = ranking_metrics(final.cpu(), labels.cpu())
        metrics.update(summarize_alpha(alpha.cpu()))
        metrics['loss'] = F.cross_entropy(final.float(), labels).item()
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train v4 Context Gate from cached frozen-v1 logits.')
    parser.add_argument('--train_cache', required=True)
    parser.add_argument('--valid_cache', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha0', type=float, default=None)
    parser.add_argument('--alpha_max', type=float, default=1.0)
    parser.add_argument('--alpha_rho', type=float, default=0.5)
    parser.add_argument('--utility_loss_weight', type=float, default=0.1)
    parser.add_argument('--shrink_loss_weight', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = load_cache(args.train_cache)
    valid = load_cache(args.valid_cache)
    feature_names = train['feature_names']
    temperature = float(train.get('fusion_temperature', 1.0))
    alpha0 = float(args.alpha0 if args.alpha0 is not None else train.get('alpha0', 0.5))

    scaler = FeatureStandardizer().fit(train['features'])
    features = scaler.transform(train['features']).to(device)
    text = train['logits_text'].float().to(device)
    ids = train['logits_id'].float().to(device)
    labels = train['labels'].long().to(device)
    utility = train['utility_label'].float().to(device)
    residual = id_confidence_residual(ids, temperature=temperature)

    fuser = ReliabilityFuser(
        input_dim=features.size(1),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        alpha0=alpha0,
        alpha_max=args.alpha_max,
        rho=args.alpha_rho,
    ).to(device)
    utility_head = torch.nn.Linear(features.size(1), 1).to(device)
    opt = torch.optim.AdamW(
        list(fuser.parameters()) + list(utility_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best = float('-inf')
    bad = 0
    history = []
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = labels.numel()
    for epoch in range(args.epochs):
        fuser.train()
        utility_head.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            final, alpha = fuser(text[idx], residual[idx], features[idx])
            rank_loss = F.cross_entropy(final.float(), labels[idx])
            util_logits = utility_head(features[idx]).squeeze(-1)
            util_loss = F.binary_cross_entropy_with_logits(util_logits, utility[idx])
            shrink_loss = ((alpha - alpha0) ** 2).mean()
            loss = rank_loss + args.utility_loss_weight * util_loss + args.shrink_loss_weight * shrink_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.numel()

        metrics = evaluate_fuser(fuser, scaler, valid, temperature, device)
        metrics['epoch'] = epoch + 1
        metrics['train_loss'] = total_loss / n
        history.append(metrics)
        print(metrics)

        target = metrics.get('NDCG@10', float('-inf'))
        if target > best:
            best = target
            bad = 0
            save_fuser(
                out_path,
                fuser,
                scaler,
                feature_names,
                extra={
                    'utility_head_state': utility_head.state_dict(),
                    'temperature': temperature,
                    'train_cache': args.train_cache,
                    'valid_cache': args.valid_cache,
                    'best_epoch': epoch + 1,
                    'best_valid_ndcg10': best,
                },
            )
        else:
            bad += 1
            if bad >= args.patience:
                break

    save_json(out_path.with_suffix('.history.json'), {'history': history})
    print(f'Saved best fuser: {out_path}')


if __name__ == '__main__':
    main()

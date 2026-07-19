import torch


def target_ranks(scores, labels):
    target_scores = scores[torch.arange(scores.size(0), device=scores.device), labels.long()].unsqueeze(-1)
    return (target_scores < scores).sum(dim=-1).float()


def ranking_metrics(scores, labels, ks=(1, 5, 10, 20, 50)):
    ranks = target_ranks(scores.float(), labels.long())
    out = {}
    for k in ks:
        hit = (ranks < k).float()
        out[f'NDCG@{k}'] = ((1.0 / torch.log2(ranks + 2.0)) * hit).mean().item()
        out[f'Recall@{k}'] = hit.mean().item()
    out['MRR'] = (1.0 / (ranks + 1.0)).mean().item()
    return out


def summarize_alpha(alpha, alpha_max=None):
    alpha = alpha.float()
    qs = torch.quantile(alpha, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=alpha.device))
    summary = {
        'alpha_mean': alpha.mean().item(),
        'alpha_std': alpha.std(unbiased=False).item(),
        'alpha_min': alpha.min().item(),
        'alpha_max': alpha.max().item(),
        'alpha_p10': qs[0].item(),
        'alpha_p25': qs[1].item(),
        'alpha_p50': qs[2].item(),
        'alpha_p75': qs[3].item(),
        'alpha_p90': qs[4].item(),
    }
    if alpha_max is not None:
        summary['alpha_floor_rate'] = (alpha <= 1e-6).float().mean().item()
        summary['alpha_ceiling_rate'] = (alpha >= float(alpha_max) - 1e-6).float().mean().item()
    return summary

import torch
import torch.nn.functional as F


FEATURE_SCHEMA_VERSION = 'v4_sequence_reliability_v1'
FEATURE_NAMES = [
    'text_entropy',
    'id_entropy',
    'text_margin',
    'id_margin',
    'branch_jsd',
    'topk_overlap',
    'history_length',
    'history_pop_mean',
    'history_pop_std',
    'text_id_score_corr',
]


def _safe_probs(logits):
    return F.softmax(logits.float(), dim=-1).clamp_min(1e-12)


def entropy_from_logits(logits):
    probs = _safe_probs(logits)
    return -(probs * probs.log()).sum(dim=-1)


def margin_from_logits(logits):
    k = min(2, logits.size(-1))
    vals = torch.topk(logits.float(), k=k, dim=-1).values
    if k == 1:
        return torch.zeros(logits.size(0), device=logits.device, dtype=torch.float32)
    return vals[:, 0] - vals[:, 1]


def js_divergence(logits_a, logits_b):
    p = _safe_probs(logits_a)
    q = _safe_probs(logits_b)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def topk_overlap(logits_a, logits_b, k=10):
    k = min(k, logits_a.size(-1), logits_b.size(-1))
    top_a = torch.topk(logits_a.float(), k=k, dim=-1).indices
    top_b = torch.topk(logits_b.float(), k=k, dim=-1).indices
    matches = (top_a.unsqueeze(-1) == top_b.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
    return matches / max(1, k)


def rowwise_corr(a, b, eps=1e-8):
    a = a.float()
    b = b.float()
    ac = a - a.mean(dim=-1, keepdim=True)
    bc = b - b.mean(dim=-1, keepdim=True)
    denom = ac.norm(dim=-1) * bc.norm(dim=-1)
    corr = (ac * bc).sum(dim=-1) / denom.clamp_min(eps)
    return torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))


def sanitize_features(features):
    return torch.nan_to_num(features.float(), nan=0.0, posinf=1e6, neginf=-1e6)


def compute_sequence_features(
    logits_text,
    logits_id,
    history_length=None,
    history_pop_mean=None,
    history_pop_std=None,
    topk=10,
):
    n = logits_text.size(0)
    device = logits_text.device

    def default_vec(value=0.0):
        return torch.full((n,), float(value), device=device, dtype=torch.float32)

    if history_length is None:
        history_length = default_vec()
    else:
        history_length = history_length.to(device=device, dtype=torch.float32)
    if history_pop_mean is None:
        history_pop_mean = default_vec()
    else:
        history_pop_mean = history_pop_mean.to(device=device, dtype=torch.float32)
    if history_pop_std is None:
        history_pop_std = default_vec()
    else:
        history_pop_std = history_pop_std.to(device=device, dtype=torch.float32)

    cols = [
        entropy_from_logits(logits_text),
        entropy_from_logits(logits_id),
        margin_from_logits(logits_text),
        margin_from_logits(logits_id),
        js_divergence(logits_text, logits_id),
        topk_overlap(logits_text, logits_id, k=topk),
        history_length,
        history_pop_mean,
        history_pop_std,
        rowwise_corr(logits_text, logits_id),
    ]
    return sanitize_features(torch.stack(cols, dim=-1))


def item_popularity_from_train(user2train, item_num):
    counts = torch.zeros(item_num, dtype=torch.float32)
    for seq in user2train.values():
        for item in seq:
            if 0 <= int(item) < item_num:
                counts[int(item)] += 1.0
    return counts


def history_pop_stats(histories, popularity):
    means = []
    stds = []
    for hist in histories:
        if not hist:
            means.append(0.0)
            stds.append(0.0)
            continue
        vals = popularity[torch.tensor(hist, dtype=torch.long).clamp(0, popularity.numel() - 1)]
        means.append(vals.mean().item())
        stds.append(vals.std(unbiased=False).item() if vals.numel() > 1 else 0.0)
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)

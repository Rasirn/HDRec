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

RESIDUAL_SCALE_SCHEMA_VERSION = 'v4_residual_scale_v1'
RESIDUAL_SCALE_FEATURE_NAMES = [
    'text_logit_mean', 'text_logit_std', 'text_logit_rms', 'text_logit_range',
    'id_logit_mean', 'id_logit_std', 'id_logit_rms', 'id_logit_range',
    'residual_mean', 'residual_std', 'residual_rms', 'residual_max',
    'residual_range', 'residual_l1_mean', 'residual_l2_norm',
    'residual_to_text_std_ratio', 'residual_to_text_rms_ratio',
    'id_to_text_std_ratio', 'residual_top10_mass_ratio',
    'residual_top50_mass_ratio', 'residual_top100_mass_ratio',
    'residual_normalized_entropy', 'residual_effective_support',
    'residual_concentration',
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


def compute_residual_scale_features(logits_text, logits_id, residual, eps=1e-8):
    text = logits_text.float()
    ids = logits_id.float()
    residual = residual.float()

    def stats(values):
        mean = values.mean(dim=-1)
        std = values.std(dim=-1, unbiased=False)
        rms = values.square().mean(dim=-1).sqrt()
        value_range = values.max(dim=-1).values - values.min(dim=-1).values
        return mean, std, rms, value_range

    text_mean, text_std, text_rms, text_range = stats(text)
    id_mean, id_std, id_rms, id_range = stats(ids)
    residual_mean, residual_std, residual_rms, residual_range = stats(residual)
    mass = residual.abs()
    mass_sum = mass.sum(dim=-1).clamp_min(eps)

    def top_mass_ratio(k):
        k = min(k, mass.size(-1))
        return torch.topk(mass, k=k, dim=-1).values.sum(dim=-1) / mass_sum

    probs = mass / mass_sum.unsqueeze(-1)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    normalized_entropy = entropy / max(float(torch.log(torch.tensor(float(mass.size(-1))))), eps)
    effective_support = entropy.exp() / mass.size(-1)
    concentration = probs.square().sum(dim=-1)
    columns = [
        text_mean, text_std, text_rms, text_range,
        id_mean, id_std, id_rms, id_range,
        residual_mean, residual_std, residual_rms, residual.max(dim=-1).values,
        residual_range, mass.mean(dim=-1), residual.norm(dim=-1),
        residual_std / text_std.clamp_min(eps),
        residual_rms / text_rms.clamp_min(eps),
        id_std / text_std.clamp_min(eps),
        top_mass_ratio(10), top_mass_ratio(50), top_mass_ratio(100),
        normalized_entropy, effective_support, concentration,
    ]
    return sanitize_features(torch.stack(columns, dim=-1))


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

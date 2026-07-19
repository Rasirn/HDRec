import random

import torch
import torch.nn.functional as F

from candidate_features import CANDIDATE_FEATURE_NAMES, candidate_feature_chunk, prepare_candidate_context
from candidate_gate import CandidateFeatureNormalizer
from ranking_metrics import target_ranks


def split_gate_users(cache, dev_ratio=0.2, seed=42):
    users = torch.unique(cache['user_ids'].long().cpu(), sorted=True).tolist()
    random.Random(seed).shuffle(users)
    dev_count = min(max(1, int(round(len(users) * dev_ratio))), len(users) - 1)
    dev_users = set(users[:dev_count])
    dev_mask = torch.tensor([int(user) in dev_users for user in cache['user_ids']], dtype=torch.bool)
    train_indices, dev_indices = torch.where(~dev_mask)[0], torch.where(dev_mask)[0]
    return train_indices, dev_indices


def index_batches(indices, batch_size, shuffle=False, seed=42):
    indices = indices.clone().cpu()
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
    for start in range(0, indices.numel(), batch_size):
        yield indices[start:start + batch_size]


def prepare_context(cache, indices, popularity, device):
    return prepare_candidate_context(
        cache['logits_text'][indices].float().to(device),
        cache['logits_id'][indices].float().to(device),
        cache['features'][indices].float().to(device),
        popularity,
        fusion_temperature=float(cache['fusion_temperature']),
    )


@torch.no_grad()
def fit_candidate_normalizer(cache, indices, popularity, batch_size, candidate_chunk, device):
    feature_dim = len(CANDIDATE_FEATURE_NAMES)
    total = torch.zeros(feature_dim, dtype=torch.float64)
    square_total = torch.zeros(feature_dim, dtype=torch.float64)
    count = 0
    for batch_indices in index_batches(indices, batch_size):
        context = prepare_context(cache, batch_indices, popularity, device)
        for start in range(0, context.num_items, candidate_chunk):
            features = candidate_feature_chunk(context, start, start + candidate_chunk)
            flat = features.reshape(-1, features.size(-1)).double().cpu()
            total += flat.sum(dim=0)
            square_total += flat.square().sum(dim=0)
            count += flat.size(0)
    return CandidateFeatureNormalizer().fit_from_moments(total, square_total, count)


def train_epoch(
    cache, indices, popularity, model, normalizer, optimizer, max_alpha,
    lambda_sparse, lambda_safe, batch_size, candidate_chunk, device, seed,
):
    model.train()
    losses = []
    for batch_indices in index_batches(indices, batch_size, shuffle=True, seed=seed):
        context = prepare_context(cache, batch_indices, popularity, device)
        text = cache['logits_text'][batch_indices].float().to(device)
        labels = cache['labels'][batch_indices].long().to(device)
        score_chunks, gates, deviations = [], [], []
        for start in range(0, context.num_items, candidate_chunk):
            end = min(start + candidate_chunk, context.num_items)
            features = normalizer.transform(candidate_feature_chunk(context, start, end))
            gate = model(features)
            delta = float(max_alpha) * gate * context.residual[:, start:end]
            score_chunks.append(text[:, start:end] + delta)
            gates.append(gate)
            deviations.append(delta)
        scores = torch.cat(score_chunks, dim=-1)
        gate_values = torch.cat(gates, dim=-1)
        delta_values = torch.cat(deviations, dim=-1)
        rank_loss = F.cross_entropy(scores, labels)
        loss = rank_loss + float(lambda_sparse) * gate_values.mean() + float(lambda_safe) * delta_values.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate_gate(cache, indices, popularity, model, normalizer, max_alpha, batch_size, candidate_chunk, device):
    model.eval()
    ranks, text_ranks, dcg_values = [], [], []
    gate_sum = gate_square_sum = gate_near_zero = gate_gt_half = gate_count = 0.0
    active_counts = []
    for batch_indices in index_batches(indices, batch_size):
        context = prepare_context(cache, batch_indices, popularity, device)
        text = cache['logits_text'][batch_indices].float().to(device)
        labels = cache['labels'][batch_indices].long().to(device)
        score_chunks, batch_gates = [], []
        for start in range(0, context.num_items, candidate_chunk):
            end = min(start + candidate_chunk, context.num_items)
            features = normalizer.transform(candidate_feature_chunk(context, start, end))
            gate = model(features)
            score_chunks.append(text[:, start:end] + float(max_alpha) * gate * context.residual[:, start:end])
            batch_gates.append(gate)
        scores, gate = torch.cat(score_chunks, dim=-1), torch.cat(batch_gates, dim=-1)
        batch_ranks, batch_text_ranks = target_ranks(scores, labels), target_ranks(text, labels)
        ranks.append(batch_ranks.cpu()); text_ranks.append(batch_text_ranks.cpu())
        dcg_values.append(((batch_ranks < 10).float() / torch.log2(batch_ranks + 2.0)).cpu())
        gate_sum += gate.sum().item(); gate_square_sum += gate.square().sum().item()
        gate_near_zero += (gate <= 1e-3).sum().item(); gate_gt_half += (gate > 0.5).sum().item()
        gate_count += gate.numel(); active_counts.append((gate > 1e-3).float().sum(dim=-1).cpu())
    ranks, text_ranks, dcg = torch.cat(ranks), torch.cat(text_ranks), torch.cat(dcg_values)
    active = torch.cat(active_counts)
    mean = gate_sum / gate_count
    metrics = {
        'NDCG@5': (((ranks < 5).float() / torch.log2(ranks + 2.0)).mean().item()),
        'NDCG@10': dcg.mean().item(),
        'Recall@5': (ranks < 5).float().mean().item(), 'Recall@10': (ranks < 10).float().mean().item(),
        'MRR': (1.0 / (ranks + 1.0)).mean().item(),
        'harm_rate': (ranks > text_ranks).float().mean().item(),
        'benefit_rate': (ranks < text_ranks).float().mean().item(),
        'average_target_rank_gain': (text_ranks - ranks).mean().item(),
        'gate_mean': mean,
        'gate_std': max(0.0, gate_square_sum / gate_count - mean * mean) ** 0.5,
        'gate_near_zero_rate': gate_near_zero / gate_count,
        'gate_gt_half_rate': gate_gt_half / gate_count,
        'active_candidates_mean': active.mean().item(),
        'active_candidate_ratio': (active / cache['logits_text'].size(1)).mean().item(),
    }
    return metrics, dcg, ranks, text_ranks

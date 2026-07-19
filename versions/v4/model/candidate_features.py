from dataclasses import dataclass

import torch
import torch.nn.functional as F

from reliability_features import FEATURE_NAMES, sanitize_features
from utility_label import id_confidence_residual


CANDIDATE_FEATURE_SCHEMA_VERSION = 'v4_candidate_reliability_v1'
CANDIDATE_LOCAL_FEATURE_NAMES = [
    'text_logit_zscore', 'id_logit_zscore',
    'text_probability', 'id_probability',
    'text_log_probability', 'id_log_probability',
    'raw_id_residual', 'normalized_id_residual',
    'text_rank_percentile', 'id_rank_percentile',
    'rank_difference', 'absolute_rank_difference',
    'both_top10', 'both_top50',
    'text_top10_id_not', 'id_top10_text_not',
    'score_product', 'probability_product',
]
CANDIDATE_POPULARITY_FEATURE_NAMES = ['log_item_popularity', 'normalized_item_popularity']
CANDIDATE_FEATURE_NAMES = CANDIDATE_LOCAL_FEATURE_NAMES + FEATURE_NAMES + CANDIDATE_POPULARITY_FEATURE_NAMES


def rank_positions(logits):
    order = torch.argsort(logits.float(), dim=-1, descending=True)
    ranks = torch.empty_like(order)
    source = torch.arange(logits.size(-1), device=logits.device).expand_as(order)
    ranks.scatter_(1, order, source)
    return ranks


@dataclass
class CandidateFeatureContext:
    text_z: torch.Tensor
    id_z: torch.Tensor
    text_prob: torch.Tensor
    id_prob: torch.Tensor
    text_log_prob: torch.Tensor
    id_log_prob: torch.Tensor
    residual: torch.Tensor
    normalized_residual: torch.Tensor
    text_rank: torch.Tensor
    id_rank: torch.Tensor
    sequence_features: torch.Tensor
    log_popularity: torch.Tensor
    normalized_popularity: torch.Tensor

    @property
    def num_items(self):
        return self.text_z.size(-1)


def prepare_candidate_context(
    logits_text,
    logits_id,
    sequence_features,
    item_popularity,
    fusion_temperature=1.0,
    eps=1e-8,
):
    if logits_text.shape != logits_id.shape:
        raise ValueError('Text and ID logits must have identical shapes.')
    if sequence_features.size(0) != logits_text.size(0):
        raise ValueError('Sequence feature batch size does not match logits.')
    if sequence_features.size(-1) != len(FEATURE_NAMES):
        raise ValueError(f'Expected {len(FEATURE_NAMES)} sequence features.')
    if item_popularity.numel() != logits_text.size(-1):
        raise ValueError('Item popularity length does not match candidate count.')

    text, ids = logits_text.float(), logits_id.float()

    def zscore(values):
        return (values - values.mean(dim=-1, keepdim=True)) / values.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)

    residual = id_confidence_residual(ids, temperature=fusion_temperature).float()
    residual_rms = residual.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    popularity = item_popularity.to(device=text.device, dtype=torch.float32).clamp_min(0)
    return CandidateFeatureContext(
        text_z=zscore(text),
        id_z=zscore(ids),
        text_prob=F.softmax(text, dim=-1),
        id_prob=F.softmax(ids, dim=-1),
        text_log_prob=F.log_softmax(text, dim=-1),
        id_log_prob=F.log_softmax(ids, dim=-1),
        residual=residual,
        normalized_residual=residual / residual_rms,
        text_rank=rank_positions(text),
        id_rank=rank_positions(ids),
        sequence_features=sequence_features.float().to(text.device),
        log_popularity=torch.log1p(popularity),
        normalized_popularity=popularity / popularity.max().clamp_min(eps),
    )


def candidate_feature_chunk(context, start=0, end=None):
    end = context.num_items if end is None else min(int(end), context.num_items)
    start = max(0, int(start))
    if start >= end:
        raise ValueError('Candidate chunk must be non-empty.')
    sl = slice(start, end)
    denominator = max(1, context.num_items - 1)
    text_rank = context.text_rank[:, sl].float() / denominator
    id_rank = context.id_rank[:, sl].float() / denominator
    text_top10, id_top10 = text_rank * denominator < 10, id_rank * denominator < 10
    text_top50, id_top50 = text_rank * denominator < 50, id_rank * denominator < 50
    sequence = context.sequence_features.unsqueeze(1).expand(-1, end - start, -1)
    popularity = torch.stack([
        context.log_popularity[sl], context.normalized_popularity[sl],
    ], dim=-1).unsqueeze(0).expand(context.text_z.size(0), -1, -1)
    local = torch.stack([
        context.text_z[:, sl], context.id_z[:, sl],
        context.text_prob[:, sl], context.id_prob[:, sl],
        context.text_log_prob[:, sl], context.id_log_prob[:, sl],
        context.residual[:, sl], context.normalized_residual[:, sl],
        text_rank, id_rank, text_rank - id_rank, (text_rank - id_rank).abs(),
        (text_top10 & id_top10).float(), (text_top50 & id_top50).float(),
        (text_top10 & ~id_top10).float(), (id_top10 & ~text_top10).float(),
        context.text_z[:, sl] * context.id_z[:, sl],
        context.text_prob[:, sl] * context.id_prob[:, sl],
    ], dim=-1)
    return sanitize_features(torch.cat([local, sequence, popularity], dim=-1))

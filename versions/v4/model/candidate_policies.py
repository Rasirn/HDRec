import itertools

import torch


POLICY_ALPHA_GRID = (0.1, 0.25, 0.5, 0.75, 1.0)


def candidate_policy_configs():
    configs = []
    configs.extend({'policy': 'P2_id_topk', 'k': k} for k in (10, 20, 50, 100, 200))
    configs.extend(
        {'policy': 'P3_intersection', 'text_k': text_k, 'id_k': id_k}
        for text_k, id_k in itertools.product((10, 50, 100), repeat=2)
    )
    configs.extend({'policy': 'P4_union', 'k': k} for k in (10, 50, 100))
    configs.extend({'policy': 'P5_rank_agreement', 'distance': d} for d in (0.01, 0.05, 0.10))
    configs.append({'policy': 'P6_soft_consensus'})
    return configs


def policy_key(config):
    policy = config['policy']
    if policy in {'P2_id_topk', 'P4_union'}:
        return f"{policy}_k{config['k']}"
    if policy == 'P3_intersection':
        return f"{policy}_t{config['text_k']}_i{config['id_k']}"
    if policy == 'P5_rank_agreement':
        return f"{policy}_d{config['distance']:.2f}"
    return policy


def policy_complexity(config):
    order = {
        'P2_id_topk': 2,
        'P3_intersection': 3,
        'P4_union': 4,
        'P5_rank_agreement': 5,
        'P6_soft_consensus': 6,
    }
    return order[config['policy']]


def policy_gate(config, text_rank, id_rank, text_probability=None, id_probability=None):
    policy = config['policy']
    if policy == 'P2_id_topk':
        return (id_rank < int(config['k'])).float()
    if policy == 'P3_intersection':
        return ((text_rank < int(config['text_k'])) & (id_rank < int(config['id_k']))).float()
    if policy == 'P4_union':
        return ((text_rank < int(config['k'])) | (id_rank < int(config['k']))).float()
    if policy == 'P5_rank_agreement':
        denominator = max(1, text_rank.size(-1) - 1)
        return ((text_rank.float() - id_rank.float()).abs() / denominator <= float(config['distance'])).float()
    if policy == 'P6_soft_consensus':
        if text_probability is None or id_probability is None:
            raise ValueError('Soft consensus requires both branch probabilities.')
        gate = (text_probability * id_probability).clamp_min(0).sqrt()
        return gate / gate.max(dim=-1, keepdim=True).values.clamp_min(1e-12)
    raise ValueError(f'Unknown candidate policy: {policy}')

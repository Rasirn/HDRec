import torch

from utility_label import id_confidence_residual


TRANSFORM_ORDER = {'R0': 0, 'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5, 'R6': 6}


def calibrate_residual(logits_id, transform='R0', fusion_temperature=1.0, topk=None, eps=1e-8):
    ids = logits_id.float()
    if transform == 'R4':
        return (ids - ids.mean(dim=-1, keepdim=True)) / ids.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(eps)
    if transform == 'R5':
        probabilities = torch.softmax(ids / float(fusion_temperature), dim=-1)
        return probabilities * ids.size(-1) - 1.0

    residual = id_confidence_residual(ids, temperature=fusion_temperature)
    if transform == 'R0':
        return residual
    if transform == 'R1':
        scale = residual.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        return residual / scale
    if transform == 'R2':
        scale = residual.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return residual / scale
    if transform == 'R3':
        scale = residual.abs().max(dim=-1, keepdim=True).values.clamp_min(eps)
        return residual / scale
    if transform == 'R6':
        if topk is None or topk <= 0:
            raise ValueError('R6 requires a positive topk.')
        k = min(int(topk), residual.size(-1))
        indices = torch.topk(residual, k=k, dim=-1).indices
        sparse = torch.zeros_like(residual).scatter(-1, indices, residual.gather(-1, indices))
        scale = sparse.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        return sparse / scale
    raise ValueError(f'Unknown residual transform: {transform}')


def residual_config_grid():
    configs = []
    temperatures = (0.5, 1.0, 2.0)
    for transform in ('R0', 'R1', 'R2', 'R3', 'R5'):
        for temperature in temperatures:
            configs.append({'transform': transform, 'temperature': temperature, 'topk': None})
    configs.append({'transform': 'R4', 'temperature': None, 'topk': None})
    for temperature in temperatures:
        for topk in (20, 50, 100):
            configs.append({'transform': 'R6', 'temperature': temperature, 'topk': topk})
    return configs


def config_complexity(config):
    return (TRANSFORM_ORDER[config['transform']], config.get('topk') or 0, config.get('temperature') or 0.0)

import torch

from utility_label import id_confidence_residual


def candidate_fusion(
    logits_text,
    logits_id,
    candidate_gate,
    fusion_temperature=1.0,
    max_alpha=0.5,
):
    residual = id_confidence_residual(logits_id, temperature=fusion_temperature)
    gate = candidate_gate.to(device=logits_text.device, dtype=logits_text.dtype).clamp(0.0, 1.0)
    if gate.shape != logits_text.shape:
        raise ValueError(f'candidate_gate shape {gate.shape} must match logits shape {logits_text.shape}.')
    return logits_text + float(max_alpha) * gate * residual.to(logits_text.dtype)

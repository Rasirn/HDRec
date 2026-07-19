import torch
import torch.nn.functional as F


def id_confidence_residual(logits_id, temperature=1.0):
    """Residual used by v1 confidence_fusion when fusion_type='text'."""
    with torch.no_grad():
        logits_id_avg = logits_id.mean(dim=-1, keepdim=True)
        logits_id_truth = torch.sigmoid((logits_id - logits_id_avg) / temperature)
        logits_id_min = logits_id.min(dim=-1, keepdim=True).values
    shifted_id = logits_id - logits_id_min + 1e-8
    return shifted_id * logits_id_truth


def fixed_text_fusion(logits_text, logits_id, alpha=0.5, temperature=1.0):
    return logits_text + alpha * id_confidence_residual(logits_id, temperature=temperature)


def cross_entropy_per_sample(logits, labels):
    return F.cross_entropy(logits.float(), labels.long(), reduction='none')


def utility_values(logits_text, logits_id, labels, temperature=1.0):
    residual = id_confidence_residual(logits_id, temperature=temperature)
    text_loss = cross_entropy_per_sample(logits_text, labels)
    residual_loss = cross_entropy_per_sample(logits_text + residual, labels)
    return text_loss - residual_loss


def utility_labels(logits_text, logits_id, labels, temperature=1.0):
    return (utility_values(logits_text, logits_id, labels, temperature=temperature) > 0).long()

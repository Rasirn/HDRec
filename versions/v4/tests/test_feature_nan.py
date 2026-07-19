import torch

from reliability_features import compute_sequence_features


def test_feature_extractor_sanitizes_nan_and_inf():
    text = torch.tensor([[float('nan'), 1.0, -1.0], [float('inf'), 0.0, float('-inf')]])
    ids = torch.tensor([[0.0, float('inf'), 1.0], [float('nan'), 2.0, -2.0]])

    features = compute_sequence_features(text, ids, topk=2)

    assert torch.isfinite(features).all()

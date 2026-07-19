import torch

from reliability_fusion import ReliabilityFuser
from utility_label import cross_entropy_per_sample, fixed_text_fusion, id_confidence_residual, utility_values


def test_dynamic_false_matches_v1_fixed_fusion():
    torch.manual_seed(5)
    text = torch.randn(7, 19)
    ids = torch.randn(7, 19)
    features = torch.randn(7, 10)
    alpha0 = 0.5
    temperature = 1.0
    residual = id_confidence_residual(ids, temperature=temperature)
    fuser = ReliabilityFuser(10, hidden_dim=4, dropout=0.0, alpha0=alpha0, alpha_max=1.0, rho=0.5)

    final, alpha = fuser(text, residual, features, dynamic_fusion=False)
    expected = fixed_text_fusion(text, ids, alpha=alpha0, temperature=temperature)

    assert torch.equal(alpha, torch.full_like(alpha, alpha0))
    assert torch.equal(final, expected)

    expected_utility = cross_entropy_per_sample(text, torch.arange(7) % 19) - cross_entropy_per_sample(
        expected, torch.arange(7) % 19
    )
    actual_utility = utility_values(
        text, ids, torch.arange(7) % 19, alpha0=alpha0, temperature=temperature
    )
    assert torch.equal(actual_utility, expected_utility)

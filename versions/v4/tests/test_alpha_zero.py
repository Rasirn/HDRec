import torch

from reliability_fusion import ReliabilityFuser


def test_alpha_zero_is_exact_text_only():
    torch.manual_seed(3)
    text = torch.randn(5, 13)
    residual = torch.randn(5, 13)
    features = torch.randn(5, 10)
    fuser = ReliabilityFuser(10, hidden_dim=4, dropout=0.0, alpha0=0.0, alpha_max=1.0, rho=0.0)

    final, alpha = fuser(text, residual, features)

    assert torch.equal(alpha, torch.zeros_like(alpha))
    assert torch.equal(final, text)

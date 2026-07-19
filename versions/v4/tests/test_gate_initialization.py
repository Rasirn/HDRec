import torch
import torch.nn.functional as F

from reliability_fusion import ReliabilityFuser


def test_gate_initialization_starts_at_alpha0():
    torch.manual_seed(23)
    fuser = ReliabilityFuser(10, hidden_dim=16, dropout=0.1, alpha0=0.5, alpha_max=1.0, rho=0.5)
    fuser.eval()
    features = torch.randn(31, 10)

    gate_logit = fuser.gate_logit(features)
    alpha = fuser.alpha(features)

    assert torch.equal(gate_logit, torch.zeros_like(gate_logit))
    assert torch.allclose(alpha, torch.full_like(alpha, fuser.alpha0), atol=1e-7)


def test_zero_initialized_gate_still_receives_utility_gradient():
    torch.manual_seed(29)
    fuser = ReliabilityFuser(10, hidden_dim=16, dropout=0.0, alpha0=0.5, alpha_max=1.0, rho=0.5)
    features = torch.randn(32, 10)
    labels = torch.arange(32).remainder(2).float()

    loss = F.binary_cross_entropy_with_logits(fuser.gate_logit(features), labels)
    loss.backward()

    final_linear = fuser.gate.net[-1]
    assert final_linear.weight.grad is not None
    assert final_linear.bias.grad is not None
    assert final_linear.weight.grad.abs().sum().item() > 0.0

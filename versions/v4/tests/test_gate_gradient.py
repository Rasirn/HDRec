import torch
import torch.nn.functional as F

from reliability_fusion import ReliabilityFuser


def test_utility_loss_backpropagates_to_alpha_gate():
    torch.manual_seed(11)
    fuser = ReliabilityFuser(10, hidden_dim=8, dropout=0.0, alpha0=0.5, alpha_max=1.0, rho=0.5)
    features = torch.randn(16, 10)
    utility_label = torch.randint(0, 2, (16,)).float()

    gate_logit = fuser.gate_logit(features)
    utility_loss = F.binary_cross_entropy_with_logits(gate_logit, utility_label)
    utility_loss.backward()

    grads = [parameter.grad for parameter in fuser.gate.parameters()]
    assert all(grad is not None for grad in grads)
    assert sum(grad.abs().sum().item() for grad in grads) > 0.0

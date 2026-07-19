import torch
import torch.nn as nn


class FeatureStandardizer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, features):
        features = features.float()
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def transform(self, features):
        if self.mean is None or self.std is None:
            raise RuntimeError('FeatureStandardizer must be fitted before transform.')
        mean = self.mean.to(features.device)
        std = self.std.to(features.device)
        return (features.float() - mean) / std

    def fit_transform(self, features):
        return self.fit(features).transform(features)

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'eps': self.eps}

    def load_state_dict(self, state):
        self.mean = state['mean'].float()
        self.std = state['std'].float()
        self.eps = float(state.get('eps', 1e-6))


class ContextGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, features):
        return self.net(features.float()).squeeze(-1)


class ReliabilityFuser(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.1, alpha0=0.5, alpha_max=1.0, rho=0.5):
        super().__init__()
        self.gate = ContextGate(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.alpha0 = float(alpha0)
        self.alpha_max = float(alpha_max)
        self.rho = float(rho)

    def alpha(self, features):
        delta = self.gate(features)
        return torch.clamp(self.alpha0 + self.rho * delta, min=0.0, max=self.alpha_max)

    def forward(self, logits_text, id_residual, features):
        alpha = self.alpha(features)
        final_logits = logits_text + alpha.unsqueeze(-1) * id_residual
        return final_logits, alpha


def save_fuser(path, fuser, standardizer, feature_names, extra=None):
    payload = {
        'model_state': fuser.state_dict(),
        'standardizer': standardizer.state_dict(),
        'feature_names': list(feature_names),
        'input_dim': len(feature_names),
        'alpha0': fuser.alpha0,
        'alpha_max': fuser.alpha_max,
        'rho': fuser.rho,
        'extra': extra or {},
    }
    torch.save(payload, path)


def load_fuser(path, hidden_dim=32, dropout=0.0, map_location='cpu'):
    payload = torch.load(path, map_location=map_location)
    fuser = ReliabilityFuser(
        input_dim=payload['input_dim'],
        hidden_dim=hidden_dim,
        dropout=dropout,
        alpha0=payload['alpha0'],
        alpha_max=payload['alpha_max'],
        rho=payload['rho'],
    )
    fuser.load_state_dict(payload['model_state'])
    standardizer = FeatureStandardizer()
    standardizer.load_state_dict(payload['standardizer'])
    return fuser, standardizer, payload

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
        )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, features):
        return self.net(features.float()).squeeze(-1)


class ReliabilityFuser(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.1, alpha0=0.5, alpha_max=1.0, rho=0.5):
        super().__init__()
        self.gate = ContextGate(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.alpha0 = float(alpha0)
        self.alpha_max = float(alpha_max)
        self.rho = float(rho)

    def gate_logit(self, features):
        return self.gate(features)

    def utility_probability(self, features):
        return torch.sigmoid(self.gate_logit(features))

    def alpha(self, features):
        utility_prob = self.utility_probability(features)
        delta = 2.0 * utility_prob - 1.0
        return torch.clamp(self.alpha0 + self.rho * delta, min=0.0, max=self.alpha_max)

    def forward(self, logits_text, id_residual, features, dynamic_fusion=True, return_gate_logit=False):
        if dynamic_fusion:
            gate_logit = self.gate_logit(features)
            utility_prob = torch.sigmoid(gate_logit)
            alpha = torch.clamp(
                self.alpha0 + self.rho * (2.0 * utility_prob - 1.0),
                min=0.0,
                max=self.alpha_max,
            )
        else:
            gate_logit = None
            alpha = logits_text.new_full((logits_text.size(0),), self.alpha0)
        final_logits = logits_text + alpha.unsqueeze(-1) * id_residual
        if return_gate_logit:
            return final_logits, alpha, gate_logit
        return final_logits, alpha


def save_fuser(path, fuser, standardizer, feature_names, dataset, seed, extra=None):
    payload = {
        'model_state_dict': fuser.state_dict(),
        'normalization_state': standardizer.state_dict(),
        'feature_names': list(feature_names),
        'input_dim': fuser.gate.input_dim,
        'hidden_dim': fuser.gate.hidden_dim,
        'dropout': fuser.gate.dropout,
        'alpha0': fuser.alpha0,
        'alpha_max': fuser.alpha_max,
        'rho': fuser.rho,
        'dataset': dataset,
        'seed': int(seed),
        'extra': extra or {},
    }
    torch.save(payload, path)


def load_fuser(path, map_location='cpu'):
    payload = torch.load(path, map_location=map_location)
    required = {
        'model_state_dict', 'input_dim', 'hidden_dim', 'dropout', 'alpha0',
        'alpha_max', 'rho', 'feature_names', 'normalization_state', 'dataset', 'seed',
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f'Invalid fuser checkpoint {path}; missing fields: {missing}')
    if payload['input_dim'] != len(payload['feature_names']):
        raise ValueError('Fuser checkpoint input_dim does not match feature_names.')
    fuser = ReliabilityFuser(
        input_dim=payload['input_dim'],
        hidden_dim=payload['hidden_dim'],
        dropout=payload['dropout'],
        alpha0=payload['alpha0'],
        alpha_max=payload['alpha_max'],
        rho=payload['rho'],
    )
    incompatible = fuser.load_state_dict(payload['model_state_dict'], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f'Fuser state mismatch; missing={incompatible.missing_keys}, '
            f'unexpected={incompatible.unexpected_keys}'
        )
    standardizer = FeatureStandardizer()
    standardizer.load_state_dict(payload['normalization_state'])
    return fuser, standardizer, payload

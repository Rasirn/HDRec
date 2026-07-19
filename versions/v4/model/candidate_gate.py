import math

import torch
import torch.nn as nn


class CandidateFeatureNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_from_moments(self, total, square_total, count):
        if count <= 0:
            raise ValueError('Cannot fit candidate normalizer without observations.')
        self.mean = (total / count).float()
        variance = square_total / count - self.mean.double().square()
        self.std = variance.clamp_min(1e-12).sqrt().float().clamp_min(1e-6)
        return self

    def transform(self, features):
        if self.mean is None or self.std is None:
            raise RuntimeError('Candidate normalizer has not been fitted.')
        return (features - self.mean.to(features.device)) / self.std.to(features.device)

    def state_dict(self):
        return {'mean': self.mean.cpu(), 'std': self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['mean'].float(), state['std'].float())


class CandidateGate(nn.Module):
    def __init__(self, input_dim, architecture='linear', hidden_dim=16, dropout=0.1, initial_probability=0.01):
        super().__init__()
        if architecture == 'linear':
            self.network = nn.Linear(input_dim, 1)
            output = self.network
        elif architecture == 'tiny_mlp':
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            output = self.network[-1]
        else:
            raise ValueError(f'Unknown candidate gate architecture: {architecture}')
        self.input_dim = int(input_dim)
        self.architecture = architecture
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.initial_probability = float(initial_probability)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, math.log(initial_probability / (1.0 - initial_probability)))

    def forward(self, features):
        return torch.sigmoid(self.network(features.float()).squeeze(-1))

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def candidate_gate_checkpoint(model, normalizer, metadata):
    return {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'architecture': model.architecture,
        'hidden_dim': model.hidden_dim,
        'dropout': model.dropout,
        'initial_probability': model.initial_probability,
        'normalization_state': normalizer.state_dict(),
        **metadata,
    }


def load_candidate_gate_checkpoint(path, map_location='cpu'):
    payload = torch.load(path, map_location=map_location)
    required = {
        'model_state_dict', 'input_dim', 'architecture', 'hidden_dim', 'dropout',
        'initial_probability', 'normalization_state', 'feature_schema', 'feature_names',
        'dataset', 'seed', 'max_alpha',
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f'Candidate gate checkpoint missing fields: {missing}')
    model = CandidateGate(
        payload['input_dim'], payload['architecture'], payload['hidden_dim'],
        payload['dropout'], payload['initial_probability'],
    )
    model.load_state_dict(payload['model_state_dict'], strict=True)
    normalizer = CandidateFeatureNormalizer.from_state_dict(payload['normalization_state'])
    return model, normalizer, payload

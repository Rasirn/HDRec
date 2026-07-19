from pathlib import Path

import torch

from reliability_features import FEATURE_NAMES
from reliability_fusion import FeatureStandardizer, ReliabilityFuser, load_fuser, save_fuser


def test_checkpoint_restores_structure_schema_and_parameters(tmp_path: Path):
    torch.manual_seed(17)
    features = torch.randn(12, len(FEATURE_NAMES))
    scaler = FeatureStandardizer().fit(features)
    fuser = ReliabilityFuser(
        len(FEATURE_NAMES), hidden_dim=7, dropout=0.25, alpha0=0.5, alpha_max=0.9, rho=0.3
    )
    checkpoint = tmp_path / 'fuser.pt'
    save_fuser(
        checkpoint,
        fuser,
        scaler,
        FEATURE_NAMES,
        dataset='Industrial_and_Scientific',
        seed=17,
    )

    loaded, loaded_scaler, payload = load_fuser(checkpoint)

    assert payload['input_dim'] == len(FEATURE_NAMES)
    assert payload['hidden_dim'] == 7
    assert payload['dropout'] == 0.25
    assert payload['feature_names'] == FEATURE_NAMES
    assert payload['dataset'] == 'Industrial_and_Scientific'
    assert payload['seed'] == 17
    for key, value in fuser.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    assert torch.equal(scaler.mean, loaded_scaler.mean)
    assert torch.equal(scaler.std, loaded_scaler.std)

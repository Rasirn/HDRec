import tempfile
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'model'
SCRIPTS_DIR = ROOT / 'scripts'
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from reliability_features import FEATURE_NAMES, compute_sequence_features
from reliability_fusion import FeatureStandardizer, ReliabilityFuser, load_fuser, save_fuser
from utility_label import fixed_text_fusion, id_confidence_residual, utility_labels
from ranking_metrics import ranking_metrics
from analyze_oracle import oracle_alpha_scores, oracle_select_scores


def main():
    torch.manual_seed(7)
    n, c = 8, 17
    text = torch.randn(n, c)
    ids = torch.randn(n, c)
    labels = torch.randint(0, c, (n,))
    hist_len = torch.arange(1, n + 1).float()
    pop_mean = torch.linspace(0, 1, n)
    pop_std = torch.linspace(1, 0, n)

    features = compute_sequence_features(text, ids, hist_len, pop_mean, pop_std, topk=5)
    assert torch.isfinite(features).all(), 'features contain NaN/Inf'
    assert features.shape == (n, len(FEATURE_NAMES))

    residual = id_confidence_residual(ids, temperature=1.0)
    fuser = ReliabilityFuser(input_dim=features.size(1), hidden_dim=4, dropout=0.0, alpha0=0.0, alpha_max=1.0, rho=0.0)
    final, alpha = fuser(text, residual, features)
    assert torch.equal(final, text), 'alpha=0 path must exactly equal text logits'
    assert torch.equal(alpha, torch.zeros_like(alpha)), 'alpha must be exactly zero'

    fixed0 = fixed_text_fusion(text, ids, alpha=0.0, temperature=1.0)
    assert torch.equal(fixed0, text), 'fixed alpha=0 must exactly equal text logits'

    fixed = fixed_text_fusion(text, ids, alpha=0.5, temperature=1.0)
    select_scores, use_fixed = oracle_select_scores(text, fixed, labels)
    alpha_scores, best_alpha = oracle_alpha_scores(text, ids, labels, [0.0, 0.5, 1.0], 1.0)
    for scores in [text, ids, fixed, select_scores, alpha_scores]:
        metrics = ranking_metrics(scores, labels)
        assert 'NDCG@10' in metrics
    assert use_fixed.shape == (n,)
    assert best_alpha.shape == (n,)

    labels_u = utility_labels(text, ids, labels)
    assert labels_u.shape == (n,)

    scaler = FeatureStandardizer().fit(features)
    scaled = scaler.transform(features)
    assert torch.isfinite(scaled).all()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'fuser.pt'
        save_fuser(
            path,
            fuser,
            scaler,
            FEATURE_NAMES,
            dataset='synthetic',
            seed=7,
            feature_schema='smoke',
            base_checkpoint_path='/tmp/base.bin',
            fusion_temperature=1.0,
            utility_target='ce_at_alpha0',
            extra={'smoke': True},
        )
        loaded, loaded_scaler, payload = load_fuser(path)
        loaded_final, loaded_alpha = loaded(text, residual, loaded_scaler.transform(features))
        assert torch.equal(loaded_final, text)
        assert payload['extra']['smoke'] is True

        cache_path = Path(tmp) / 'cache.pt'
        payload_cache = {
            'version': 'smoke',
            'feature_schema': 'smoke',
            'feature_names': FEATURE_NAMES,
            'dataset': 'synthetic',
            'split': 'valid',
            'alpha0': 0.5,
            'fusion_temperature': 1.0,
            'checkpoint_path': '/tmp/base.bin',
            'utility_target': 'ce_at_alpha0',
            'logits_text': text,
            'logits_id': ids,
            'labels': labels,
            'features': features,
            'utility_label': labels_u,
            'utility_ce_label': labels_u,
            'utility_rank_label': labels_u,
        }
        torch.save(payload_cache, cache_path)
        loaded_cache = torch.load(cache_path, map_location='cpu')
        assert torch.equal(loaded_cache['logits_text'], text)
        assert torch.equal(loaded_cache['labels'], labels)

    print('v4 smoke test passed')


if __name__ == '__main__':
    main()

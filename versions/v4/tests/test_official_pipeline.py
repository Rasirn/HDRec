import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'versions' / 'v4' / 'scripts'))
from provenance import validate_provenance


def test_v4_run_entry_cannot_reach_legacy_fixed_fusion():
    content = (ROOT / 'versions' / 'v4' / 'run.sh').read_text()
    assert 'run_v4_candidate.sh' in content
    assert 'run_common.sh' not in content
    assert 'model/main.py' not in content


def test_legacy_main_requires_explicit_opt_in():
    content = (ROOT / 'versions' / 'v4' / 'model' / 'main.py').read_text()
    assert 'allow_legacy_fixed_fusion' in content
    assert 'does not run v4 Candidate Gate' in content


def test_checkpoint_provenance_mismatch_fails():
    expected = {
        'v1_checkpoint_path': '/tmp/a.bin', 'v1_checkpoint_sha256': 'a',
        'v1_dataset': 'Prime_Pantry', 'v1_profile': 'src_original',
    }
    actual = dict(expected, v1_checkpoint_sha256='b')
    with pytest.raises(ValueError, match='v1 provenance mismatch'):
        validate_provenance(expected, actual)


def test_official_evaluator_declares_candidate_identity():
    content = (ROOT / 'versions' / 'v4' / 'scripts' / 'evaluate_candidate_gate_test.py').read_text()
    assert "'method': 'v4_candidate_gate'" in content
    assert "'uses_candidate_gate': True" in content
    assert '==Test set (v4 Candidate Gate)==' in content

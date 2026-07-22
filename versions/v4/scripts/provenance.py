"""Checkpoint provenance shared by the official v4 Candidate Gate pipeline."""

import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_provenance(checkpoint_path, dataset):
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'v1 checkpoint not found: {checkpoint}')
    manifest_path = checkpoint.parent / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and manifest.get('dataset') not in (None, dataset):
        raise ValueError(f'v1 manifest dataset mismatch: {manifest.get("dataset")} != {dataset}')
    return {
        'v1_checkpoint_path': str(checkpoint),
        'v1_checkpoint_sha256': sha256(checkpoint),
        'v1_checkpoint_size': checkpoint.stat().st_size,
        'v1_dataset': dataset,
        # Existing checkpoints predate manifests. Preserve that fact instead of
        # inventing a profile; new v1 runs always provide src_original/legacy_tuned.
        'v1_profile': (manifest or {}).get('profile', 'legacy_unknown'),
        'v1_run_manifest_path': str(manifest_path.resolve()) if manifest_path.exists() else None,
        'v1_run_manifest_sha256': sha256(manifest_path) if manifest_path.exists() else None,
    }


def validate_provenance(expected, actual):
    fields = ('v1_checkpoint_path', 'v1_checkpoint_sha256', 'v1_dataset', 'v1_profile')
    for field in fields:
        if expected.get(field) != actual.get(field):
            raise ValueError(f'v1 provenance mismatch for {field}: {expected.get(field)!r} != {actual.get(field)!r}')

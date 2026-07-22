"""Run provenance for reproducible v1 baselines."""

import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch


def _sha256(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _command(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions():
    names = ('torch', 'transformers', 'peft', 'accelerate')
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _gpu_info():
    value = _command(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    return [] if not value else [line.strip() for line in value.splitlines()]


def manifest_path(args):
    return Path(args.run_manifest_path or Path(args.output_path) / 'run_manifest.json')


def write_initial_manifest(args):
    data_root = Path(args.data_root)
    data_dir = data_root / args.dataset
    sasrec = Path('./temp/SASRec') / f'{args.dataset}-{args.preference_dim}.pth'
    payload = {
        'method': 'v1', 'profile': args.profile,
        'entrypoint': str(Path(sys.argv[0]).resolve()), 'command': list(sys.argv),
        'dataset': args.dataset, 'seed': args.seed, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'git_commit': _command(['git', 'rev-parse', 'HEAD']),
        'git_dirty': bool(_command(['git', 'status', '--porcelain'])),
        'python': sys.version, 'package_versions': _package_versions(), 'gpu_models': _gpu_info(),
        'data_root': str(data_root.resolve()), 'base_llm_path': args.model_name_or_path,
        'sasrec_checkpoint_path': str(sasrec.resolve()),
        'final_args': {key: value for key, value in vars(args).items() if key not in {'logger', 'device'}},
        'sha256': {
            'train': _sha256(data_dir / 'train.json'), 'valid': _sha256(data_dir / 'val.json'),
            'test': _sha256(data_dir / 'test.json'), 'smap': _sha256(data_dir / 'smap.json'),
            'sasrec': _sha256(sasrec), 'final_checkpoint': None,
        },
        'checkpoint_selection_epoch': None, 'best_validation_metric': None, 'test_metrics': None,
    }
    path = manifest_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')
    return path


def update_final_manifest(args, best_epoch, best_metric, test_metrics):
    path = manifest_path(args)
    payload = json.loads(path.read_text()) if path.exists() else {}
    checkpoint = Path(args.output_path) / 'pytorch_model.bin'
    payload.update({
        'checkpoint_selection_epoch': best_epoch,
        'best_validation_metric': best_metric,
        'test_metrics': test_metrics,
    })
    payload.setdefault('sha256', {})['final_checkpoint'] = _sha256(checkpoint)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')

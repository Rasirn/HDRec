#!/usr/bin/env python3
"""Launch v1 with an explicit, reproducible profile."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from profile_config import profile_values


ROOT = Path(__file__).resolve().parents[3]
V1_DIR = ROOT / 'versions' / 'v1'


def parse_args():
    parser = argparse.ArgumentParser(description='Run HDRec v1 with an explicit profile.')
    parser.add_argument('--dataset', default='Industrial_and_Scientific')
    parser.add_argument('--profile', default=os.environ.get('V1_PROFILE', 'src_original'), choices=['src_original', 'legacy_tuned'])
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--gpu', default=os.environ.get('GPU_IDS', '0'))
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--output-dir', default=str(V1_DIR / 'outputs'))
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_command(args):
    values = profile_values(args.profile, args.dataset)
    suffix = args.suffix or args.profile
    command = [
        sys.executable, str(V1_DIR / 'model' / 'main.py'),
        '--dataset', args.dataset, '--profile', args.profile,
        '--model_name_or_path', args.model, '--suffix', suffix,
        '--data_root', args.data_root, '--output_dir', args.output_dir,
    ]
    for key, value in values.items():
        command.extend([f'--{key}', str(value)])
    command.extend(['--fix_backbone', '--fix_emb', '--use_small_model', '--use_gate', '--use_lora'])
    if args.gradient_checkpointing:
        command.append('--gradient_checkpointing_enable')
    return command


def main():
    args = parse_args()
    command = build_command(args)
    print(f'v1 profile: {args.profile}', flush=True)
    print('resolved command:', ' '.join(command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()

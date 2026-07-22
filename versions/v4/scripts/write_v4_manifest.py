#!/usr/bin/env python3
"""Write a small provenance manifest for one official v4 Candidate Gate run."""

import argparse
import json
from pathlib import Path

from provenance import checkpoint_provenance, git_commit, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--v1_checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate_checkpoint')
    parser.add_argument('--metrics_json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    payload = {
        'method': 'v4_candidate_gate', 'uses_candidate_gate': True,
        'dataset': args.dataset, 'git_commit': git_commit(root),
        'v1_provenance': checkpoint_provenance(args.v1_checkpoint, args.dataset),
    }
    if args.candidate_checkpoint:
        path = Path(args.candidate_checkpoint).resolve()
        payload['candidate_gate_checkpoint'] = str(path)
        payload['candidate_gate_checkpoint_sha256'] = sha256(path)
    if args.metrics_json:
        payload['metrics_json'] = str(Path(args.metrics_json).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path

from common import save_json


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description='Summarize Industrial v4 diagnostics and Gate conditions.')
    parser.add_argument('--full_oracle', required=True)
    parser.add_argument('--calibration_oracle', required=True)
    parser.add_argument('--utility_analysis', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_markdown', required=True)
    parser.add_argument('--memory_safe', action='store_true')
    parser.add_argument('--min_utility_auc', type=float, default=0.60)
    parser.add_argument('--min_oracle_gap', type=float, default=0.001)
    args = parser.parse_args()

    full_oracle = load_json(args.full_oracle)
    calibration_oracle = load_json(args.calibration_oracle)
    utility = load_json(args.utility_analysis)

    text_ndcg = calibration_oracle['text_only']['NDCG@10']
    fixed_ndcg = calibration_oracle['v1_fixed']['NDCG@10']
    global_ndcg = calibration_oracle['best_global_metrics']['NDCG@10']
    oracle_ndcg = calibration_oracle['oracle_alpha']['NDCG@10']
    oracle_gap = oracle_ndcg - global_ndcg
    oracle_gain = oracle_ndcg - text_ndcg
    unexplained_fraction = oracle_gap / oracle_gain if oracle_gain > 0 else 0.0
    ce_direction = calibration_oracle['ce_utility_direction']
    rank_direction = calibration_oracle['rank_utility_direction']
    ce_positive_rate = utility['ce_utility']['train_positive_rate']
    rank_positive_rate = utility['rank_utility_improved_vs_non_improved']['train_positive_rate']

    conditions = {
        'oracle_above_best_global': oracle_gap >= args.min_oracle_gap,
        'best_global_does_not_explain_most_oracle_gain': unexplained_fraction >= 0.25,
        'ce_utility_auc_at_least_threshold': utility['ce_utility_auc'] >= args.min_utility_auc,
        'rank_utility_auc_at_least_threshold': utility['rank_utility_auc'] >= args.min_utility_auc,
        'ce_direction_supported': bool(
            ce_direction['positive_direction_supported'] and ce_direction['negative_direction_supported']
        ),
        'rank_direction_supported': bool(
            rank_direction['positive_direction_supported'] and rank_direction['negative_direction_supported']
        ),
        'labels_not_collapsed': 0.1 <= ce_positive_rate <= 0.9 and 0.1 <= rank_positive_rate <= 0.9,
        'memory_safe': args.memory_safe,
    }
    result = {
        'dataset': calibration_oracle['dataset'],
        'selection_split': calibration_oracle['split'],
        'full_validation_oracle_ndcg10': full_oracle['oracle_alpha']['NDCG@10'],
        'v1_alpha0': calibration_oracle['alpha0'],
        'v1_alpha0_ndcg10': fixed_ndcg,
        'best_global_alpha': calibration_oracle['best_global_alpha'],
        'best_global_ndcg10': global_ndcg,
        'oracle_alpha_ndcg10': oracle_ndcg,
        'oracle_gap_over_best_global': oracle_gap,
        'oracle_gain_unexplained_by_global_fraction': unexplained_fraction,
        'ce_utility_auc': utility['ce_utility_auc'],
        'rank_utility_auc': utility['rank_utility_auc'],
        'ce_rank_label_agreement': utility['ce_rank_label_agreement'],
        'rank_improved_rate': utility['rank_improved_rate'],
        'rank_harmed_rate': utility['rank_harmed_rate'],
        'rank_tie_rate': utility['rank_tie_rate'],
        'ce_utility_direction': ce_direction,
        'rank_utility_direction': rank_direction,
        'gate_conditions': conditions,
        'gate_training_allowed': all(conditions.values()),
    }
    save_json(args.output_json, result)

    lines = [
        '# Industrial v4 全量诊断摘要',
        '',
        '本文件由诊断脚本生成；所有 alpha 选择均来自 calibration-valid，未使用 test。',
        '',
        '## 核心结果',
        '',
        f'- v1 alpha0: {result["v1_alpha0"]}',
        f'- v1 fixed NDCG@10: {fixed_ndcg:.6f}',
        f'- best global alpha: {result["best_global_alpha"]}',
        f'- best global NDCG@10: {global_ndcg:.6f}',
        f'- Oracle-Alpha NDCG@10: {oracle_ndcg:.6f}',
        f'- Oracle 相对 best global 差值: {oracle_gap:.6f}',
        f'- CE Utility AUC: {utility["ce_utility_auc"]:.6f}',
        f'- Rank Utility AUC: {utility["rank_utility_auc"]:.6f}',
        f'- CE/Rank 标签一致率: {utility["ce_rank_label_agreement"]:.6f}',
        '',
        '## Gate 条件',
        '',
    ]
    lines.extend(f'- {name}: {value}' for name, value in conditions.items())
    lines.extend(['', f'结论：允许训练 Context Gate = {result["gate_training_allowed"]}', ''])
    markdown_path = Path(args.output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text('\n'.join(lines), encoding='utf-8')
    print(result)
    print(f'Saved summary JSON: {args.output_json}')
    print(f'Saved summary Markdown: {args.output_markdown}')


if __name__ == '__main__':
    main()

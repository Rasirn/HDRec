"""
extract_sasrec_embeddings.py

训练完 SASRec 后，从 RecBole 保存的 .pth 检查点中提取 item embeddings，
按 smap.json 的 HDRec item ID 顺序重排后，保存到 ./temp/SASRec/ 目录。

用法:
    python extract_sasrec_embeddings.py \
        --dataset Arts_Crafts_and_Sewing \
        --model_path /path/to/SASRec-xxx.pth \
        --recbole_dataset_dir /path/to/RecBole/dataset \
        --smap_path ./data/Arts_Crafts_and_Sewing/smap.json \
        --emb_dim 128 \
        --output_dir ./temp/SASRec

需要在激活了 recbole 环境、且在 RecBole 目录下运行（以便 import recbole）。
"""

import argparse
import json
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称，如 Arts_Crafts_and_Sewing')
    parser.add_argument('--model_path', type=str, required=True,
                        help='RecBole 保存的 SASRec 模型文件路径')
    parser.add_argument('--recbole_dataset_dir', type=str, required=True,
                        help='RecBole dataset 目录（含 {dataset}/{dataset}.inter）')
    parser.add_argument('--smap_path', type=str, required=True,
                        help='HDRec smap.json 路径')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='embedding 维度（与 preference_dim 一致，默认 128）')
    parser.add_argument('--output_dir', type=str, default='./temp/SASRec',
                        help='输出目录')
    args = parser.parse_args()

    # ---- 1. 加载 RecBole 检查点 ----
    print(f'[1] 加载模型: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']

    item_emb_key = 'item_embedding.weight'
    if item_emb_key not in state_dict:
        emb_keys = [k for k in state_dict if 'item_embedding' in k or 'item_emb' in k]
        print(f'  找到的 item embedding keys: {emb_keys}')
        item_emb_key = emb_keys[0]
    recbole_item_emb = state_dict[item_emb_key]  # shape: [n_items+1, dim]
    print(f'  RecBole item embedding shape: {recbole_item_emb.shape}')

    # ---- 2. 用 RecBole API 加载 Dataset 获取 token 映射 ----
    print(f'[2] 加载 RecBole Dataset 以获取 item_id token 映射...')
    # 将 RecBole 目录加入 path
    recbole_root = os.path.dirname(os.path.abspath(args.recbole_dataset_dir))
    if recbole_root not in sys.path:
        sys.path.insert(0, recbole_root)

    from recbole.config import Config
    from recbole.data import create_dataset

    config_dict = {
        'model': 'SASRec',
        'dataset': args.dataset,
        'data_path': args.recbole_dataset_dir,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'train_neg_sample_args': None,
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full',
        },
        'MAX_ITEM_LIST_LENGTH': 50,
        'use_gpu': False,
        'show_progress': False,
    }

    config = Config(model='SASRec', dataset=args.dataset, config_dict=config_dict)
    dataset = create_dataset(config)
    item_token2id = dataset.field2token_id['item_id']  # asin -> recbole_internal_id
    print(f'  RecBole item token 数量 (含padding): {len(item_token2id)}')

    # ---- 3. 加载 HDRec smap.json ----
    print(f'[3] 加载 smap: {args.smap_path}')
    with open(args.smap_path, 'r') as f:
        smap = json.load(f)  # asin -> hdrec_item_id (0-indexed)
    hdrec_item_num = len(smap)
    print(f'  HDRec item 数量: {hdrec_item_num}')

    # ---- 4. 构建重排后的 embedding 矩阵 ----
    # HDRec embedding shape: [item_num+1, emb_dim]，index 0 保留为 padding
    output_emb = torch.zeros(hdrec_item_num + 1, args.emb_dim)
    missing = 0
    for asin, hdrec_id in smap.items():
        recbole_id = item_token2id.get(asin, None)
        if recbole_id is None:
            missing += 1
            continue
        output_emb[hdrec_id] = recbole_item_emb[recbole_id]

    if missing > 0:
        print(f'  警告: {missing} 个 item 在 RecBole 中未找到，对应行保持为 0 向量。')
    print(f'  输出 embedding shape: {output_emb.shape}')

    # ---- 5. 保存 ----
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.dataset}-{args.emb_dim}.pth')
    torch.save(output_emb, save_path)
    print(f'[4] 已保存: {save_path}')


if __name__ == '__main__':
    main()

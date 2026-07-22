import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
V4_DIR = ROOT / 'versions' / 'v4'
MODEL_DIR = V4_DIR / 'model'
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data.data import Collator, RecDataset  # noqa: E402
from data.data_utils import load_data, tokenize_items  # noqa: E402
from models.model_utils import get_model_config_tokenizer  # noqa: E402


DATASET_DEFAULTS = {
    'Arts_Crafts_and_Sewing': {'fusion_alpha': 0.7, 'fusion_temperature': 1.0, 'epochs': 10, 'lr': 1.6e-4, 'score_dropout': 0.4},
    'Industrial_and_Scientific': {'fusion_alpha': 0.5, 'fusion_temperature': 1.0, 'epochs': 10, 'lr': 1.5e-4, 'score_dropout': 0.5},
    'Musical_Instruments': {'fusion_alpha': 0.7, 'fusion_temperature': 1.0, 'epochs': 8, 'lr': 1.5e-4, 'score_dropout': 0.4},
    'Prime_Pantry': {'fusion_alpha': 0.3, 'fusion_temperature': 1.0, 'epochs': 10, 'lr': 1.5e-4, 'score_dropout': 0.5},
    'Video_Games': {'fusion_alpha': 0.5, 'fusion_temperature': 1.2, 'epochs': 12, 'lr': 1.5e-4, 'score_dropout': 0.4},
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_alpha(dataset):
    return DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS['Industrial_and_Scientific'])['fusion_alpha']


def default_temperature(dataset):
    return DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS['Industrial_and_Scientific'])['fusion_temperature']


def make_v1_args(cli_args, item_num=None, user_num=None, manifest_args=None):
    defaults = DATASET_DEFAULTS.get(cli_args.dataset, DATASET_DEFAULTS['Industrial_and_Scientific'])
    values = dict(
        debug=cli_args.debug,
        suffix='v4_cache',
        checkpoint_dir=None,
        load_model='pytorch_model',
        data_root=cli_args.data_root,
        dataset=cli_args.dataset,
        output_dir=str(V4_DIR / 'outputs'),
        train_attr=['title', 'brand'],
        max_attr_length=32,
        max_item_num=cli_args.max_item_num,
        max_token_num=1024,
        model_name_or_path=cli_args.model_name_or_path,
        model_cache_dir=cli_args.model_cache_dir,
        fix_backbone=True,
        fix_emb=True,
        pad_right=False,
        query_same=False,
        no_prompt=False,
        seed=cli_args.seed,
        deepspeed=None,
        mixed_precision=cli_args.mixed_precision,
        num_train_epochs=defaults['epochs'],
        gradient_accumulation_steps=1,
        batch_size=cli_args.batch_size,
        gradient_checkpointing_enable=False,
        num_workers=cli_args.num_workers,
        learning_rate=defaults['lr'],
        weight_decay=1e-2,
        warmup_steps=0,
        skip_valid=0,
        interval=1,
        patient=1,
        save_interval=100,
        metric_ks=[1, 5, 10, 20, 50],
        valid_metric='NDCG@10',
        only_test=False,
        use_item_alignment=1,
        only_id=False,
        late_fusion=False,
        late_fusion_load=False,
        early_fusion=False,
        use_small_model=True,
        method_of_preference='SASRec',
        preference_dim=128,
        use_gate=True,
        alternating_learning=2,
        kl_loss_weight=0.0,
        kl_temperature=1.0,
        fusion_temperature=cli_args.fusion_temperature if cli_args.fusion_temperature is not None else defaults['fusion_temperature'],
        fusion_alpha=cli_args.alpha0 if cli_args.alpha0 is not None else defaults['fusion_alpha'],
        fusion_type='text',
        fusion_before_loss=False,
        use_two_score=False,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        hidden_dropout=0.05,
        adapter_dropout=0.5,
        hd_frequency=8,
        lora_frequency=1,
        score_dropout=defaults['score_dropout'],
        item_num=item_num,
        user_num=user_num,
        logger=SimpleLogger(),
        device=torch.device(cli_args.device),
    )
    # New v1 runs persist their resolved profile arguments.  Reconstructing the
    # frozen model with those values avoids treating a src_original checkpoint
    # as the old legacy-tuned architecture/configuration.
    for key, value in (manifest_args or {}).items():
        if key in values and key not in {'device', 'logger', 'data_root', 'dataset', 'model_name_or_path'}:
            values[key] = value
    values['data_root'] = cli_args.data_root
    values['dataset'] = cli_args.dataset
    values['model_name_or_path'] = cli_args.model_name_or_path
    values['device'] = torch.device(cli_args.device)
    return SimpleNamespace(**values)


class SimpleLogger:
    def info(self, msg):
        print(msg)


def add_common_cache_args(parser):
    parser.add_argument('--dataset', default='Industrial_and_Scientific')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--model_name_or_path', default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    parser.add_argument('--model_cache_dir', default='path/to/your/model_cache_dir')
    parser.add_argument('--checkpoint_path', default=None, help='Frozen v1 pytorch_model.bin path.')
    parser.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'calibration'])
    parser.add_argument('--cache_path', default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_item_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mixed_precision', default='bf16')
    parser.add_argument('--alpha0', type=float, default=None)
    parser.add_argument('--fusion_temperature', type=float, default=None)
    parser.add_argument('--feature_topk', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')


def infer_checkpoint_path(dataset, model_name_or_path, suffix='v1'):
    model_prefix = model_name_or_path.replace('/', '-')
    output_root = ROOT / 'versions' / 'v1' / 'outputs' / dataset
    exact = output_root / f'{model_prefix}_{suffix}' / 'pytorch_model.bin'
    if exact.exists():
        return exact
    candidates = sorted(output_root.glob(f'{model_prefix}_*/pytorch_model.bin'))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return exact
    raise RuntimeError(
        f'Multiple v1 checkpoints found for {dataset}; pass --checkpoint_path explicitly: '
        f'{[str(path) for path in candidates]}'
    )


def load_frozen_v1(cli_args):
    set_seed(cli_args.seed)
    train, val, test, item_meta_dict, item2id = load_data(SimpleNamespace(data_root=cli_args.data_root, dataset=cli_args.dataset))
    checkpoint_path = Path(cli_args.checkpoint_path) if cli_args.checkpoint_path else infer_checkpoint_path(cli_args.dataset, cli_args.model_name_or_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'v1 checkpoint not found: {checkpoint_path}')
    manifest_path = checkpoint_path.parent / 'run_manifest.json'
    manifest_args = None
    if manifest_path.exists():
        with manifest_path.open() as handle:
            manifest_args = json.load(handle).get('final_args', {})
    args = make_v1_args(cli_args, item_num=len(item2id), user_num=len(train), manifest_args=manifest_args)
    model, _, tokenizer = get_model_config_tokenizer(args)
    state = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(state, dict):
        raise TypeError(f'v1 checkpoint must contain a state dict: {checkpoint_path}')
    info = model.load_state_dict(state, strict=False)
    model_config = getattr(model, 'config', None)
    config_name = getattr(model_config, '_name_or_path', cli_args.model_name_or_path)
    print(f'Checkpoint path: {checkpoint_path.resolve()}')
    print(f'Model class: {model.__class__.__module__}.{model.__class__.__name__}')
    print(f'Model config: {config_name}')
    print(f'Missing keys: {info.missing_keys}')
    print(f'Unexpected keys: {info.unexpected_keys}')
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(
            'Frozen v1 checkpoint is incompatible with the constructed model; '
            f'missing={info.missing_keys}, unexpected={info.unexpected_keys}'
        )
    args.loaded_checkpoint_path = str(checkpoint_path.resolve())
    print('Frozen v1 checkpoint loaded with an exact key match.')
    model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenized_items = tokenize_items(item_meta_dict, tokenizer, args, 0)
    return args, model, tokenizer, (train, val, test, tokenized_items)


def build_loader(args, tokenizer, data_tuple, split):
    train, val, test, tokenized_items = data_tuple
    mode = 'valid' if split == 'calibration' else split
    dataset = RecDataset(train, val, test, tokenized_items, args, mode, tokenizer)
    collator = Collator(args, tokenizer)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator), dataset


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, tuple):
            out[key] = tuple(x.to(device) if torch.is_tensor(x) else x for x in value)
        elif isinstance(value, dict):
            out[key] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in value.items()}
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def save_json(path, obj):
    def to_jsonable(value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        return value

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def load_cache(path):
    payload = torch.load(path, map_location='cpu')
    if isinstance(payload, dict) and 'logits_text' in payload:
        return payload
    raise ValueError(f'Unsupported cache format: {path}')


def default_cache_path(dataset, split):
    return V4_DIR / 'results' / 'cache' / dataset / f'{split}.pt'

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from common import add_common_cache_args, build_loader, default_cache_path, load_frozen_v1, move_batch
from reliability_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, compute_sequence_features, history_pop_stats, item_popularity_from_train
from utility_label import fixed_text_fusion, id_confidence_residual, utility_labels, utility_values, cross_entropy_per_sample


def history_for_user(dataset, user_id, split):
    train_hist = dataset.user2train.get(int(user_id), [])
    if split == 'test':
        return train_hist + dataset.user2val.get(int(user_id), [])
    return train_hist


def main():
    parser = argparse.ArgumentParser(description='Cache frozen v1 text/id logits for v4 diagnostics and fuser training.')
    add_common_cache_args(parser)
    args = parser.parse_args()

    out_path = Path(args.cache_path) if args.cache_path else default_cache_path(args.dataset, args.split)
    if out_path.exists() and not args.overwrite:
        print(f'Cache exists, reuse without overwrite: {out_path}')
        return

    v1_args, model, tokenizer, data_tuple = load_frozen_v1(args)
    loader, dataset = build_loader(v1_args, tokenizer, data_tuple, args.split)
    train, _, _, _ = data_tuple
    popularity = item_popularity_from_train(train, v1_args.item_num)

    all_text, all_id, all_labels, all_users = [], [], [], []
    all_features, all_hist_len, all_pop_mean, all_pop_std = [], [], [], []
    all_text_loss, all_fixed_loss, all_utility, all_utility_label = [], [], [], []

    seen = 0
    device = v1_args.device
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc=f'Cache {args.split}'):
            batch = move_batch(batch, device)
            input_ids, attention_mask, _, labels = batch['user_seq_data']
            item_input_ids, item_seq_mask, _ = batch['item_data']

            if hasattr(model, 'set_adapter'):
                model.set_adapter('lora_text')
            logits_text, _ = model(input_ids=input_ids, attention_mask=attention_mask, adapter_name='lora_text')

            if hasattr(model, 'set_adapter'):
                model.set_adapter('lora_cf')
            logits_id, _ = model(input_ids=item_input_ids, attention_mask=item_seq_mask, adapter_name='lora_cf', is_text=False)

            labels = labels.long()
            user_ids = batch['user_ids'].long()
            histories = [history_for_user(dataset, uid.item(), args.split) for uid in user_ids.cpu()]
            hist_len = torch.tensor([len(h) for h in histories], dtype=torch.float32)
            pop_mean, pop_std = history_pop_stats(histories, popularity)

            features = compute_sequence_features(
                logits_text.cpu(),
                logits_id.cpu(),
                history_length=hist_len,
                history_pop_mean=pop_mean,
                history_pop_std=pop_std,
                topk=args.feature_topk,
            )
            fixed_logits = fixed_text_fusion(logits_text.cpu(), logits_id.cpu(), alpha=v1_args.fusion_alpha, temperature=v1_args.fusion_temperature)
            text_loss = cross_entropy_per_sample(logits_text.cpu(), labels.cpu())
            fixed_loss = cross_entropy_per_sample(fixed_logits, labels.cpu())
            utility = utility_values(logits_text.cpu(), logits_id.cpu(), labels.cpu(), temperature=v1_args.fusion_temperature)
            util_label = utility_labels(logits_text.cpu(), logits_id.cpu(), labels.cpu(), temperature=v1_args.fusion_temperature)

            all_text.append(logits_text.cpu())
            all_id.append(logits_id.cpu())
            all_labels.append(labels.cpu())
            all_users.append(user_ids.cpu())
            all_features.append(features)
            all_hist_len.append(hist_len)
            all_pop_mean.append(pop_mean)
            all_pop_std.append(pop_std)
            all_text_loss.append(text_loss)
            all_fixed_loss.append(fixed_loss)
            all_utility.append(utility)
            all_utility_label.append(util_label)

            seen += labels.numel()
            if args.max_samples is not None and seen >= args.max_samples:
                break

    payload = {
        'version': 'v4_cache_v1',
        'feature_schema': FEATURE_SCHEMA_VERSION,
        'feature_names': FEATURE_NAMES,
        'dataset': args.dataset,
        'split': args.split,
        'seed': args.seed,
        'checkpoint_path': str(args.checkpoint_path or 'auto:v1'),
        'alpha0': float(v1_args.fusion_alpha),
        'fusion_temperature': float(v1_args.fusion_temperature),
        'logits_text': torch.cat(all_text, dim=0),
        'logits_id': torch.cat(all_id, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'user_ids': torch.cat(all_users, dim=0),
        'features': torch.cat(all_features, dim=0),
        'history_length': torch.cat(all_hist_len, dim=0),
        'history_pop_mean': torch.cat(all_pop_mean, dim=0),
        'history_pop_std': torch.cat(all_pop_std, dim=0),
        'text_loss': torch.cat(all_text_loss, dim=0),
        'fixed_loss': torch.cat(all_fixed_loss, dim=0),
        'utility': torch.cat(all_utility, dim=0),
        'utility_label': torch.cat(all_utility_label, dim=0),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f'Saved cache: {out_path}')
    print(f'Samples: {payload["labels"].numel()}, items: {payload["logits_text"].size(-1)}')


if __name__ == '__main__':
    main()

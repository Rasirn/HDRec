"""Explicit, auditable v1 execution profiles.

`src_original` mirrors the defaults in ``src/parameters.py``.  The model-mode
flags are explicit because HDRec's original command line enables its dual
branch LoRA model even though argparse flags default to false.
"""

SRC_ORIGINAL = {
    'num_train_epochs': 40,
    'learning_rate': 5e-5,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,
    'weight_decay': 0.0,
    'warmup_steps': 2000,
    'mixed_precision': 'no',
    'score_dropout': 0.5,
    'hidden_dropout': 0.0,
    'adapter_dropout': 0.3,
    'lora_r': 8,
    'lora_alpha': 32,
    'lora_frequency': 1,
    'hd_frequency': 1,
    'kl_loss_weight': 1.0,
    'alternating_learning': 2,
    'fusion_alpha': 0.5,
    'fusion_temperature': 0.5,
    'fusion_type': 'text',
    'max_item_num': 30,
    'max_token_num': 1024,
    'skip_valid': 15,
    'patient': 10,
    'seed': 42,
    'num_workers': 1,
}

# Retains the previous versions/run_common.sh behaviour only when explicitly
# requested.  It is not a v1 baseline profile.
LEGACY_TUNED = {
    'default': {
        **SRC_ORIGINAL,
        'weight_decay': 1e-2, 'hidden_dropout': 0.05, 'adapter_dropout': 0.5,
        'hd_frequency': 8, 'max_item_num': 10, 'patient': 1,
        'skip_valid': 0, 'mixed_precision': 'bf16',
    },
    'Arts_Crafts_and_Sewing': {
        'num_train_epochs': 10, 'learning_rate': 1.6e-4, 'score_dropout': 0.4,
        'kl_loss_weight': 0.5, 'fusion_alpha': 0.7, 'fusion_temperature': 1.0,
    },
    'Industrial_and_Scientific': {
        'num_train_epochs': 10, 'learning_rate': 1.5e-4, 'score_dropout': 0.5,
        'kl_loss_weight': 0.7, 'fusion_alpha': 0.5, 'fusion_temperature': 1.0,
    },
    'Musical_Instruments': {
        'num_train_epochs': 8, 'learning_rate': 1.5e-4, 'score_dropout': 0.4,
        'kl_loss_weight': 0.7, 'fusion_alpha': 0.7, 'fusion_temperature': 1.0,
    },
    'Prime_Pantry': {
        'num_train_epochs': 10, 'learning_rate': 1.5e-4, 'score_dropout': 0.5,
        'kl_loss_weight': 0.3, 'fusion_alpha': 0.3, 'fusion_temperature': 1.0,
    },
    'Video_Games': {
        'num_train_epochs': 12, 'learning_rate': 1.5e-4, 'score_dropout': 0.4,
        'kl_loss_weight': 0.3, 'fusion_alpha': 0.5, 'fusion_temperature': 1.2,
    },
}


def profile_values(profile, dataset):
    if profile == 'src_original':
        return dict(SRC_ORIGINAL)
    if profile == 'legacy_tuned':
        values = dict(LEGACY_TUNED['default'])
        values.update(LEGACY_TUNED.get(dataset, {}))
        return values
    raise ValueError(f'Unknown v1 profile: {profile}')

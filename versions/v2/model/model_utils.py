import os
import torch
from transformers import AutoConfig, AutoTokenizer

import models
from flylora import replace_with_flylora


def print_trainable_parameters(args, model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    args.logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"Total params: {total_params:,} || "
        f"Trainable percentage: {100 * trainable_params / total_params:.2f}%"
    )


def get_model_config_tokenizer(args):
    kwargs = {'cache_dir': args.model_cache_dir, 'local_files_only': os.path.exists(args.model_cache_dir)}
    if args.checkpoint_dir is not None:
        kwargs = {}
        args.model_name_or_path = args.checkpoint_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    process_query_tokens(args, tokenizer)

    config = AutoConfig.from_pretrained(args.model_name_or_path, **kwargs)
    setup_config(config, args)

    model = load_model(args, config, tokenizer, kwargs)
    if args.mixed_precision == 'bf16':
        model.to(torch.bfloat16)

    model = apply_model_customizations(args, model)
    return model, config, tokenizer


def process_query_tokens(args, tokenizer):
    special_tokens = ['[SEQ]']
    if args.checkpoint_dir is None:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.query_token_ids = [tokenizer.convert_tokens_to_ids('[SEQ]')]

    if args.query_same:
        args.alignment_ids = args.query_token_ids
    else:
        alignment_tokens = ['[ALIGN]']
        if args.checkpoint_dir is None:
            special_tokens = special_tokens + alignment_tokens
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        args.alignment_ids = [tokenizer.convert_tokens_to_ids('[ALIGN]')]

    for key, value in tokenizer.special_tokens_map.items():
        args.logger.info(f'{key}: {value}; {tokenizer.convert_tokens_to_ids(value)}')


def setup_config(config, args):
    config.linear_dim = args.item_num
    config.use_small_model = args.use_small_model
    config.preference_dim = args.preference_dim
    config.method_of_preference = args.method_of_preference
    config.user_num = args.user_num
    config.item_num = args.item_num
    config.dataset = args.dataset
    config.score_dropout = args.score_dropout
    config.hd_frequency = args.hd_frequency
    config.hidden_dropout = args.hidden_dropout
    config.only_test = args.only_test
    config.late_fusion = args.late_fusion
    config.late_fusion_load = args.late_fusion_load
    config.use_two_score = args.use_two_score


def load_model(args, config, tokenizer, kwargs):
    model_name = args.model_name_or_path.lower()

    if args.checkpoint_dir is None:
        model = load_pretrained_model(args, model_name, config, kwargs)
        model.resize_token_embeddings(len(tokenizer))
        model.set_emb()
        if model.config.use_small_model:
            model.set_small_model()
    else:
        model = load_model_from_checkpoint(args, model_name, config)

    config.vocab_size = len(tokenizer)
    if args.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()

    return model


def load_pretrained_model(args, model_name, config, kwargs):
    model_map = {
        'llama': models.llama_rec,
    }

    for key in model_map:
        if key in model_name:
            if args.mixed_precision == 'bf16':
                return model_map[key].from_pretrained(
                    args.model_name_or_path,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map=args.device,
                    **kwargs,
                )
            return model_map[key].from_pretrained(
                args.model_name_or_path,
                config=config,
                low_cpu_mem_usage=True,
                device_map=args.device,
                **kwargs,
            )

    args.logger.info('Other models are not supported now.')
    raise ValueError('Unsupported model')


def load_model_from_checkpoint(args, model_name, config):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    state_dict = torch.load(os.path.join(args.checkpoint_dir, args.load_model + '.bin'), map_location=device)

    model_map = {
        'llama': models.llama_rec,
    }

    for key in model_map:
        if key in model_name:
            model = model_map[key](config)
            model.set_linear(config.linear_dim)
            model.set_emb()
            if args.use_small_model:
                model.set_small_model()

            if not args.only_test:
                state_dict.pop('score.weight', None)
            info = model.load_state_dict(state_dict, strict=False)
            args.logger.info(f'Load info: {info}')
            return model

    args.logger.info('Other models are not supported now.')
    raise ValueError('Unsupported model')


def _collect_target_linear_modules(args, model):
    if args.lora_frequency == 1:
        target_layer_indices = [i for i in range(model.config.num_hidden_layers) if i % args.lora_frequency == 0]
    else:
        target_layer_indices = [i for i in range(model.config.num_hidden_layers) if i % args.lora_frequency != 0]

    target_module_types = ['k_proj', 'v_proj', 'q_proj', 'o_proj', 'up_proj', 'down_proj']
    target_modules = []

    for name, _ in model.named_modules():
        if 'layers.' in name and 'small_model' not in name:
            layer_idx = int(name.split('.')[1])
            if layer_idx in target_layer_indices:
                module_type = name.split('.')[-1]
                if module_type in target_module_types:
                    target_modules.append(name)
    return target_modules


def set_flylora(args, model):
    target_modules = _collect_target_linear_modules(args, model)
    replaced = replace_with_flylora(
        model,
        target_modules,
        r=args.flylora_r,
        k=args.flylora_k,
        alpha=args.flylora_alpha,
        sparsity_ratio=args.flylora_sparsity_ratio,
        bias_lr=args.flylora_bias_lr,
    )
    args.logger.info(f'FlyLoRA injected modules: {replaced}')

    for _, param in model.named_parameters():
        param.requires_grad = False

    need_train = ['flylora_B', 'score', 'norm', 'new_emb', 'small_model']
    for name, param in model.named_parameters():
        if any(x in name for x in need_train):
            param.requires_grad = True
        if args.use_small_model and ('projection' in name):
            param.requires_grad = True
    return model


def apply_model_customizations(args, model):
    if args.use_flylora:
        model = set_flylora(args, model)

    if args.fix_backbone:
        freeze_model_layers(args, model)

    if args.fix_emb:
        freeze_embedding_layers(args, model)
    return model


def freeze_model_layers(args, model):
    layer_attr = None
    model_name = args.model_name_or_path.lower()

    if 'opt' in model_name:
        layer_attr = model.decoder.named_parameters()
    elif 'bloom' in model_name:
        layer_attr = model.h.named_parameters()
    else:
        layer_attr = model.layers.named_parameters()

    if layer_attr:
        need_train = ['flylora_B', 'norm']
        for name, param in layer_attr:
            if not any(x in name for x in need_train):
                param.requires_grad = False

    if 'opt' in model_name:
        for _, param in model.decoder.embed_tokens.named_parameters():
            param.requires_grad = True


def freeze_embedding_layers(args, model):
    embedding_attr = None
    model_name = args.model_name_or_path.lower()

    if 'opt' in model_name:
        embedding_attr = model.decoder.embed_tokens.named_parameters()
    elif 'bloom' in model_name:
        embedding_attr = model.word_embeddings.named_parameters()
    else:
        embedding_attr = model.embed_tokens.named_parameters()

    if embedding_attr is not None:
        for name, param in embedding_attr:
            if 'embedding' in name:
                param.requires_grad = False

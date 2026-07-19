import os
import torch
from transformers import AutoConfig, AutoTokenizer

import models

def print_trainable_parameters(args, model):
    """Logs the number of trainable and total parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    args.logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"Total params: {total_params:,} || "
        f"Trainable percentage: {100 * trainable_params / total_params:.2f}%"
    )

def get_model_config_tokenizer(args):
    """Initializes model, tokenizer, and configuration based on arguments."""
    
    kwargs = {"cache_dir": args.model_cache_dir, "local_files_only": os.path.exists(args.model_cache_dir)}
    if args.checkpoint_dir is not None:
        kwargs = {}
        args.model_name_or_path = args.checkpoint_dir

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **kwargs)
    if tokenizer.pad_token_id is None:      #! different model may have different pad token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Handle query token special tokens
    process_query_tokens(args, tokenizer)

    # Initialize configuration
    config = AutoConfig.from_pretrained(args.model_name_or_path, **kwargs)
    setup_config(config, args)

    # Initialize model
    model = load_model(args, config, tokenizer, kwargs)
    if args.mixed_precision == 'bf16':
        model.to(torch.bfloat16) #!

    model = apply_model_customizations(args, model)

    return model, config, tokenizer


def process_query_tokens(args, tokenizer):
    """Handles special tokens for query."""
    special_tokens = ['[SEQ]']
    if args.checkpoint_dir is None:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.query_token_ids = [tokenizer.convert_tokens_to_ids('[SEQ]')]

    # Handle alignment tokens
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
    """Sets up configuration parameters for the model."""
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
    """Loads the model based on the provided configuration and tokenizer."""
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
    """Loads a pretrained model based on its type."""
    model_map = {
        'llama': models.llama_rec,
    }

    for key in model_map:
        if key in model_name:
            if args.mixed_precision == 'bf16':
                return model_map[key].from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device, **kwargs)
            else:
                return model_map[key].from_pretrained(args.model_name_or_path, config=config, low_cpu_mem_usage=True, device_map=args.device, **kwargs)
    
    args.logger.info('Other models are not supported now.')
    exit()


def load_model_from_checkpoint(args, model_name, config):
    """Loads a model from a specified checkpoint."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
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
            if args.use_lora and args.only_test:
                set_lora(args, model)

            if not args.only_test:
                state_dict.pop('score.weight')
            info = model.load_state_dict(state_dict, strict=False)
            args.logger.info(f'Load info: {info}')
            return model

    args.logger.info('Other models are not supported now.')
    exit()

def set_lora(args, model):
    if args.lora_frequency == 1:
        target_layer_indices = [i for i in range(model.config.num_hidden_layers) if i % args.lora_frequency == 0]
    else:
        target_layer_indices = [i for i in range(model.config.num_hidden_layers) if i % args.lora_frequency != 0]

    target_module_types = ['k_proj', 'v_proj', 'q_proj', 'o_proj', 'up_proj', 'down_proj']


    target_modules = []
    for name, _ in model.named_modules():
        if "layers." in name and 'small_model' not in name:
            layer_idx = int(name.split(".")[1])
            if layer_idx in target_layer_indices:

                module_type = name.split(".")[-1]
                if module_type in target_module_types:
                    target_modules.append(name)
    from peft import get_peft_model, LoraConfig #, AdaLoraConfig
    lora_config = LoraConfig(r=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            lora_dropout=args.adapter_dropout,
                            target_modules=target_modules,
                            task_type="SEQ_CLS",)
    model = get_peft_model(model, lora_config, adapter_name="lora_text")
    if args.use_gate and args.alternating_learning > 0:
        model.add_adapter("lora_cf", lora_config)
    
    need_train = ['lora', 'score', 'norm', 'new_emb', 'small_model']
    for name, param in model.named_parameters():
        if any([x in name for x in need_train]):
            param.requires_grad = True
        if args.use_small_model and ('projection' in name):
            param.requires_grad = True
    return model

def apply_model_customizations(args, model):
    """Applies optional customizations to the model, such as LoRA or freezing layers."""
    if args.use_lora and not args.only_test:
        model = set_lora(args, model)

    if args.fix_backbone:
        freeze_model_layers(args, model)

    if args.fix_emb:
        freeze_embedding_layers(args, model)
    return model

def freeze_model_layers(args, model):
    """Freezes the backbone layers of the model."""
    layer_attr = None
    model_name = args.model_name_or_path.lower()

    if 'opt' in model_name:
        layer_attr = model.decoder.named_parameters()
    elif 'bloom' in model_name:
        layer_attr = model.h.named_parameters()
    else:
        layer_attr = model.layers.named_parameters()

    if layer_attr:
        need_train = ['lora', 'norm']
        for name, param in layer_attr:
            if not any([x in name for x in need_train]):
                param.requires_grad = False
    
    if 'opt' in model_name: # opt's embed_tokens are in decoder
        for name, param in model.decoder.embed_tokens.named_parameters():
            param.requires_grad = True


def freeze_embedding_layers(args, model):
    """Freezes the embedding layers of the model."""
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

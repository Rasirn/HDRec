import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    # debug
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for more verbose output.')
    parser.add_argument('--suffix', type=str, default='flylora', help='Suffix for the output directory.')

    # data and log path
    parser.add_argument('--checkpoint_dir', type=str, default=None, required=False, help='Directory to save checkpoints.')
    parser.add_argument('--load_model', type=str, default='pytorch_model', required=False, help='Model name to be loaded.')
    parser.add_argument('--data_root', type=str, default='./data', required=False, help='Root directory of the dataset.')
    parser.add_argument('--dataset', type=str, default='Industrial_and_Scientific', required=False, help='Dataset name.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output logs and results.')

    # data process
    parser.add_argument('--train_attr', nargs='+', default=['title', 'brand'], help='Attributes to use for training.')
    parser.add_argument('--max_attr_length', type=int, default=32, help='Maximum length of each attribute.')
    parser.add_argument('--max_item_num', type=int, default=30, help='Maximum number of items to process.')
    parser.add_argument('--max_token_num', type=int, default=1024, help='Maximum number of tokens to process.')

    # backbone model
    parser.add_argument('--model_name_or_path', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B', help='Backbone model path.')
    parser.add_argument('--model_cache_dir', type=str, default='path/to/your/model_cache_dir', help='Model cache directory.')
    parser.add_argument('--fix_backbone', action='store_true', help='Freeze backbone model.')
    parser.add_argument('--fix_emb', action='store_true', help='Freeze embedding layer.')

    # tokenizer and virtual token
    parser.add_argument('--pad_right', action='store_true', help='Pad sequences to the right side.')
    parser.add_argument('--query_same', action='store_true', help='Use the same query token for all positions.')
    parser.add_argument('--no_prompt', action='store_true', help='Disable prompt tokens.')

    # Training
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization.')
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed config file path.')
    parser.add_argument('--mixed_precision', type=str, default='no', help='Mixed precision mode.')
    parser.add_argument('--num_train_epochs', type=int, default=40, help='Total number of training epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per training step.')
    parser.add_argument('--gradient_checkpointing_enable', action='store_true', help='Enable gradient checkpointing.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker threads for loading data.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for model.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for model.')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Warmup steps for scheduler.')
    parser.add_argument('--skip_valid', type=int, default=15, help='Number of epochs to skip validation.')
    parser.add_argument('--interval', type=int, default=1, help='Validation interval.')
    parser.add_argument('--patient', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--save_interval', type=int, default=100, help='Interval for saving model checkpoint.')

    # Validation and Testing
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50], help='Metric@k list.')
    parser.add_argument('--valid_metric', type=str, default='NDCG@10', help='Metric for model selection.')
    parser.add_argument('--only_test', action='store_true', help='Only perform testing without training.')

    # item alignment
    parser.add_argument('--use_item_alignment', type=int, default=1, help='Enable item alignment.')

    # small model
    parser.add_argument('--only_id', action='store_true')
    parser.add_argument('--late_fusion', action='store_true')
    parser.add_argument('--late_fusion_load', action='store_true')
    parser.add_argument('--early_fusion', action='store_true')

    parser.add_argument('--use_small_model', action='store_true', help='Enable small model branch.')
    parser.add_argument('--method_of_preference', type=str, default='SASRec', help='Preference model method.')
    parser.add_argument('--preference_dim', type=int, default=128, help='Preference embedding dimension.')

    # fusion + KL divergence
    parser.add_argument('--use_gate', action='store_true', help='Enable gate branch.')
    parser.add_argument('--alternating_learning', type=int, default=0, help='Kept for compatibility, ignored in FlyLoRA mode.')
    parser.add_argument('--kl_loss_weight', type=float, default=1.0)
    parser.add_argument('--kl_temperature', type=float, default=1.0)
    parser.add_argument('--fusion_temperature', type=float, default=0.5)
    parser.add_argument('--fusion_alpha', type=float, default=0.5)
    parser.add_argument('--fusion_type', type=str, default='text')
    parser.add_argument('--fusion_before_loss', action='store_true')
    parser.add_argument('--use_two_score', action='store_true')
    parser.add_argument('--cf_loss_weight', type=float, default=1.0, help='Loss weight for collaborative branch.')

    # Keep old LoRA flags for compatibility with existing scripts
    parser.add_argument('--use_lora', action='store_true', help='Compatibility flag. FlyLoRA path does not use PEFT adapters.')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--hidden_dropout', type=float, default=0)
    parser.add_argument('--adapter_dropout', type=float, default=0.3)
    parser.add_argument('--hd_frequency', type=int, default=1)
    parser.add_argument('--lora_frequency', type=int, default=1)
    parser.add_argument('--score_dropout', type=float, default=0.5)

    # FlyLoRA
    parser.add_argument('--use_flylora', action='store_true', help='Enable FlyLoRA for dual-signal joint training.')
    parser.add_argument('--flylora_r', type=int, default=16, help='Total FlyLoRA rank.')
    parser.add_argument('--flylora_k', type=int, default=4, help='Top-k activated ranks per sample.')
    parser.add_argument('--flylora_alpha', type=float, default=32.0, help='Scaling factor of FlyLoRA update.')
    parser.add_argument('--flylora_sparsity_ratio', type=float, default=0.25, help='Sparse random projection density ratio.')
    parser.add_argument('--flylora_bias_lr', type=float, default=1e-3, help='Bias update lr for load balancing.')

    args = parser.parse_args()

    model_prefix = args.model_name_or_path.replace('/', '-')
    args.output_path = os.path.join(args.output_dir, args.dataset, model_prefix + '_' + args.suffix)

    if args.checkpoint_dir is not None:
        if args.load_model != 'pytorch_model':
            args.output_path = args.output_path + '_' + args.load_model

    if args.only_test:
        if not os.path.exists(args.checkpoint_dir):
            raise ValueError('checkpoint_dir not exist')
        args.output_path = args.checkpoint_dir
        args.log_file = os.path.join(args.output_path, 'test_' + args.dataset + '.log')
    else:
        args.log_file = os.path.join(args.output_path, 'train.log')

    os.makedirs(args.output_path, exist_ok=True)

    return args

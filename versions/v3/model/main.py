import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

from torch.utils.data import ConcatDataset, DataLoader
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

from parameters import get_args
from model_utils import get_model_config_tokenizer, print_trainable_parameters
from trainer import Trainer

from utils import get_logger
from data.data import Collator, ItemDataset, RecDataset
from data.data_utils import load_data, tokenize_items


def setup_accelerator(args):
    deepspeed_plugin = None
    if args.deepspeed:
        hf_ds = HfTrainerDeepSpeedConfig(args.deepspeed)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
    )
    return accelerator


def main():
    args = get_args()
    if not args.use_flylora_dual:
        raise ValueError('Please set --use_flylora_dual for v3 pipeline.')

    set_seed(args.seed)
    accelerator = setup_accelerator(args)

    logger = get_logger(args.log_file, name=__name__)
    args.logger = logger
    args.device = accelerator.device

    train, val, test, item_meta_dict, item2id = load_data(args)
    args.user_num = len(train)
    args.item_num = len(item2id)

    model, config, tokenizer = get_model_config_tokenizer(args)

    for arg, value in vars(args).items():
        logger.info(f'{arg:30} {value}')

    logger.info(model)
    print_trainable_parameters(args, model)

    if accelerator.is_local_main_process:
        tokenizer.save_pretrained(args.output_path)
        config.save_pretrained(args.output_path)

    tokenized_items = tokenize_items(item_meta_dict, tokenizer, args, accelerator)

    train_dataset = RecDataset(train, val, test, tokenized_items, args, 'train', tokenizer)
    if args.use_item_alignment:
        item_dataset = ItemDataset(train, tokenized_items, args, tokenizer)
        train_dataset = ConcatDataset([train_dataset, item_dataset])

    val_dataset = RecDataset(train, val, test, tokenized_items, args, 'valid', tokenizer)
    test_dataset = RecDataset(train, val, test, tokenized_items, args, 'test', tokenizer)

    collator = Collator(args, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator)

    trainer = Trainer(args, accelerator, model, train_loader, val_loader, test_loader)
    trainer.train()

    logger.info('Finish.')


if __name__ == '__main__':
    main()

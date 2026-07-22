import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import torch
from accelerate.utils import set_seed
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
from torch.utils.data import DataLoader, ConcatDataset

from parameters import get_args
from utils import get_logger
from data.data import Collator, RecDataset, ItemDataset
from data.data_utils import load_data, tokenize_items
from models.model_utils import get_model_config_tokenizer, print_trainable_parameters
from trainer import Trainer

# Encapsulate Accelerator setup
def setup_accelerator(args):
    deepspeed_plugin = None
    if args.deepspeed:
        hf_deepspeed_config = HfTrainerDeepSpeedConfig(args.deepspeed)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_deepspeed_config)
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
    if not args.allow_legacy_fixed_fusion:
        raise RuntimeError(
            'This is a legacy fixed-fusion entrypoint and does not run v4 Candidate Gate. '
            'Use versions/v4/run.sh with --v1_checkpoint.'
        )
    set_seed(args.seed)

    # Set up Accelerator
    accelerator = setup_accelerator(args)
    
    # Initialize logger
    logger = get_logger(args.log_file, name=__name__)
    arg_dict = vars(args)
    args.logger = logger
    args.device = accelerator.device

    # Prepare datasets
    train, val, test, item_meta_dict, item2id = load_data(args)
    args.user_num = len(train)
    args.item_num = len(item2id)
    
    # Load model, config, and tokenizer
    model, config, tokenizer = get_model_config_tokenizer(args)

    for arg, value in arg_dict.items():
        logger.info(f'{arg:30} {value}')

    # Log model information and save tokenizer/config if main process
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
    
    if args.debug:
        logger.info('dataset...')
        for i in range(1):
            train_data = train_dataset.__getitem__(i)
            logger.info('[Train Data]')
            # logger.info('[Sequence]: \n' + tokenizer.decode(train_data[0]))
            logger.info('[Sequence]: \n' + str(train_data[0]))
            logger.info('[Target]: \n' + str(train_data[1]))
            logger.info('[Label]: ' + str(train_data[2]))
        # logger.info('data loarder...')
        # for batch in train_loader:
        #     user_seq_data, interactions, context_mask = batch["user_seq_data"], batch["interactions"], batch["context_mask"]
        #     input_ids, attention_mask, target_ids, labels = user_seq_data
        #     user_ids, seq_item_ids, pos_item_ids, neg_item_ids = interactions['user_ids'], interactions['seq_item_ids'], interactions['pos_item_ids'], interactions['neg_item_ids']
        #     logger.info('[Train Loader]')
        #     logger.info('[Sequence]: \n' + str(input_ids[0]))
        #     logger.info('[Target]: \n' + str(target_ids[0]))
        #     logger.info('[Label]: \n' + str(labels[0]))
        #     logger.info('[interactions]: \n')
        #     logger.info('user_ids: ' + str(user_ids[0]))
        #     logger.info('seq_item_ids: ' + str(seq_item_ids[0]))
        #     logger.info('pos_item_ids: ' + str(pos_item_ids[0]))
        #     logger.info('neg_item_ids: ' + str(neg_item_ids[0]))
        #     break
    
    # Initialize trainer and start training
    trainer = Trainer(args, accelerator, model, train_loader, val_loader, test_loader)
    trainer.train()
    
    logger.info('Finish.')

if __name__ == "__main__":
    main()

import os
import json
from tqdm import tqdm

def read_json(path, as_int=False):
    """
    Reads a JSON file and optionally converts its keys to integers.

    Args:
        path (str): Path to the JSON file.
        as_int (bool): Whether to convert the keys to integers. Defaults to False.

    Returns:
        dict: The parsed JSON data, with keys converted to integers if specified.
    """
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(key) if as_int else key: value for key, value in raw.items()}

def load_data(args):
    """
    Loads the dataset for training, validation, and testing along with item metadata, 
    item-to-ID mappings, session-to-user mappings, and negative items.

    Args:
        args: Argument object that contains data path information.

    Returns:
        tuple: Training, validation, and testing data, filtered item metadata, 
               item-to-ID mappings, session-to-user mappings, and negative items.
    """
    dataset_path = os.path.join(args.data_root, args.dataset)

    # Load datasets
    train = read_json(os.path.join(dataset_path, 'train.json'), as_int=True)
    val = read_json(os.path.join(dataset_path, 'val.json'), as_int=True)
    test = read_json(os.path.join(dataset_path, 'test.json'), as_int=True)
    
    # Load metadata
    with open(os.path.join(dataset_path, 'meta_data.json'), 'r') as f:
        item_meta_dict = json.load(f)

    # Load item-to-ID mappings
    item2id = read_json(os.path.join(dataset_path, 'smap.json'))
    
    # Filter item metadata to include only items in item2id
    item_meta_dict_filtered = {item2id[asin]: meta for asin, meta in item_meta_dict.items() if asin in item2id}
    
    return train, val, test, item_meta_dict_filtered, item2id

def tokenize_items(item_meta_dict, tokenizer, args, local_rank):
    """
    Tokenizes item metadata using the provided tokenizer.

    Args:
        item_meta_dict (dict): Dictionary containing item metadata.
        tokenizer: The tokenizer to use for tokenizing the item attributes.
        args: Argument object that contains training and tokenization parameters.
        local_rank (int): Rank of the current process in distributed training, 
                          used to control the display of progress bar.

    Returns:
        dict: A dictionary mapping item IDs to tokenized item attributes.
    """
    item_id2tokens = {}
    show_progress = (local_rank == 0)

    for item_id, item_attrs in tqdm(item_meta_dict.items(), ncols=100, desc='Tokenize Items', disable=not show_progress):
        item_tokens = []

        for idx, attr_name in enumerate(args.train_attr):
            attr_value = item_attrs[attr_name]
            gap = ': '

            # Tokenize attribute value
            if idx == 0:
                attr_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f'{attr_name}: {attr_value}'))
            else:
                attr_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f', {attr_name}: {attr_value}'))

            # Truncate tokens to max attribute length
            attr_tokens = attr_tokens[:args.max_attr_length]
            item_tokens.extend(attr_tokens)
        
        item_id2tokens[int(item_id)] = {'item_tokens': item_tokens}
    
    return item_id2tokens

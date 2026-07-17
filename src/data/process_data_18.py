import os

import torch
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import re
import json
import gzip
import pandas as pd
from collections import defaultdict

class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.label_num = 0

    def get_id(self, label):
        if label not in self.label2id:
            self.label2id[label] = self.label_num
            self.label_num += 1
        return self.label2id[label]
    
def extract_meta_data(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return {
            json.loads(line)['asin']: {
                'title': json.loads(line)['title'],
                'brand': json.loads(line)['brand'],
                'category': ' '.join(json.loads(line)['category'])
            } for line in f
        }

def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

def clean_text(text):
    text = re.sub('<[^<]+?>', '', text)
    return re.sub('\t|\n', ' ', text).strip()

def save_metadata_as_csv(meta_dict, output_path, dataset):
    df = pd.DataFrame.from_dict(meta_dict, orient='index')
    df.index.name = 'item_id:token'
    df.rename(columns={'title': 'title:token', 'category': 'categories:token_seq', 'brand': 'brand:token'}, inplace=True)
    for col in df.columns:
        df[col] = df[col].apply(clean_text)
    df.to_csv(os.path.join(output_path, f'{dataset}.item'), encoding='utf-8', sep='\t')

def save_ratings_as_csv_and_process_sequences(seq_file_path, meta_dict, output_path, dataset, min_len=5, min_item_count=5):
    # filter out items not in meta_dict
    temp_sequences = defaultdict(list)
    with gzip.open(seq_file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if data['asin'] in meta_dict:
                user_id = data['reviewerID']
                item_id = data['asin']
                int_time = data['unixReviewTime']
                temp_sequences[user_id].append((item_id, int_time))
    
    # save the last 30 interactions for each user
    temp_sequences = {user_id: items[-30:] for user_id, items in temp_sequences.items()}

    while True:
        # compute item interaction counts for user interaction sequences with len >= min_len
        item_counts = defaultdict(int)
        for user_id, items in temp_sequences.items():
            if len(items) >= min_len:
                for item_id, int_time in items:
                    item_counts[item_id] += 1

        # keep only items that appear more than min_item_count times
        valid_items = {item_id for item_id, count in item_counts.items() if count >= min_item_count}

        # filter temp_sequences to keep only items in valid_items
        filtered_sequences = defaultdict(list)
        for user_id, items in temp_sequences.items():
            filtered_items = [(item_id, int_time) for item_id, int_time in items if item_id in valid_items]
            if len(filtered_items) >= min_len:
                filtered_sequences[user_id] = filtered_items

        # check if the number of sequences has changed
        if len(filtered_sequences) == len(temp_sequences):
            break
        else:
            temp_sequences = filtered_sequences

    # process the final sequences
    user_field = LabelField()
    s_field = LabelField()
    sequences = defaultdict(list)
    rating_data = []
    for user_id, items in filtered_sequences.items():
        user_label = user_field.get_id(user_id)
        for item_id, int_time in items:
            item_label = s_field.get_id(item_id)
            sequences[user_label].append((item_label, int_time))
            rating_data.append({
                'user_id:token': user_id,
                'item_id:token': item_id,
                'timestamp:float': int_time
            })
    pd.DataFrame(rating_data).to_csv(os.path.join(output_path, f'{dataset}.inter'), encoding='utf-8', sep='\t', index=False)
    del rating_data
    
    meta_dict = {item_token: meta_dict[item_token] for item_token in s_field.label2id.keys() if item_token in meta_dict}

    train_dict, dev_dict, test_dict, index_user_dict = {}, {}, {}, {}

    for idx, (user_label, items) in enumerate(sequences.items()):
        sorted_items = sorted(items, key=lambda x: x[1])
        train_dict[idx] = [item[0] for item in sorted_items[:-2]]
        dev_dict[idx] = [sorted_items[-2][0]]
        test_dict[idx] = [sorted_items[-1][0]]
        index_user_dict[idx] = user_label

    for name, data in [('train', train_dict), ('val', dev_dict), ('test', test_dict), ('meta_data', meta_dict), ('smap', s_field.label2id), ('umap', user_field.label2id)]: #! index_user_dict is not used
        with open(os.path.join(output_path, f'{name}.json'), 'w', encoding='utf8') as f:
            json.dump(data, f)
    print(f'user_num: {len(user_field.label2id):,}, item_num: {len(s_field.label2id):,}, interaction_num: {sum(len(v) for v in train_dict.values()):,}')
    print(f'avg sequence length: {sum([len(seq) for seq in sequences.values()]) / len(sequences)}')
    
    return meta_dict

if __name__ == "__main__":
    base_path = '/data/lgd/HDRec/src/data/raw_data'  # Update this path to your raw data directory
    
    datasets = ['Industrial_and_Scientific', 'Arts_Crafts_and_Sewing', 'Musical_Instruments', 'Office_Products', 'Prime_Pantry', 'Video_Games']

    for dataset in datasets: 
        if dataset != 'Office_Products':
            print(f'\nSkipping {dataset} dataset\n')
            continue
        set_seed(42)

        seq_file_path = f'{base_path}/seq_root/{dataset}_5.json.gz'
        meta_file_path = f'{base_path}/meta_root/meta_{dataset}.json.gz'
        output_path = f'./save/{dataset}/'

        if not os.path.exists(seq_file_path) or not os.path.exists(meta_file_path):
            print(f'\n{seq_file_path} or {meta_file_path} does not exist\n')
            continue

        min_len = 5
        min_item_count = 5
        print(f'\nProcessing {dataset} dataset, user_len>={min_len}, item_time>={min_item_count}')

        meta_dict = extract_meta_data(meta_file_path)
        ensure_directory_exists(output_path)
        meta_dict = save_ratings_as_csv_and_process_sequences(seq_file_path, meta_dict, output_path, dataset, min_len, min_item_count)
        save_metadata_as_csv(meta_dict, output_path, dataset)
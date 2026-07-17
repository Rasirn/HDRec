import torch
import random
import numpy as np
from torch.utils.data import Dataset

class RecDataset(Dataset):
    def __init__(self, user2train, user2val, user2test, tokenized_items, args, mode, tokenizer):

        self.user2train = user2train
        self.user2val   = user2val
        self.user2test  = user2test
        self.tokenized_items = tokenized_items
        self.max_item_num = args.max_item_num
        self.max_token_num = args.max_token_num
        self.mode = mode
        self.tokenizer = tokenizer
        self.args = args
        self.query_token_ids = args.query_token_ids
        
        if self.args.no_prompt:
            user_prompt_ids = []
        else:
            prompt = "Provide the next item's hidden state for each [SEQ]. User's history: "
            user_prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
        
        self.input_ids_prefix = user_prompt_ids
        self.target_ids_prefix = [-100] * len(user_prompt_ids)

        self.item_num = args.item_num
        self.user_num = args.user_num
       
        if self.mode == 'train':
            self.userIDs = list(self.user2train.keys())
        elif self.mode == "valid":
            self.userIDs = list(self.user2val.keys())
        else:
            self.userIDs = list(self.user2test.keys())
        
        if args.debug:
            self.userIDs = self.userIDs[:16]
            self.args.logger.info(f'[Debug: {mode}], user num: {len(self.userIDs)}')
        else:
            self.args.logger.info(f'[Mode: {mode}], user num: {len(self.userIDs)}')
    
    def __len__(self):
        return len(self.userIDs)
            
    def get_train_history(self, userID):
        return self.user2train.get(userID, [])

    def __getitem__(self, index):
        userID = self.userIDs[index]
        # get sequence data, including user's item sequence, labe
        items_seq, label, interactions = self.get_sequence_data(userID)
        
        input_ids, target_ids, item_input_ids, item_target_ids, query_token_indices, item_query_token_indices, id_indices = self.process_sequence(items_seq, label)
        
        assert len(input_ids) == len(target_ids)
        if item_input_ids != None:
            assert len(item_input_ids) == len(item_target_ids)
        
        return input_ids, target_ids, label, userID, item_input_ids, item_target_ids, interactions #, query_token_indices, item_query_token_indices, id_indices

    def get_sequence_data(self, userID):
        if self.mode == 'train':
            items_seq_all = self.user2train[userID]
            if len(items_seq_all) > self.max_item_num:
                start = random.randint(0, len(items_seq_all) - self.max_item_num - 1)
                items_seq = items_seq_all[start:start+self.max_item_num]
                if start+self.max_item_num < len(items_seq_all):
                    label = items_seq_all[start+self.max_item_num]
                else:
                    label = items_seq_all[-1]
            else:
                start = -1
                items_seq = items_seq_all[:-1]
                label = items_seq_all[-1]
            interactions = self.sample_interactions(userID, start) if self.args.late_fusion else None
        elif self.mode == 'valid':
            items_seq = self.user2train[userID]
            if len(items_seq) > self.max_item_num:
                items_seq = items_seq[-self.max_item_num:]
            label = self.user2val[userID][0]
            interactions = self.sample_interactions(userID) if self.args.late_fusion else None
        else:
            items_seq = self.user2train[userID] + self.user2val[userID]
            if len(items_seq) > self.max_item_num:
                items_seq = items_seq[-self.max_item_num:]
            label = self.user2test[userID][0]
            interactions = self.sample_interactions(userID) if self.args.late_fusion else None

        return items_seq[::-1], label, interactions
    
    def sample_negative(self, userID):
        user_history = self.get_train_history(userID)
        while True:
            neg_item_id = random.randint(0, self.item_num - 1)
            if neg_item_id not in user_history:
                return neg_item_id

    def sample_interactions(self, userID, start=-1): 
        user_history = np.array(self.user2train[userID])
        seq_len = len(user_history) - 1
        if seq_len >= self.max_item_num:
            seq_len = self.max_item_num
            if start != -1:
                seq = user_history[start:start+seq_len]
                pos = user_history[start+1:start+seq_len+1]
            else:
                seq = user_history[-seq_len-1:-1]
                pos = user_history[-seq_len:]
        else:
            seq = user_history[:-1]
            pos = user_history[1:]

        neg = np.array([self.sample_negative(userID) for _ in range(seq_len)])
        user = np.array([userID] * seq_len)
        positions = np.array(range(1, seq_len + 1))
        return (user, seq, pos, neg, positions)
    
    def process_sequence(self, items_seq, label):
        items_tokens_list = []
        new_item_seq = []

        count = 0
        for item_id in items_seq:
            if self.args.only_id:
                item_tokens = [-item_id-2]
            else:
                item_tokens = self.tokenized_items[item_id]['item_tokens']
                
                if self.args.early_fusion:
                    item_tokens = [-item_id-2] + item_tokens
            
            count += len(item_tokens)
            if count < self.max_token_num:
                items_tokens_list.append(item_tokens)
                new_item_seq.append(item_id)
            else:
                break
            
        items_tokens_list = items_tokens_list[::-1] # reverse items order
        new_item_seq = new_item_seq[::-1]
        
        target_items = new_item_seq[1:] + [label]
        
        input_ids = []
        target_ids = []
        if self.args.use_gate:
            item_input_ids = []
            item_target_ids = []
            item_query_token_indices = []
        else:
            item_input_ids = None
            item_target_ids = None
            item_query_token_indices = None
        
        input_ids.extend(self.input_ids_prefix)
        target_ids.extend(self.target_ids_prefix)
        
        if self.args.use_gate:
            item_input_ids.extend(self.input_ids_prefix)
            item_target_ids.extend(self.target_ids_prefix)
        
        current_idx = 0 
        prefix_len = len(self.input_ids_prefix)
        id_indices = []
        query_token_indices = []
        
        current_idx += prefix_len

        for idx, item_tokens in enumerate(items_tokens_list):
            input_ids.extend(item_tokens)
            target_ids.extend([-100] * len(item_tokens))
            input_ids, target_ids = self.add_query_tokens(input_ids, target_ids, idx, target_items)
            if self.args.use_gate:
                item_input_ids.extend([-new_item_seq[idx]-2])
                item_target_ids.extend([-100])
                item_input_ids, item_target_ids = self.add_query_tokens(item_input_ids, item_target_ids, idx, target_items)
                item_query_token_indices.append(len(item_input_ids)-1)

            item_len = len(item_tokens)+1
            id_indices.append(current_idx)

            current_idx += item_len
            query_token_indices.append(current_idx-1)

        return input_ids, target_ids, item_input_ids, item_target_ids, query_token_indices, item_query_token_indices, id_indices

    def add_query_tokens(self, input_ids, target_ids, idx, target_items):
        """
        add query tokens to the input and target sequences
        """
        input_ids.extend(self.query_token_ids)
        target_ids.append(target_items[idx])
        
        return input_ids, target_ids

class ItemDataset(Dataset):
    def __init__(self, user2train, tokenized_items, args, tokenizer):
        self.user2train = user2train
        self.tokenized_items = tokenized_items
        self.tokenizer = tokenizer
        self.args = args
        self.alignment_ids = args.alignment_ids
        
        if self.args.no_prompt:
            item_prompt_ids = []
        else:
            prompt = "Provide the item's hidden state for [ALIGN]. Item Description: "
            item_prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
        
        self.input_ids_prefix = item_prompt_ids
        self.target_ids_prefix = [-100] * len(item_prompt_ids)

        self.item_num = args.item_num
        self.user_num = args.user_num
        self.itemIDs = list(range(self.item_num))

        if args.debug:
            self.itemIDs = self.itemIDs[:16]
        self.args.logger.info(f'[Item data], item num: {len(self.itemIDs)}')
    
    def __len__(self):
        return len(self.itemIDs)
    
    def add_query_tokens(self, item_token_ids, target_ids, item_id):
        item_token_ids.extend(self.alignment_ids)
        target_ids.extend([item_id])
        return item_token_ids, target_ids

    def process_item(self, item_token_ids, item_id):

        input_ids = []
        target_ids = []
        if self.args.use_gate:
            item_input_ids = []
            item_target_ids = []
        else:
            item_input_ids = None
            item_target_ids = None

        input_ids.extend(self.input_ids_prefix)
        target_ids.extend(self.target_ids_prefix)
        
        if self.args.early_fusion:
            input_ids.extend([-item_id-2])
            target_ids.extend([-100])
        input_ids.extend(item_token_ids)
        target_ids.extend([-100] * len(item_token_ids))

        input_ids, target_ids = self.add_query_tokens(input_ids, target_ids, item_id)

        if self.args.use_gate:
            item_input_ids.extend(self.input_ids_prefix)
            item_target_ids.extend(self.target_ids_prefix)
            item_input_ids.extend([-item_id-2])
            item_target_ids.extend([-100])
            item_input_ids, item_target_ids = self.add_query_tokens(item_input_ids, item_target_ids, item_id)

        id_idx = len(self.input_ids_prefix)
        return input_ids, target_ids, id_idx, item_input_ids, item_target_ids
    
    def get_train_history(self, userID):
        return self.user2train.get(userID, [])
    
    def __getitem__(self, index):
        item_id = self.itemIDs[index]
        item_token_ids = self.tokenized_items[item_id]['item_tokens']
        input_ids, target_ids, id_idx, item_input_ids, item_target_ids = self.process_item(item_token_ids, item_id)
        label = item_id
        assert len(input_ids) == len(target_ids)
        interactions = self.sample_interactions(item_id) if self.args.late_fusion else None
        return input_ids, target_ids, label, item_id+self.user_num, item_input_ids, item_target_ids, interactions#, [len(input_ids)-1], [len(input_ids)-1], [id_idx]
    
    def sample_negative(self, itemID):
        while True:
            neg_item_id = random.randint(0, self.item_num - 1)
            if neg_item_id != itemID:
                return neg_item_id
        
    def sample_interactions(self, itemID): 
        seq_len = 1
        seq = np.array([itemID])
        pos = np.array([itemID])
        neg = np.array([self.sample_negative(itemID)])
        user = np.array([itemID+self.user_num])
        positions = np.array(range(1, seq_len + 1))
        return (user, seq, pos, neg, positions)
    
class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_seq_ids, target_seq_ids, labels = self.init_tensors(batch)
        user_or_item_ids = torch.LongTensor([sample[3] for sample in batch])
        if self.args.use_gate:
            item_input_ids, item_target_ids = self.init_item_tensors(batch)
        if self.args.late_fusion:
            user_ids, seq_item_ids, pos_item_ids, neg_item_ids, positions  = self.init_tensors_for_CF(batch)
            
        #     if batch[0][8] is not None:
        #         item_query_token_indices = []
        #     else:
        #         item_query_token_indices = None
        # if batch[0][7] is not None:
        #     query_token_indices = []
        #     id_indices = []
        # else:
        #     query_token_indices = None
        #     id_indices = None

        for idx, sample in enumerate(batch):
            num = len(sample[0])
            # Fill in the input sequences and target sequences
            self.fill_sequence_data(input_seq_ids, target_seq_ids, sample, idx, num)
            labels.append(sample[2])

            # if sample[7] is not None:
            #     pad_num = len(input_seq_ids[idx]) - len(sample[0])
            #     query_token_indices.extend([i+pad_num for i in sample[7]])
            #     id_indices.extend([i+pad_num for i in sample[9]])
            if self.args.use_gate:
                self.fill_item_data(item_input_ids, item_target_ids, sample, idx)
                # if sample[8] is not None:
                #     pad_num = len(item_input_ids[idx]) - len(sample[5])
                #     item_query_token_indices.extend([i+pad_num for i in sample[8]])
            if self.args.late_fusion:
                self.fill_interactions(sample[6], user_ids, seq_item_ids, pos_item_ids, neg_item_ids, positions, idx)

        seq_attention_mask = input_seq_ids != self.tokenizer.pad_token_id
        labels = torch.LongTensor(labels)
        user_seq_data = (input_seq_ids, seq_attention_mask, target_seq_ids, labels)
        
        if self.args.use_gate:
            item_seq_mask = item_input_ids != self.tokenizer.pad_token_id
            item_data = (item_input_ids, item_seq_mask, item_target_ids)
        else:
            item_data = (None, None, None)
        
        if self.args.late_fusion:
            interactions = {
                'user_ids': user_ids,
                'seq_item_ids': seq_item_ids,
                'pos_item_ids': pos_item_ids,
                'neg_item_ids': neg_item_ids,
                'positions': positions
            }
        else:
            interactions = {
                'user_ids': None,
                'seq_item_ids': None,
                'pos_item_ids': None,
                'neg_item_ids': None,
                'positions': None
            }

        return {
            'user_seq_data': user_seq_data,
            'user_ids': user_or_item_ids,
            'item_data': item_data,
            'interactions': interactions,
        }

    def init_tensors(self, batch):
        """
        Initialize the required tensors for input sequences, target sequences, labels, and item mask.
        """
        batch_num = len(batch)
        max_seq_len = max([len(sample[0]) for sample in batch])
        input_seq_ids = torch.ones(batch_num, max_seq_len, dtype=torch.long) * self.tokenizer.pad_token_id
        target_seq_ids = torch.ones(batch_num, max_seq_len, dtype=torch.long) * -100
        labels = []
        return input_seq_ids, target_seq_ids, labels

    def fill_sequence_data(self, input_seq_ids, target_seq_ids, sample, idx, num):
        """
        Fill in the input and target sequence data.
        """
        if not self.args.pad_right:
            input_seq_ids[idx, -num:] = torch.LongTensor(sample[0])
            target_seq_ids[idx, -num:] = torch.LongTensor(sample[1])
        else:
            input_seq_ids[idx, :num] = torch.LongTensor(sample[0])
            target_seq_ids[idx, :num] = torch.LongTensor(sample[1])
    
    def init_item_tensors(self, batch):
        batch_num = len(batch)
        max_item_len = max([len(sample[5]) for sample in batch])
        item_input_ids = torch.ones(batch_num, max_item_len, dtype=torch.long) * self.tokenizer.pad_token_id
        item_target_ids = torch.ones(batch_num, max_item_len, dtype=torch.long) * -100
        return item_input_ids, item_target_ids
    
    def fill_item_data(self, item_input_ids, item_target_ids, sample, idx):
        num = len(sample[5])
        if not self.args.pad_right:
            item_input_ids[idx, -num:] = torch.LongTensor(sample[4])
            item_target_ids[idx, -num:] = torch.LongTensor(sample[5])
        else:
            item_input_ids[idx, :num] = torch.LongTensor(sample[4])
            item_target_ids[idx, :num] = torch.LongTensor(sample[5])

    def init_tensors_for_CF(self, batch):
        batch_num = len(batch)
        max_seq_len = max([len(sample[6][0]) for sample in batch])
        user_ids = torch.full((batch_num, max_seq_len), -100, dtype=torch.long)
        seq_item_ids = torch.full((batch_num, max_seq_len), -100, dtype=torch.long)
        pos_item_ids = torch.full((batch_num, max_seq_len), -100, dtype=torch.long)
        neg_item_ids = torch.full((batch_num, max_seq_len), -100, dtype=torch.long)
        positions = torch.full((batch_num, max_seq_len), 0, dtype=torch.long)
        return user_ids, seq_item_ids, pos_item_ids, neg_item_ids, positions
    
    def fill_interactions(self, interactions, user_ids, seq_item_ids, pos_item_ids, neg_item_ids, positions, idx):
        num = len(interactions[0])

        user_ids[idx, -num:] = torch.LongTensor(interactions[0])
        seq_item_ids[idx, -num:] = torch.LongTensor(interactions[1])
        pos_item_ids[idx, -num:] = torch.LongTensor(interactions[2])
        neg_item_ids[idx, -num:] = torch.LongTensor(interactions[3])
        positions[idx, -num:] = torch.LongTensor(interactions[4])

import torch
import torch.nn as nn
import torch.nn.functional as F

class PtuningEmbedding(nn.Module):
    def __init__(self, embed_tokens, shared_item_embedding, shared_user_embedding, item_num):
        super().__init__()
        self.embeddings = embed_tokens
        self.item_embedding = shared_item_embedding
        self.user_embedding = shared_user_embedding
        if shared_item_embedding is not None:
            # self.projection = nn.Linear(shared_item_embedding.embedding_dim, embed_tokens.embedding_dim)
            self.projection = nn.Sequential(
                nn.Linear(shared_item_embedding.embedding_dim, embed_tokens.embedding_dim // 2),
                # nn.Dropout(0.5),
                nn.SiLU(),
                nn.Linear(embed_tokens.embedding_dim // 2, embed_tokens.embedding_dim)
            )
        else:
            self.projection = None
        self.item_num = item_num
        self.vocab_size = embed_tokens.num_embeddings
        self.new_emb = nn.Embedding(2, embed_tokens.embedding_dim)
        with torch.no_grad():
            self.new_emb.weight.fill_(0.0)

    def forward(self, input_ids):
        # Create masks for the three cases
        item_emb_mask = (input_ids < 0) & (input_ids >= -self.item_num-1)
        input_ids[item_emb_mask] = -input_ids[item_emb_mask] - 1
        # user_emb_mask = input_ids < -self.item_num-1
        # input_ids[user_emb_mask] = -input_ids[user_emb_mask] - self.item_num - 2
        token_embeddings = self.embeddings(input_ids.clamp(min=0, max=self.vocab_size - 1))
        
        # Process item_embedding mask
        item_positions = torch.nonzero(item_emb_mask, as_tuple=True)
        item_ids = input_ids[item_positions]  # Convert negative IDs to item IDs
        if item_ids.size(0) > 0:
            item_embeddings = self.item_embedding(item_ids)
            item_embeddings = self.projection(item_embeddings)
            token_embeddings[item_emb_mask] = item_embeddings

        # # Process user_embedding mask
        # user_positions = torch.nonzero(user_emb_mask, as_tuple=True)
        # user_ids = input_ids[user_positions]  # Adjust IDs for user_embedding
        # if user_ids.size(0) > 0:
        #     user_embeddings = self.user_embedding(user_ids)
        #     user_embeddings = self.projection(user_embeddings)
        #     token_embeddings[user_emb_mask] = user_embeddings

        new_emb_mask = input_ids >= self.vocab_size - 2
        new_positions = torch.nonzero(new_emb_mask, as_tuple=True)
        new_ids = input_ids[new_positions]
        if new_ids.size(0) > 0:
            new_embeddings = self.new_emb(new_ids - self.vocab_size + 2)
            token_embeddings[new_emb_mask] += new_embeddings

        return token_embeddings

def confidence_fusion(logits_text, logits_id, temperature=1.0, alpha=0.5, fusion_type='text'):
    if fusion_type == 'text':
        with torch.no_grad():
            logits_id_avg = logits_id.mean(dim=-1, keepdim=True)
            logits_id_truth = torch.sigmoid((logits_id - logits_id_avg) / temperature)
            logits_id_min = logits_id.min(dim=-1, keepdim=True).values
        logits_id = logits_id - logits_id_min + 1e-8
        enhancement = alpha * logits_id * logits_id_truth
        return logits_text + enhancement
    elif fusion_type == 'id':
        with torch.no_grad():
            logits_text_avg = logits_text.mean(dim=-1, keepdim=True)
            logits_text_truth = torch.sigmoid((logits_text - logits_text_avg) / temperature)
            logits_text_min = logits_text.min(dim=-1, keepdim=True).values
        logits_text = logits_text - logits_text_min + 1e-8
        enhancement = alpha * logits_text * logits_text_truth
        return logits_id + enhancement
    elif fusion_type == 'both':
        with torch.no_grad():
            logits_text_avg = logits_text.mean(dim=-1, keepdim=True)
            logits_text_truth = torch.sigmoid((logits_text - logits_text_avg) / temperature)
            logits_id_avg = logits_id.mean(dim=-1, keepdim=True)
            logits_id_truth = torch.sigmoid((logits_id - logits_id_avg) / temperature)
        return logits_text * logits_text_truth + logits_id * logits_id_truth

def symmetric_kl(logits1, logits2, temperature=1.0):
    """
    Compute the symmetric KL divergence between two sets of logits.
    """
    p_log = F.log_softmax(logits1 / temperature, dim=-1)
    q_log = F.log_softmax(logits2 / temperature, dim=-1)
    p_prob = F.softmax(logits1 / temperature, dim=-1)
    q_prob = F.softmax(logits2 / temperature, dim=-1)
    kl_pq = F.kl_div(p_log, q_prob, reduction='batchmean')
    kl_qp = F.kl_div(q_log, p_prob, reduction='batchmean')
    return (kl_pq + kl_qp) / 2

def kl_divergence(teacher_logits, student_logits, temperature=1.0):
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    student_log = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log, teacher_prob, reduction='batchmean')

def get_logits_label(logits, labels):
    pooled_logits_list = []
    target_labels_list = []
    index_list = []
    idx = 0
    mask = labels != -100
    for b in range(labels.size(0)):
        pos = torch.where(mask[b])[0]
        pooled_logits_list.append(logits[b, pos])
        target_labels_list.append(labels[b, pos])
        index_list.extend([idx] * pos.size(0))
        idx += 1
    pooled_logits = torch.cat(pooled_logits_list, dim=0)
    target_labels_list = torch.cat(target_labels_list, dim=0)
    index_list = torch.tensor(index_list, dtype=torch.long).to(pooled_logits.device)
    return pooled_logits, target_labels_list, index_list

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
class SASRecBackbone(nn.Module):
    def __init__(self, hidden_size, num_heads, num_blocks, dropout):
        super().__init__()
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  torch.nn.MultiheadAttention(hidden_size,
                                                            num_heads,
                                                            dropout)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(hidden_size, dropout)
            self.forward_layers.append(new_fwd_layer)
    def forward(self, seqs, log_seqs):
        #timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

class SASRec(nn.Module):
    def __init__(self, user_num, item_num, factors, max_item_num, shared_item_embedding, num_heads=2, num_blocks=1, dropout=0.5):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = factors
        if shared_item_embedding is not None:
            self.item_emb = shared_item_embedding
            self.filter_init_modules = ['item_emb.weight']
        else:
            self.filter_init_modules = []
            self.item_emb = nn.Embedding(item_num+1, factors)
        self.pos_emb = nn.Embedding(max_item_num+10, factors)
        self.emb_dropout = torch.nn.Dropout(p=dropout)
        self.backbone = SASRecBackbone(factors, num_heads, num_blocks, dropout)
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            try:
                flag = True     # denote initialize this param 
                for fm in self.filter_init_modules:
                    if fm in name:  # if the param in filter_modules, do not initialize
                        flag = False    
                if flag:
                    nn.init.xavier_normal_(param.data)
            except:
                pass
    
    def _get_embedding(self, log_seqs):
        item_seq_emb = self.item_emb(log_seqs)
        return item_seq_emb

    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats

    def get_item_emb(self):
        return self.item_emb.weight
    
    def get_user_emb(self, seq, positions):
        log_feats = self.log2feats(seq, positions) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        return final_feat
    
    def get_logits_label(self, seq_item_ids, pos_item_ids, positions):
        seq_item_ids = self.preprocessing(seq_item_ids)
        pos_item_ids = self.preprocessing(pos_item_ids)
        log_feats = self.log2feats(seq_item_ids, positions)
        embs = self.get_item_emb()
        logits = torch.matmul(log_feats, embs.T)
        return logits, pos_item_ids
    
    def calculate_loss(self, seq_item_ids, pos_item_ids, positions):
        seq_item_ids = self.preprocessing(seq_item_ids)
        pos_item_ids = self.preprocessing(pos_item_ids)
        log_feats = self.log2feats(seq_item_ids, positions)
        embs = self.get_item_emb()
        logits = torch.matmul(log_feats, embs.T)
        logits = logits.view(-1, logits.size(-1))  # logits shape: [128*50, 5194]
        pos_labels = pos_item_ids.view(-1).long()  # pos_labels shape: [128*50]
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, pos_labels)
        return loss

    def preprocessing(self, item_ids):
        mask = item_ids != -100
        item_ids = torch.where(mask, item_ids + 1, torch.tensor(0, dtype=item_ids.dtype))
        return item_ids
    

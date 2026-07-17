import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig, Cache, DynamicCache
LLAMA_INPUTS_DOCSTRING = ""
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast 
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import add_start_docstrings_to_model_forward, logging
logger = logging.get_logger(__name__)

from models.modules import PtuningEmbedding, get_logits_label

class LlamaModel_rec(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.not_change = False

        self.dropouts = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.hidden_dropout != 0 and i % config.hd_frequency == config.hd_frequency - 1:
                self.dropouts.append(nn.Dropout(config.hidden_dropout))
            else:
                self.dropouts.append(None)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        from transformers.models.llama.modeling_llama import create_causal_mask

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer, layer_dropout in zip(self.layers[: self.config.num_hidden_layers], self.dropouts):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_dropout is not None:
                hidden_states = layer_dropout(hidden_states)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class llama_rec(LlamaModel_rec):
    def __init__(self, config):
        super().__init__(config)
        model_prefix = "temp/"+self.config._name_or_path.replace("/", "-")
        self.config.decoder_temp_path = model_prefix
        self.late_fusion = False

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model = super().from_pretrained(model_name_or_path, **kwargs)
        model.set_linear(model.config.linear_dim)
        return model
    
    def set_small_model(self):
        import os
        if self.config.method_of_preference in ['SASRec', 'GRU4Rec']:
            if self.config.late_fusion:
                from models.modules import SASRec
                self.small_model = SASRec(self.config.user_num, self.config.item_num, self.config.preference_dim, 50, self.shared_item_embedding)
                self.late_fusion = True
            if not self.config.late_fusion or self.config.late_fusion_load:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f"cuda:{local_rank}")
                state_dict = torch.load(f'./temp/{self.config.method_of_preference}/{self.config.dataset}-{self.config.preference_dim}.pth', map_location=device)
                self.shared_item_embedding.weight.data = state_dict
        else:
            logger.info("The method of preference is not SASRec, so the item embedding is randomly initialized.")
    
    def set_emb(self):
        if self.config.use_small_model:
            self.shared_item_embedding = nn.Embedding(self.config.item_num+1, self.config.preference_dim)
        else:
            self.shared_item_embedding = None
        self.shared_user_embedding = None
        self.embed_tokens = PtuningEmbedding(self.embed_tokens, self.shared_item_embedding, self.shared_user_embedding, self.config.item_num)

    def set_linear(self, linear_dim):
        
        hidden_size = self.config.hidden_size
        # self.score_classify = nn.Linear(hidden_size, linear_dim, bias=False) #!
        self.score_classify = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(p=self.config.score_dropout),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, linear_dim, bias=False),
        )
        if self.config.use_two_score:
            self.score_classify2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Dropout(p=self.config.score_dropout),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, linear_dim, bias=False),
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        is_text=True,
        interactions=None,
        **kwargs
    ):
        if self.late_fusion and not is_text:
            seq_item_ids, pos_item_ids, positions = interactions['seq_item_ids'], interactions['pos_item_ids'], interactions['positions']
            logits, pos_item_ids = self.small_model.get_logits_label(seq_item_ids, pos_item_ids, positions)
            # logits remove padding
            if labels is None:
                return_logits = logits[:, -1]
                return_logits = return_logits[:, 1:]
                return return_logits, None
            logits = logits.view(-1, logits.size(-1))
            pos_item_ids = pos_item_ids.view(-1).long()
            mask = pos_item_ids != 0
            return_logits = logits[mask].view(-1, self.config.linear_dim+1)
            return_labels = pos_item_ids[mask].view(-1)
            return return_logits, return_labels
        else:
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            pooled_output = outputs.last_hidden_state
            if not self.config.use_two_score:
                pooled_logits = self.score_classify(pooled_output)
            else:
                if is_text:
                    pooled_logits = self.score_classify(pooled_output)
                else:
                    pooled_logits = self.score_classify2(pooled_output)

            if labels is None:
                return pooled_logits[:, -1], None
            return_logits, return_labels, _ = get_logits_label(pooled_logits, labels)
            return return_logits, return_labels

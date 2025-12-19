# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu

from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .diff_fake_quant_mx import diffLinear, QdiffLinear, diffEmbedding, diffLayerNorm

# from transformers.models.roberta.configuration_roberta import RobertaConfig

import loralib as lora

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]

#新增 dataclass 用在encoder的forward_delta当中
@dataclass
class BaseModelOutputWithDelta(ModelOutput):
    last_hidden_state: torch.Tensor = None
    diff_last_hidden_state: torch.Tensor = None

    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    hidden_states: Optional[Tuple[torch.Tensor]] = None
    diff_hidden_states: Optional[Tuple[torch.Tensor]] = None

    attentions: Optional[Tuple[torch.Tensor]] = None
    diff_attentions: Optional[Tuple[torch.Tensor]] = None

    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    diff_cross_attentions: Optional[Tuple[torch.Tensor]] = None
    
@dataclass
class BaseModelOutputWithPoolingAndDelta(ModelOutput):
    last_hidden_state: torch.Tensor = None
    diff_last_hidden_state: torch.Tensor = None

    pooler_output: torch.Tensor = None
    diff_pooler_output: torch.Tensor = None

    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    hidden_states: Optional[Tuple[torch.Tensor]] = None
    diff_hidden_states: Optional[Tuple[torch.Tensor]] = None

    attentions: Optional[Tuple[torch.Tensor]] = None
    diff_attentions: Optional[Tuple[torch.Tensor]] = None

    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    diff_cross_attentions: Optional[Tuple[torch.Tensor]] = None
    
@dataclass
class CausalLMOutputWithDelta(ModelOutput):
    loss_base: Optional[torch.Tensor] = None
    loss_perturbed: Optional[torch.Tensor] = None

    logits: torch.Tensor = None                     # base logits
    perturbed_logits: torch.Tensor = None           # perturbed logits

    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    hidden_states: Optional[Tuple[torch.Tensor]] = None
    perturbed_hidden_states: Optional[Tuple[torch.Tensor]] = None

    attentions: Optional[Tuple[torch.Tensor]] = None
    perturbed_attentions: Optional[Tuple[torch.Tensor]] = None

    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    perturbed_cross_attentions: Optional[Tuple[torch.Tensor]] = None
    
@dataclass
class MaskedLMOutputWithDelta(ModelOutput):
    loss_base: Optional[torch.Tensor] = None
    loss_perturbed: Optional[torch.Tensor] = None

    logits: torch.Tensor = None                     # base logits
    perturbed_logits: torch.Tensor = None           # perturbed logits

    hidden_states: Optional[Tuple[torch.Tensor]] = None
    perturbed_hidden_states: Optional[Tuple[torch.Tensor]] = None

    attentions: Optional[Tuple[torch.Tensor]] = None
    perturbed_attentions: Optional[Tuple[torch.Tensor]] = None

from transformers import BertConfig
class RobertaConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.
    Examples::
        >>> from transformers import RobertaConfig, RobertaModel
        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()
        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, apply_lora=False, lora_r=None, lora_alpha=None, 
                 apply_forward_delta = False, 
                 enable_x = True, enable_diffx = True, enable_w = True, enable_diffw = True,
                 mx_w_elem_format=None, mx_a_elem_format=None, mx_diffw_elem_format=None, mx_diffa_elem_format=None,
                 uv_provider=None, z_provider=None,
                 **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.apply_lora = apply_lora
        self.lora_r = lora_r 
        self.lora_alpha = lora_alpha
        
        # forward_delta
        self.apply_forward_delta = apply_forward_delta
        self.uv_provider = uv_provider
        self.z_provider = z_provider
        self.enable_x = enable_x
        self.enable_diffx = enable_diffx
        self.enable_w = enable_w
        self.enable_diffw = enable_diffw
        self.mx_w_elem_format = mx_w_elem_format
        self.mx_a_elem_format = mx_a_elem_format
        self.mx_diffw_elem_format = mx_diffw_elem_format
        self.mx_diffa_elem_format = mx_diffa_elem_format
        

class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        if getattr(config, "apply_forward_delta", False):
            self.word_embeddings = diffEmbedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=config.pad_token_id, layer_name='roberta.embeddings.word_embeddings', uv_provider=config.uv_provider)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        self.padding_idx = config.pad_token_id
        if getattr(config, "apply_forward_delta", False):
            self.position_embeddings = diffEmbedding(num_embeddings=config.max_position_embeddings, embedding_dim=config.hidden_size, padding_idx=config.pad_token_id, layer_name='roberta.embeddings.position_embeddings', uv_provider=config.uv_provider)
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id)
        
        if getattr(config, "apply_forward_delta", False):
            self.token_type_embeddings = diffEmbedding(num_embeddings=config.type_vocab_size, embedding_dim=config.hidden_size, layer_name='roberta.embeddings.token_type_embeddings', uv_provider=config.uv_provider)
        else:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        if getattr(config, "apply_forward_delta", False):
            self.LayerNorm = diffLayerNorm(config.hidden_size, eps=config.layer_norm_eps, layer_name="roberta.embeddings.LayerNorm", z_provider=config.z_provider)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # self.padding_idx = config.pad_token_id
        # self.position_embeddings = nn.Embedding(
            # config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        # )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def forward_delta(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        # word_embeddings: base + delta 
        if inputs_embeds is None:
            inputs_embeds, diff_inputs_embeds = self.word_embeddings.forward_delta(input_ids)
        else:
            diff_inputs_embeds = torch.zeros_like(inputs_embeds)

        # token_type_embeddings: base + delta 
        token_type_embeddings, diff_token_type_embeddings = self.token_type_embeddings.forward_delta(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        diff_embeddings = diff_inputs_embeds + diff_token_type_embeddings

        # position_embeddings: only in absolute
        if self.position_embedding_type == "absolute":
            position_embeddings, diff_position_embeddings = self.position_embeddings.forward_delta(position_ids)

            embeddings = embeddings + position_embeddings
            diff_embeddings = diff_embeddings + diff_position_embeddings

        # diffLayerNorm 
        embeddings, diff_embeddings = self.LayerNorm.forward_delta(embeddings, diff_embeddings,)

        embeddings_perturbed = self.dropout(embeddings + diff_embeddings)
        embeddings = self.dropout(embeddings)
        diff_embeddings = embeddings_perturbed - embeddings

        return embeddings, diff_embeddings


    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
class RobertaSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        use_diff = getattr(config, "apply_forward_delta", False)
        prefix = f"roberta.encoder.layer.{layer_idx}.attention.self"
        
        # query
        if use_diff:
            self.query = QdiffLinear(enable_x = config.enable_x, enable_diffx = config.enable_diffx, enable_w = config.enable_w, enable_diffw = config.enable_diffw,
                                     layer_name = f"{prefix}.query", 
                                     in_features = config.hidden_size, out_features = self.all_head_size, bias=True,
                                     mx_w_elem_format=config.mx_w_elem_format, mx_a_elem_format=config.mx_a_elem_format, mx_diffw_elem_format=config.mx_diffw_elem_format, mx_diffa_elem_format=config.mx_diffa_elem_format,
                                     uv_provider=config.uv_provider, z_provider=config.z_provider,)
        elif getattr(config, "apply_lora", False):
            self.query = lora.Linear(config.hidden_size, self.all_head_size, config.lora_r, lora_alpha=config.lora_alpha)
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        # key
        if use_diff:
            self.key = QdiffLinear(enable_x = config.enable_x, enable_diffx = config.enable_diffx, enable_w = config.enable_w, enable_diffw = config.enable_diffw,
                                     layer_name = f"{prefix}.key", 
                                     in_features = config.hidden_size, out_features = self.all_head_size, bias=True,
                                     mx_w_elem_format=config.mx_w_elem_format, mx_a_elem_format=config.mx_a_elem_format, mx_diffw_elem_format=config.mx_diffw_elem_format, mx_diffa_elem_format=config.mx_diffa_elem_format,
                                     uv_provider=config.uv_provider, z_provider=config.z_provider,)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
        
        # value
        if use_diff:
            self.value = QdiffLinear(enable_x = config.enable_x, enable_diffx = config.enable_diffx, enable_w = config.enable_w, enable_diffw = config.enable_diffw,
                                     layer_name = f"{prefix}.value", 
                                     in_features = config.hidden_size, out_features = self.all_head_size, bias=True,
                                     mx_w_elem_format=config.mx_w_elem_format, mx_a_elem_format=config.mx_a_elem_format, mx_diffw_elem_format=config.mx_diffw_elem_format, mx_diffa_elem_format=config.mx_diffa_elem_format,
                                     uv_provider=config.uv_provider, z_provider=config.z_provider,)
        elif getattr(config, "apply_lora", False):
            self.value = lora.Linear(config.hidden_size, self.all_head_size, config.lora_r, lora_alpha=config.lora_alpha)
        else:
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_key_value: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder or output_key_value:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder or output_key_value:
            outputs = outputs + (past_key_value,)
        return outputs
    
    def forward_delta(
        self,
        hidden_states: torch.Tensor,
        diff_hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_key_value: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """
        Delta 版本的 self-attention:
        - hidden_states: 原始输入
        - diff_hidden_states: 对应的 delta 输入（形状和 hidden_states 一样）
        return:
            if output_attentions=False:
                (context_layer, diff_context_layer)
            if output_attentions=True:
                (context_layer, diff_context_layer, attention_probs, diff_attention_probs)
        """
        # 目前只支持 encoder 自注意力（最常见的 RoBERTa 场景）
        if encoder_hidden_states is not None or past_key_value is not None:
            raise NotImplementedError("forward_delta currently only supports encoder self-attention "
                                      "(no encoder_hidden_states, no past_key_value).")
        
        if hasattr(self.query, "forward_delta"):
            mixed_query_layer, diff_mixed_query_layer = self.query.forward_delta(
                hidden_states, diff_hidden_states
            )
        else:
            mixed_query_layer = self.query(hidden_states)
            diff_mixed_query_layer = torch.zeros_like(mixed_query_layer)

        if hasattr(self.key, "forward_delta"):
            key_proj, diff_key_proj = self.key.forward_delta(hidden_states, diff_hidden_states)
        else:
            key_proj = self.key(hidden_states)
            diff_key_proj = torch.zeros_like(key_proj)

        if hasattr(self.value, "forward_delta"):
            value_proj, diff_value_proj = self.value.forward_delta(hidden_states, diff_hidden_states)
        else:
            value_proj = self.value(hidden_states)
            diff_value_proj = torch.zeros_like(value_proj)
            
        query_layer = self.transpose_for_scores(mixed_query_layer)
        diff_query_layer = self.transpose_for_scores(diff_mixed_query_layer)

        key_layer = self.transpose_for_scores(key_proj)
        diff_key_layer = self.transpose_for_scores(diff_key_proj)
        
        value_layer = self.transpose_for_scores(value_proj)
        diff_value_layer = self.transpose_for_scores(diff_value_proj)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
       
        attention_scores_perturbed = torch.matmul(
            query_layer + diff_query_layer,
            (key_layer + diff_key_layer).transpose(-1, -2),
        )
        diff_attention_scores = attention_scores_perturbed - attention_scores
        
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            
            # only encoder: no use_cache
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
            
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores

                # diff
                relative_position_scores_perturbed = torch.einsum(
                    "bhld,lrd->bhlr", query_layer + diff_query_layer, positional_embedding
                )
                diff_relative_scores = relative_position_scores_perturbed - relative_position_scores
                diff_attention_scores = diff_attention_scores + diff_relative_scores

            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

                # diff
                relative_position_scores_query_perturbed = torch.einsum(
                    "bhld,lrd->bhlr", query_layer + diff_query_layer, positional_embedding
                )
                relative_position_scores_key_perturbed = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer + diff_key_layer, positional_embedding
                )
                diff_relative_scores = (
                    (relative_position_scores_query_perturbed + relative_position_scores_key_perturbed)
                    - (relative_position_scores_query + relative_position_scores_key)
                )
                diff_attention_scores = diff_attention_scores + diff_relative_scores
                
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        diff_attention_scores = diff_attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            # diff_attention_scores = (S + M + ΔS) - (S + M) = ΔS 无需加attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs_perturbed = nn.functional.softmax(
            attention_scores + diff_attention_scores,
            dim=-1,
        )
        # diff_attention_probs = attention_probs_perturbed - attention_probs
        
        attention_probs_perturbed = self.dropout(attention_probs_perturbed)
        attention_probs = self.dropout(attention_probs)
        diff_attention_probs = attention_probs_perturbed - attention_probs

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            diff_attention_probs = diff_attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_perturbed = torch.matmul(
            attention_probs + diff_attention_probs,
            value_layer + diff_value_layer,
        )
        diff_context_layer = context_layer_perturbed - context_layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        diff_context_layer = diff_context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        diff_context_layer = diff_context_layer.view(new_context_layer_shape)

        if output_attentions:
            outputs = (context_layer, diff_context_layer, attention_probs, diff_attention_probs)
        else:
            outputs = (context_layer, diff_context_layer)

        # 目前不支持 cache 的 delta，统一不返回 past_key_value
        if self.is_decoder or output_key_value:
            raise NotImplementedError("forward_delta does not yet support returning key/value cache.")

        return outputs



# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class RobertaSelfOutput(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        use_diff = getattr(config, "apply_forward_delta", False)
        prefix = f"roberta.encoder.layer.{layer_idx}.attention.output"
        if use_diff:
            self.dense = QdiffLinear(enable_x = config.enable_x, enable_diffx = config.enable_diffx, enable_w = config.enable_w, enable_diffw = config.enable_diffw,
                                     layer_name = f"{prefix}.dense", 
                                     in_features = config.hidden_size, out_features = config.hidden_size, bias=True,
                                     mx_w_elem_format=config.mx_w_elem_format, mx_a_elem_format=config.mx_a_elem_format, mx_diffw_elem_format=config.mx_diffw_elem_format, mx_diffa_elem_format=config.mx_diffa_elem_format,
                                     uv_provider=config.uv_provider, z_provider=config.z_provider,)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if use_diff:
            self.LayerNorm = diffLayerNorm(config.hidden_size, eps=config.layer_norm_eps, layer_name=f"{prefix}.LayerNorm", z_provider=config.z_provider)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def forward_delta(self, hidden_states: torch.Tensor, diff_hidden_states: torch.Tensor, input_tensor: torch.Tensor, diff_input_tensor: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.dense, "forward_delta"):
            dense_out, diff_dense_out = self.dense.forward_delta(
                hidden_states,
                diff_hidden_states,
            )
        else:
            dense_out = self.dense(hidden_states)
            diff_dense_out = self.dense(diff_hidden_states)

        dense_out_perturbed = self.dropout(dense_out + diff_dense_out)
        dense_out = self.dropout(dense_out)
        diff_dense_out = dense_out_perturbed - dense_out

        # base:
        residual = dense_out + input_tensor
        # delta: dy = d(dense_out) + d(input_tensor)
        diff_residual = diff_dense_out + diff_input_tensor

        if hasattr(self.LayerNorm, "forward_delta"):
            output, diff_output = self.LayerNorm.forward_delta(
                residual,
                diff_residual,
            )
        else:
            base_out = self.LayerNorm(residual)
            perturbed_out = self.LayerNorm(residual + diff_residual)
            diff_output = perturbed_out - base_out
            output = base_out

        return output, diff_output


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class RobertaAttention(nn.Module):
    def __init__(self, config, layer_idx, position_embedding_type=None):
        super().__init__()
        self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type, layer_idx = layer_idx)
        self.output = RobertaSelfOutput(config, layer_idx = layer_idx)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_key_value: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            output_key_value=output_key_value
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    
    def forward_delta(
        self,
        hidden_states: torch.Tensor,
        diff_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_key_value: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self.forward_delta(
            hidden_states = hidden_states,
            diff_hidden_states = diff_hidden_states,
            attention_mask = attention_mask,
            head_mask = head_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            output_key_value=output_key_value
        )
        context_layer, diff_context_layer = self_outputs[:2]
        attention_output, diff_attention_output = self.output.forward_delta(
            hidden_states=context_layer,
            diff_hidden_states=diff_context_layer,
            input_tensor=hidden_states,           # 残差
            diff_input_tensor=diff_hidden_states, # 残差的 delta
        )
        outputs = (attention_output, diff_attention_output) + self_outputs[2:]
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config,layer_idx):
        super().__init__()
        use_diff = getattr(config, "apply_forward_delta", False)
        prefix = f"roberta.encoder.layer.{layer_idx}.intermediate"
        if use_diff:
            self.dense = QdiffLinear(enable_x = config.enable_x, enable_diffx = config.enable_diffx, enable_w = config.enable_w, enable_diffw = config.enable_diffw,
                                     layer_name = f"{prefix}.dense", 
                                     in_features = config.hidden_size, out_features = config.intermediate_size, bias=True,
                                     mx_w_elem_format=config.mx_w_elem_format, mx_a_elem_format=config.mx_a_elem_format, mx_diffw_elem_format=config.mx_diffw_elem_format, mx_diffa_elem_format=config.mx_diffa_elem_format,
                                     uv_provider=config.uv_provider, z_provider=config.z_provider,)
        else:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
            
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
    def forward_delta(self, hidden_states: torch.Tensor, diff_hidden_states) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.dense, "forward_delta"):
            hidden, diff_hidden = self.dense.forward_delta(
                hidden_states,
                diff_hidden_states,
            )
        else:
            hidden = self.dense(hidden_states)
            diff_hidden = self.dense(diff_hidden_states)

        activated = self.intermediate_act_fn(hidden)
        activated_perturbed = self.intermediate_act_fn(hidden + diff_hidden)
        diff_activated = activated_perturbed - activated

        return activated, diff_activated


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        use_diff =  getattr(config, "apply_forward_delta", False)
        prefix = f"roberta.encoder.layer.{layer_idx}.output"
        if use_diff:
            self.dense = QdiffLinear(enable_x = config.enable_x, enable_diffx = config.enable_diffx, enable_w = config.enable_w, enable_diffw = config.enable_diffw,
                                     layer_name = f"{prefix}.dense", 
                                     in_features = config.intermediate_size, out_features = config.hidden_size, bias=True,
                                     mx_w_elem_format=config.mx_w_elem_format, mx_a_elem_format=config.mx_a_elem_format, mx_diffw_elem_format=config.mx_diffw_elem_format, mx_diffa_elem_format=config.mx_diffa_elem_format,
                                     uv_provider=config.uv_provider, z_provider=config.z_provider,)
        else:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        if use_diff:
            self.LayerNorm = diffLayerNorm(config.hidden_size, eps=config.layer_norm_eps, layer_name=f"{prefix}.LayerNorm", z_provider=config.z_provider)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def forward_delta(self, hidden_states: torch.Tensor, diff_hidden_states: torch.Tensor, input_tensor: torch.Tensor, diff_input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.dense, "forward_delta"):
            dense_out, diff_dense_out = self.dense.forward_delta(
                hidden_states,
                diff_hidden_states,
            )
        else:
            dense_out = self.dense(hidden_states)
            diff_dense_out = self.dense(diff_hidden_states)
        
        dense_out_perturbed = self.dropout(dense_out + diff_dense_out)
        dense_out = self.dropout(dense_out)
        diff_dense_out = dense_out_perturbed - dense_out

        residual = dense_out + input_tensor
        diff_residual = diff_dense_out + diff_input_tensor

        if hasattr(self.LayerNorm, "forward_delta"):
            output, diff_output = self.LayerNorm.forward_delta(
                residual,
                diff_residual,
            )
        else:
            base_out = self.LayerNorm(residual)
            perturbed_out = self.LayerNorm(residual + diff_residual)
            diff_output = perturbed_out - base_out
            output = base_out

        return output, diff_output


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        self.layer_idx = layer_idx
        
        self.attention = RobertaAttention(config, layer_idx = layer_idx)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute", layer_idx = layer_idx)
        self.intermediate = RobertaIntermediate(config, layer_idx = layer_idx)
        self.output = RobertaOutput(config, layer_idx = layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_key_value: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            output_key_value=output_key_value
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder or output_key_value:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder or output_key_value:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def forward_delta(
        self,
        hidden_states: torch.Tensor,
        diff_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_key_value: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """
        forward 的 delta 版本，目前只支持：
          - encoder 自注意力（RoBERTa 标准用法）
          - 不支持 cross-attention / decoder / KV cache
        """
        if (self.is_decoder or self.add_cross_attention or encoder_hidden_states is not None or past_key_value is not None):
            raise NotImplementedError(
                "RobertaLayer.forward_delta currently only supports encoder self-attention "
                "(no decoder, no cross-attention, no past_key_value)."
            )

        self_attention_outputs = self.attention.forward_delta(
            hidden_states=hidden_states,
            diff_hidden_states=diff_hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_key_value=output_key_value,
        )
        # output_attentions=False: (attn_out, diff_attn_out)
        # output_attentions=True:  (attn_out, diff_attn_out, attn_probs, diff_attn_probs)
        attention_output, diff_attention_output = self_attention_outputs[:2]
        extra_outputs = self_attention_outputs[2:]  # 可能为空，也可能是 (attn_probs, diff_attn_probs)

        # FFN（intermediate + output）的 delta
        # 这里不再做 chunking，直接一次性算完整序列；
        # RoBERTa 默认 chunk_size_feed_forward=0，本来就不 chunk
        intermediate_output, diff_intermediate_output = self.intermediate.forward_delta(
            attention_output,
            diff_attention_output,
        )

        layer_output, diff_layer_output = self.output.forward_delta(
            hidden_states=intermediate_output,
            diff_hidden_states=diff_intermediate_output,
            input_tensor=attention_output,
            diff_input_tensor=diff_attention_output,
        )

        outputs = (layer_output, diff_layer_output) + tuple(extra_outputs)
        # 不支持 cache，所以没有 present_key_value
        return outputs



# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 新增layer_idx以传入Encoder层下diff层的layername
        self.layer = nn.ModuleList([RobertaLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    output_key_value=use_cache
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        
    def forward_delta(
        self,
        hidden_states: torch.Tensor,
        diff_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithDelta]:
        """
        在 RoBERTa encoder 标准路径下（encoder-only、自注意力）的 delta 版本：
        """
        # 检查：只针对 encoder-only RoBERTa
        if self.config.is_decoder or self.config.add_cross_attention:
            raise NotImplementedError(
                "RobertaEncoder.forward_delta 目前只支持 encoder-only RoBERTa (is_decoder=False, add_cross_attention=False)."
            )
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise NotImplementedError(
                "forward_delta 默认路径下不支持 encoder_hidden_states / encoder_attention_mask（encoder-only RoBERTa 用不到）。"
            )
        if past_key_values is not None:
            raise NotImplementedError(
                "forward_delta 目前不支持 past_key_values（encoder-only RoBERTa 一般不用 cache）。"
            )

        all_hidden_states = () if output_hidden_states else None
        all_diff_hidden_states = () if output_hidden_states else None

        all_self_attentions = () if output_attentions else None
        all_diff_self_attentions = () if output_attentions else None

        # encoder-only 场景几乎不用 cache，这里只是结构对齐
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False` in forward_delta..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None  # RoBERTa encoder 下一般是 None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                all_diff_hidden_states = all_diff_hidden_states + (diff_hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None
            if layer_past_key_value is not None:
                raise NotImplementedError(
                    "forward_delta 暂不支持 per-layer past_key_value（encoder-only RoBERTa 一般不用）。"
                )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        hs, diff_hs, attn_mask, layer_hm, enc_hs, enc_attn_mask = inputs
                        return module.forward_delta(
                            hs,
                            diff_hs,
                            attn_mask,
                            layer_hm,
                            enc_hs,
                            enc_attn_mask,
                            layer_past_key_value,
                            output_attentions,
                            output_key_value=use_cache,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    diff_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module.forward_delta(
                    hidden_states,
                    diff_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    layer_past_key_value,
                    output_attentions,
                    output_key_value=use_cache,
                )

            hidden_states, diff_hidden_states = layer_outputs[:2]

            # cache（encoder-only 下一般不用）
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # output_attentions=False: (layer_out, diff_layer_out)
            # output_attentions=True : (layer_out, diff_layer_out, attn_probs, diff_attn_probs, [present_kv...])
            if output_attentions:
                attn_probs, diff_attn_probs = layer_outputs[2:4]
                all_self_attentions = all_self_attentions + (attn_probs,)
                all_diff_self_attentions = all_diff_self_attentions + (diff_attn_probs,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_diff_hidden_states = all_diff_hidden_states + (diff_hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,             # last_hidden_state
                    diff_hidden_states,        
                    next_decoder_cache,        # 仍然占位；encoder-only 下通常为 None / ()
                    all_hidden_states,         
                    all_diff_hidden_states,    
                    all_self_attentions,       
                    all_diff_self_attentions,  
                    # cross-attentions 在 encoder-only 下为 None，这里不扩展 diff_cross_attentions 进 tuple
                ]
                if v is not None
            )

        # return_dict=True：BaseModelOutputWithDelta
        return BaseModelOutputWithDelta(
            last_hidden_state=hidden_states,
            diff_last_hidden_state=diff_hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            diff_hidden_states=all_diff_hidden_states,
            attentions=all_self_attentions,
            diff_attentions=all_diff_self_attentions,
            # 我们的 forward_delta 中只支持 encoder-only 标准路径下的 RoBERTa 没有 cross-attention，这两个就保持 None
            cross_attentions=None,
            diff_cross_attentions=None,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        use_diff =  getattr(config, "apply_forward_delta", False)
        if use_diff:
            self.dense = diffLinear(layer_name = "roberta.pooler.dense", in_features = config.hidden_size, out_features = config.hidden_size, bias =True, uv_provider=config.uv_provider, z_provider=config.z_provider,)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
    def forward_delta(self, hidden_states: torch.Tensor, diff_hidden_states: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        first_token_tensor = hidden_states[:, 0]            # [B, H]
        diff_first_token_tensor = diff_hidden_states[:, 0]  # [B, H]

        if hasattr(self.dense, "forward_delta"):
            pooled, diff_pooled = self.dense.forward_delta(
                first_token_tensor,
                diff_first_token_tensor,
            )
        else:
            pooled = self.dense(first_token_tensor)
            diff_pooled = self.dense(diff_first_token_tensor)

        pooled_activated = self.activation(pooled)
        pooled_activated_perturbed = self.activation(pooled + diff_pooled)
        diff_pooled_activated = pooled_activated_perturbed - pooled_activated

        return pooled_activated, diff_pooled_activated


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RobertaEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class RobertaModel(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if self.config.is_decoder:
        #     use_cache = use_cache if use_cache is not None else self.config.use_cache
        # else:
        #     use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        
    def forward_delta(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPoolingAndDelta]:
        """
        和 forward 结构对应的 delta 版本：
        - 调用 embeddings.forward_delta -> 得到 (embedding_output, diff_embedding_output)
        - 调用 encoder.forward_delta   -> 得到 (sequence_output, diff_sequence_output, ...)
        - 调用 pooler.forward_delta    -> 得到 (pooled_output, diff_pooled_output)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if hasattr(self.embeddings, "forward_delta"):
            embedding_output, diff_embedding_output = self.embeddings.forward_delta(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            diff_embedding_output = torch.zeros_like(embedding_output)

        encoder_outputs = self.encoder.forward_delta(
            hidden_states=embedding_output,
            diff_hidden_states=diff_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            sequence_output = encoder_outputs.last_hidden_state
            diff_sequence_output = encoder_outputs.diff_last_hidden_state
        else:
            # tuple 模式下：encoder_outputs[0], [1] 分别是 base / diff
            sequence_output, diff_sequence_output = encoder_outputs[:2]

        if self.pooler is not None:
            if hasattr(self.pooler, "forward_delta"):
                pooled_output, diff_pooled_output = self.pooler.forward_delta(
                    sequence_output,
                    diff_sequence_output,
                )
            else:
                pooled_output = self.pooler(sequence_output)
                diff_pooled_output = torch.zeros_like(pooled_output) if pooled_output is not None else None
        else:
            pooled_output = None
            diff_pooled_output = None

        if not return_dict:
            return (sequence_output, diff_sequence_output, pooled_output, diff_pooled_output) + encoder_outputs[2:]

        # return_dict=True：返回我们自定义的 BaseModelOutputWithPoolingAndDelta
        return BaseModelOutputWithPoolingAndDelta(
            last_hidden_state=sequence_output,
            diff_last_hidden_state=diff_sequence_output,
            pooler_output=pooled_output,
            diff_pooler_output=diff_pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            diff_hidden_states=getattr(encoder_outputs, "diff_hidden_states", None),
            attentions=encoder_outputs.attentions,
            diff_attentions=getattr(encoder_outputs, "diff_attentions", None),
            cross_attentions=getattr(encoder_outputs, "cross_attentions", None),
            diff_cross_attentions=getattr(encoder_outputs, "diff_cross_attentions", None),
        )


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)
class RobertaForCausalLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        >>> config = AutoConfig.from_pretrained("roberta-base")
        >>> config.is_decoder = True
        >>> model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        
    def forward_delta(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithDelta]:
            
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        roberta_outputs = self.roberta.forward_delta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,   # 这里强制 dict，下面统一用字段
        )

        sequence_output = roberta_outputs.last_hidden_state           # [B, L, H]
        diff_sequence_output = roberta_outputs.diff_last_hidden_state # [B, L, H]

        
        logits_base, diff_logits = self.lm_head.forward_delta(sequence_output, diff_sequence_output)  # base logits
        logits_perturbed = logits_base + diff_logits
        # diff_prediction_scores = prediction_scores_perturbed - prediction_scores

        loss_base = None
        loss_perturbed = None
        if labels is not None:
            # next-token 预测：logits/labels 右移一位对齐
            shifted_labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()

            shifted_base = logits_base[:, :-1, :].contiguous()
            loss_base = loss_fct(
                shifted_base.view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
            )
            
            shifted_pert = logits_perturbed[:, :-1, :].contiguous()
            loss_perturbed = loss_fct(
                shifted_pert.view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
            )
        
        hidden_states = roberta_outputs.hidden_states
        diff_hidden_states = getattr(roberta_outputs, "diff_hidden_states", None)
        if hidden_states is not None and diff_hidden_states is not None:
            perturbed_hidden_states = tuple(
                h + dh for h, dh in zip(hidden_states, diff_hidden_states)
            )
        else:
            perturbed_hidden_states = None

        attentions = roberta_outputs.attentions
        diff_attentions = getattr(roberta_outputs, "diff_attentions", None)
        if attentions is not None and diff_attentions is not None:
            perturbed_attentions = tuple(
                a + da for a, da in zip(attentions, diff_attentions)
            )
        else:
            perturbed_attentions = None

        cross_attentions = roberta_outputs.cross_attentions
        diff_cross_attentions = getattr(roberta_outputs, "diff_cross_attentions", None)
        if cross_attentions is not None and diff_cross_attentions is not None:
            perturbed_cross_attentions = tuple(
                ca + dca for ca, dca in zip(cross_attentions, diff_cross_attentions)
            )
        else:
            perturbed_cross_attentions = None

        if not return_dict:
            output = (
                logits_base,
                logits_perturbed,
                roberta_outputs.past_key_values,
                hidden_states,
                perturbed_hidden_states,
                attentions,
                perturbed_attentions,
                cross_attentions,
                perturbed_cross_attentions,
            )
            if loss_base is not None:
                return (loss_base, loss_perturbed) + output
            else:
                return output
            
        # return_dict=True：返回我们自定义的 CausalLMOutputWithDelta
        return CausalLMOutputWithDelta(
            loss_base=loss_base,
            loss_perturbed=loss_perturbed,
            logits=logits_base,
            perturbed_logits=logits_perturbed,
            past_key_values=roberta_outputs.past_key_values,
            hidden_states=hidden_states,
            perturbed_hidden_states=perturbed_hidden_states,
            attentions=attentions,
            perturbed_attentions=perturbed_attentions,
            cross_attentions=cross_attentions,
            perturbed_cross_attentions=perturbed_cross_attentions,
        )
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def forward_delta(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MaskedLMOutputWithDelta]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        roberta_outputs = self.roberta.forward_delta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,   # 强制 dict，方便取字段
        )

        hidden_base = roberta_outputs.last_hidden_state           # [B, L, H] base
        # hidden_perturbed = sequence_output + roberta_outputs.diff_last_hidden_state

        logits_base, diff_logits = self.lm_head.forward_delta(hidden_base, diff_last_hidden_state)          # [B, L, V]
        logits_perturbed = logits_base + diff_logits

        loss_base = None
        loss_perturbed = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_base = loss_fct(
                logits_base.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            loss_perturbed = loss_fct(
                logits_perturbed.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        # 重建中间层的 perturbed hidden / attentions（base + diff）
        hidden_states = roberta_outputs.hidden_states
        diff_hidden_states = getattr(roberta_outputs, "diff_hidden_states", None)
        if hidden_states is not None and diff_hidden_states is not None:
            perturbed_hidden_states = tuple(
                h + dh for h, dh in zip(hidden_states, diff_hidden_states)
            )
        else:
            perturbed_hidden_states = None

        attentions = roberta_outputs.attentions
        diff_attentions = getattr(roberta_outputs, "diff_attentions", None)
        if attentions is not None and diff_attentions is not None:
            perturbed_attentions = tuple(
                a + da for a, da in zip(attentions, diff_attentions)
            )
        else:
            perturbed_attentions = None

        if not return_dict:
            output = (
                logits_base,
                logits_perturbed,
                hidden_states,
                perturbed_hidden_states,
                attentions,
                perturbed_attentions,
            )
            if loss_base is not None:
                return (loss_base, loss_perturbed) + output
            else:
                return output

        return MaskedLMOutputWithDelta(
            loss_base=loss_base,
            loss_perturbed=loss_perturbed,
            logits=logits_base,
            perturbed_logits=logits_perturbed,
            hidden_states=hidden_states,
            perturbed_hidden_states=perturbed_hidden_states,
            attentions=attentions,
            perturbed_attentions=perturbed_attentions,
        )


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        use_diff = getattr(config, "apply_forward_delta", False)
        if use_diff:
            self.dense = diffLinear(layer_name = "lm_head.dense", in_features = config.hidden_size, out_features = config.hidden_size, bias =True, uv_provider=config.uv_provider, z_provider=config.z_provider,)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if use_diff:
            self.layer_norm = diffLayerNorm(config.hidden_size, eps=config.layer_norm_eps, layer_name="lm_head.layer_norm", z_provider=config.z_provider)
        else:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if use_diff:
            self.decoder = diffLinear(layer_name = "lm_head.decoder", in_features = config.hidden_size, out_features = config.vocab_size, bias =True, uv_provider=config.uv_provider, z_provider=config.z_provider)
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        
        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # self.decoder.bias = self.bias
        self.bias = self.decoder.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x
    
    def forward_delta(self, features, diff_features=None, **kwargs):
        if diff_features is None:
            diff_features = torch.zeros_like(features)

        x, diff_x = self.dense.forward_delta(features, diff_features)
        x_pert = x + diff_x 
        x = gelu(x)
        x_pert = gelu(x_pert)
        l, diffl = self.layer_norm.forward_delta(x, x_pert-x)
        l, diffl = self.decoder.forward_delta(l, diffl)
    
        return l, diffl

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias
            
            
#  --------------------------------------------------------------------------
#  以上修改了delta的版本 对应着models.py中ForPromptFinetuning的版本


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForMultipleChoice(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        
        use_diff = getattr(config, "apply_forward_delta", False)
        # classifier.dense
        if use_diff:
            self.dense = diffLinear(layer_name = "classifier.dense", in_features = config.hidden_size, out_features = config.hidden_size, bias =True, uv_provider=config.uv_provider, z_provider=config.z_provider)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if use_diff:
            self.out_proj = diffLinear(layer_name = "classifier.out_proj", in_features = config.hidden_size, out_features = config.num_labels, bias =True, uv_provider=config.uv_provider, z_provider=config.z_provider)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    def forward_delta(self, features, diff_features=None, **kwargs):
        if diff_features is None:
            diff_features = torch.zeros_like(features)
            
        x_base = features[:, 0, :]         # [B, H]
        dx = diff_features[:, 0, :]        # [B, H]
        x_pert = x_base + dx
        
        x_base = self.dropout(x_base)
        x_pert = self.dropout(x_pert)
        dx = x_pert - x_base
        
        x_base, dx = self.dense.forward_delta(x_base, dx)
        x_base = torch.tanh(x_base)
        x_pert = torch.tanh(x_base + dx)
        x_base = self.dropout(x_base)
        x_pert = self.dropout(x_pert)
        dx = x_pert - x_base
        
        x_base, dx = self.out_proj.forward_delta(x_base, dx)
        return x_base, dx
        


@add_start_docstrings(
    """
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

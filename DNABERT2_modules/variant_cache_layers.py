import copy
import logging
import math
import warnings
from typing import List, Optional, Tuple, Union

from DNABERT2_modules.bert_padding import pad_input, unpad_input
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers.activations import ACT2FN
from dataclasses import dataclass
from transformers.modeling_outputs import (MaskedLMOutput,
                                           SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from DNABERT2_modules.bert_layers import BertGatedLinearUnitMLP, BertLayer, BertUnpadAttention, _get_alibi_head_slopes


class VariantCacheBertEncoder(nn.Module):
    """A stack of BERT layers providing the backbone of Mosaic BERT.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertEncoder`,
    but with substantial modifications to implement unpadding and ALiBi.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.num_attention_heads = config.num_attention_heads
        self.attn_impl = getattr(config, 'attn_impl', 'flash')

        # Per-head ALiBi slopes (positive), used by the flash path to compute the
        # bias in-kernel. Registered as a non-persistent buffer so it follows the
        # model's device/dtype moves but stays out of the state_dict.
        self.register_buffer(
            'alibi_slopes',
            torch.tensor(_get_alibi_head_slopes(self.num_attention_heads),
                         dtype=torch.float32),
            persistent=False)

        # The dense ALiBi+mask bias is only needed by the triton/torch paths.
        # The alibi mask will be dynamically expanded if it is too small for
        # the input the model receives. But it generally helps to initialize it
        # to a reasonably large size to help pre-allocate CUDA memory.
        # The default `alibi_starting_size` is 512.
        self._current_alibi_size = int(config.alibi_starting_size)
        self.alibi = torch.zeros(
            (1, self.num_attention_heads, self._current_alibi_size,
             self._current_alibi_size))
        self.rebuild_alibi_tensor(size=config.alibi_starting_size)

    def forward_reference(
        self,
        hidden_states: torch.Tensor, # (reference_seqlen, hidden) 
        attention_mask: torch.Tensor,
        output_all_encoded_layers: Optional[bool] = True,
        subset_mask: Optional[torch.Tensor] = None,
        output_layer: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:

        attention_mask_bool = attention_mask.bool()
        hidden_states = rearrange(hidden_states, 's d -> 1 s d')
        batch, seqlen = hidden_states.shape[:2]

        hidden_states, indices, cu_seqlens, _ = unpad_input(
            hidden_states, attention_mask_bool)

        if self.attn_impl == 'flash':
            # FlashAttention computes ALiBi in-kernel from per-head slopes, and
            # padding is fully encoded by cu_seqlens, so no dense bias/mask is
            # built. Pass slopes (kept on the right device) down to each layer.
            alibi_slopes = self.alibi_slopes.to(device=hidden_states.device)
            alibi_attn_mask = None
        else:
            # throw unimplemented error
            raise NotImplementedError(
                f"attn_impl {self.attn_impl} is not implemented. Only 'flash' is supported."
            )

        # Resolve negative layer index once so we can compare against loop index
        n_layers = len(self.layer)
        target_layer = None
        if output_layer is not None:
            target_layer = output_layer if output_layer >= 0 else n_layers + output_layer
            if not (0 <= target_layer < n_layers):
                raise ValueError(
                    f"output_layer {output_layer} is out of range for a model with "
                    f"{n_layers} layers (valid range: [{-n_layers}, {n_layers - 1}])"
                )

        intermediate_unpadded: Optional[torch.Tensor] = None

        all_encoder_layers = []
        if subset_mask is None:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states,
                                             cu_seqlens,
                                             seqlen,
                                             None,
                                             indices,
                                             attn_mask=attention_mask,
                                             bias=alibi_attn_mask,
                                             alibi_slopes=alibi_slopes)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if target_layer is not None and i == target_layer:
                    intermediate_unpadded = hidden_states
            # Pad inputs and mask. It will insert back zero-padded tokens.
            # Assume ntokens is total number of tokens (padded and non-padded)
            # and ntokens_unpad is total number of non-padded tokens.
            # Then padding performs the following de-compression:
            #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            if intermediate_unpadded is not None:
                intermediate_hidden_states = pad_input(
                    intermediate_unpadded, indices, batch, seqlen)
            else:
                intermediate_hidden_states = None
        else:
            for i in range(len(self.layer) - 1):
                layer_module = self.layer[i]
                hidden_states = layer_module(hidden_states,
                                             cu_seqlens,
                                             seqlen,
                                             None,
                                             indices,
                                             attn_mask=attention_mask,
                                             bias=alibi_attn_mask,
                                             alibi_slopes=alibi_slopes)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if target_layer is not None and i == target_layer:
                    intermediate_unpadded = hidden_states
            subset_idx = torch.nonzero(subset_mask[attention_mask_bool],
                                       as_tuple=False).flatten()
            hidden_states = self.layer[-1](hidden_states,
                                           cu_seqlens,
                                           seqlen,
                                           subset_idx=subset_idx,
                                           indices=indices,
                                           attn_mask=attention_mask,
                                           bias=alibi_attn_mask,
                                           alibi_slopes=alibi_slopes)
            if intermediate_unpadded is not None:
                intermediate_hidden_states = pad_input(
                    intermediate_unpadded, indices, batch, seqlen)
            else:
                intermediate_hidden_states = None

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, intermediate_hidden_states
    
    def forward(self, variant_cache: torch.Tensor, # (batch, cache_seqlen, hidden), embeddings of the variant cache tokens
                variant_cache_indices: torch.Tensor, # (cache_seqlen,), indices of the variant cache tokens in the original input sequence)
                hidden_states: torch.Tensor, # (reference_seqlen, hidden) 
                attention_mask: torch.Tensor,
                output_all_encoded_layers: Optional[bool] = True,
                subset_mask: Optional[torch.Tensor] = None,
                output_layer: Optional[int] = None,
            ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
            reference_layers, reference_hidden_states = self.forward_reference(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_all_encoded_layers=output_all_encoded_layers,
                subset_mask=subset_mask,
                output_layer=output_layer,
            )
            for i in range(len(reference_hidden_states)):
                reference_hidden_states[i][:, variant_cache_indices, :] = 0
            
            raise NotImplementedError("forward is not implemented. Use forward_reference instead.")

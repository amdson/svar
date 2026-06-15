# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local, self-contained re-implementation of the Llama decoder stack that the
# Carbon genome language model (HuggingFaceBio/Carbon-*) ships as. See the module
# docstring below for why this file exists and how it relates to the upstream
# transformers `modeling_llama.py`.
"""
CARBON_modules/carbon_layers.py
-------------------------------
A standalone copy of the Carbon backbone, analogous to
`DNABERT2_modules/bert_layers.py`.

Carbon is **not** a custom-architecture checkpoint: its config declares
``architectures=["LlamaForCausalLM"]`` and ``model_type="llama"``, so HuggingFace
serves it with the *built-in* `transformers` Llama modeling code (there is no
``modeling_*.py`` in the repo — ``trust_remote_code=True`` is only needed for the
custom ``tokenizer.py``). The 500M checkpoint is a decoder-only causal LM with:

  hidden_size 1024 · 28 layers · 16 query heads / 8 KV heads (GQA) · head_dim 64
  intermediate_size 3072 · SwiGLU MLP (SiLU) · RMSNorm · RoPE (theta 500000)
  tied input/output embeddings · vocab 155776

The upstream `transformers.models.llama.modeling_llama` is intentionally generic:
it dispatches attention through `ALL_ATTENTION_FUNCTIONS`, builds masks via
`create_causal_mask`, threads an HF `Cache`, and wraps everything in kernel-hub /
auto-docstring decorators. That machinery makes the forward pass hard to edit.

This file vendors the same math as a flat, dependency-light stack — pure-PyTorch
eager attention, an explicit additive causal mask, no `Cache`, no kernel hub — so
the forward pass is easy to fork. It mirrors `bert_layers.py`'s role for
DNABERT-2: the clean base on top of which a customized (variant-cache) forward
pass will be built (see the planned `variant_cache_layers.py` + `model_dev/`).

Parity
------
Module/parameter *attribute* names match the upstream Llama checkpoint exactly
(``model.embed_tokens``, ``model.layers.{i}.self_attn.{q,k,v,o}_proj``,
``model.layers.{i}.mlp.{gate,up,down}_proj``, ``input_layernorm``,
``post_attention_layernorm``, ``model.norm``, tied ``lm_head``), so the
safetensors load straight in with no key remapping. Class names are ``Carbon*``
to avoid colliding with the real Llama classes. Outputs match
``AutoModelForCausalLM`` to within fp32 eager numerics (see
`CARBON_modules/test_carbon_equivalence.py`).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)

try:
    # Carbon's config is a stock LlamaConfig; reuse it rather than re-vendoring
    # the ~100 fields and risking drift. Only a handful are read below.
    from transformers import LlamaConfig
except Exception:  # pragma: no cover - transformers always provides this
    LlamaConfig = None


def _rope_theta(config) -> float:
    """Read the RoPE base, tolerating both the flat (`rope_theta`) and the
    transformers>=5 nested (`rope_parameters["rope_theta"]`) config layouts."""
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        params = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None)
        if params is not None:
            theta = params.get("rope_theta")
    if theta is None:
        raise ValueError("Could not resolve rope_theta from the Carbon config.")
    return float(theta)


def _head_dim(config) -> int:
    return int(
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads)


class CarbonRMSNorm(nn.Module):
    """RMSNorm (T5-style): normalize by RMS in fp32, then scale. Equivalent to
    upstream ``LlamaRMSNorm``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class CarbonRotaryEmbedding(nn.Module):
    """Precomputes the per-position cos/sin tables for rotary embeddings.

    Implements only the *default* RoPE flavor (Carbon's config uses
    ``rope_type="default"`` with no scaling). ``inv_freq`` is a non-persistent
    buffer so it follows the model's device moves but stays out of the
    state_dict, matching upstream.
    """

    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        dim = _head_dim(config)
        base = _rope_theta(config)
        self.attention_scaling = 1.0  # default RoPE applies no extra scaling
        inv_freq = 1.0 / (base**(torch.arange(
            0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) /
                                 dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: (batch, seqlen)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force fp32 for the angle accumulation regardless of autocast.
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float()
                     @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: (x1, x2) -> (-x2, x1)."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Apply RoPE to query and key tensors of shape (batch, heads, seq, head_dim).

    ``cos``/``sin`` are (batch, seq, head_dim); ``unsqueeze_dim=1`` broadcasts
    them across the head dimension.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query KV heads to full multi-head width.

    (batch, num_kv_heads, seq, head_dim) -> (batch, num_kv_heads * n_rep, seq, head_dim).
    Equivalent to ``torch.repeat_interleave(x, n_rep, dim=1)``.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # (batch, heads, q_len, head_dim)
    key: torch.Tensor,  # (batch, kv_heads, kv_len, head_dim)
    value: torch.Tensor,  # (batch, kv_heads, kv_len, head_dim)
    attention_mask: Optional[torch.Tensor],  # additive (batch, 1, q_len, kv_len)
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch scaled-dot-product attention with GQA and an additive mask.

    Softmax is computed in fp32 (then cast back) to match upstream Llama eager
    attention. Returns ``(attn_output, attn_weights)`` where ``attn_output`` is
    (batch, q_len, heads, head_dim) — i.e. already transposed back so the caller
    can flatten the head dim.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Slice the mask to the current key length (no-op for full sequences).
        attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout,
                                         training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class CarbonAttention(nn.Module):
    """Grouped-query self-attention with rotary embeddings (Llama-style).

    The pure-PyTorch ``eager`` path is used unconditionally — it needs neither
    CUDA nor flash-attn and keeps the math inspectable, which is the whole point
    of vendoring this file. Bit-for-bit it matches HF's ``attn_implementation=
    "eager"`` and is within fp32 tolerance of the SDPA/flash paths.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = _head_dim(config)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (config.num_attention_heads //
                                     config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(config.hidden_size,
                                config.num_attention_heads * self.head_dim,
                                bias=attention_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=attention_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim,
                                config.hidden_size,
                                bias=attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]  # (batch, seq)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states, cos, sin)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CarbonMLP(nn.Module):
    """SwiGLU feed-forward: ``down(act(gate(x)) * up(x))`` (SiLU activation)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                   bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                 bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
                                   bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CarbonDecoderLayer(nn.Module):
    """Pre-norm transformer block: RMSNorm -> attn -> residual, then
    RMSNorm -> MLP -> residual. Equivalent to ``LlamaDecoderLayer``."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CarbonAttention(config=config, layer_idx=layer_idx)
        self.mlp = CarbonMLP(config)
        self.input_layernorm = CarbonRMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)
        self.post_attention_layernorm = CarbonRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_causal_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the additive (batch, 1, q_len, kv_len) attention bias.

    Combines the lower-triangular causal constraint with an optional padding
    mask (``attention_mask``: (batch, kv_len), 1 = keep, 0 = pad). Masked
    entries are set to ``finfo(dtype).min``, matching upstream
    ``create_causal_mask`` for the eager path so softmax zeroes them out.
    """
    batch, seq_len = input_shape
    min_val = torch.finfo(dtype).min

    # Lower-triangular causal mask: query i may attend to key j <= i.
    causal = torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)  # zero on/below diagonal
    causal = causal[None, None, :, :].expand(batch, 1, seq_len, seq_len).clone()

    if attention_mask is not None:
        # (batch, 1, 1, kv_len): broadcast padding across all query rows.
        pad = attention_mask[:, None, None, :].to(dtype=dtype, device=device)
        causal = causal.masked_fill(pad == 0, min_val)

    return causal


class CarbonModel(nn.Module):
    """Carbon backbone: token embeddings -> stack of decoder layers -> final
    RMSNorm. Mirrors ``LlamaModel`` (attribute names match for weight loading).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            CarbonDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CarbonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = CarbonRotaryEmbedding(config=config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        causal_mask = build_causal_mask(attention_mask, (batch, seq_len),
                                        inputs_embeds.dtype, device)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states: list[torch.Tensor] = []
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            # Match HF: hidden_states tuple has one entry per layer input plus
            # the final normed output (len == num_layers + 1).
            all_hidden_states.append(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=None,
        )


class CarbonForCausalLM(nn.Module):
    """Carbon causal LM: backbone + (tied) LM head.

    Drop-in for the embedder/scoring use of ``AutoModelForCausalLM``: returns a
    ``CausalLMOutputWithPast`` with ``.logits`` and, when
    ``output_hidden_states=True``, ``.hidden_states`` (per-layer + final).
    Generation/KV-cache are intentionally omitted — this base targets
    full-sequence embedding and LM scoring, and is the foundation the
    variant-cache forward will fork.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = CarbonModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)
        if getattr(config, "tie_word_embeddings", True):
            self.tie_weights()

    def tie_weights(self):
        """Tie the LM head to the input embeddings (Carbon ships tied)."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = False,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Standard causal-LM shift: predict token t+1 from position t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )

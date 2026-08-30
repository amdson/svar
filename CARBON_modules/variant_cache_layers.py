"""
CARBON_modules/variant_cache_layers.py
--------------------------------------
Variant-cache forward pass for Carbon, the causal analogue of
`DNABERT2_modules/variant_cache_layers.py`. Built on the local, hackable Llama
stack in `carbon_layers.py`.

The idea (shared with the DNABERT2 version)
-------------------------------------------
Given one *reference* sequence, we want the hidden states of a handful of
*variant* tokens substituted at chosen positions, **without** re-running the
whole model per variant. We freeze the reference residual stream (the input to
every layer is cached once by `forward_reference`) and, at each layer, recompute
only the variant tokens: a variant query at original position ``p`` attends to

    {reference keys at non-variant positions} ∪ {variant keys},

which is exactly a single softmax over the union of those two key sets. We never
concatenate them (that would force expanding the reference to the variant batch
size); instead we run two attentions and merge them with the log-sum-exp trick.
Keeping the reference at batch 1 also requires folding the variant scenarios into
the cross branch's query axis, or autograd re-expands it anyway — see
`_cross_attention_with_lse`.

What's different from DNABERT2 (bidirectional + ALiBi)
------------------------------------------------------
* **Causal.** Carbon is a decoder-only LM, so the key set is restricted to
  positions ``<= p``. The union {non-variant ≤ p} ∪ {variant ≤ p} is exactly
  {all real positions ≤ p} — a clean partition, no double counting. A variant at
  position 0 can have an *empty* cross branch; the `finfo.min` masking makes that
  branch contribute log-sum-exp weight 0, so the merge is still well defined.
* **RoPE, not ALiBi.** Position enters by rotating Q/K per their absolute
  position rather than via an additive bias. We rotate variant Q/K by the variant
  positions and reference K by ``0..ref-1`` (the rotary table already accepts
  arbitrary `position_ids`), then do plain scaled dot-product attention. The only
  additive terms left are the causal/padding masks.
* **GQA + pre-norm Llama block.** K/V live in `num_key_value_heads` and are
  `repeat_kv`-expanded; each block is RMSNorm→attn→residual, RMSNorm→SwiGLU→
  residual, with separate q/k/v/o projections.

Backends
--------
* ``efficient``  -> `VariantCacheCarbonEncoder.forward` (log-sum-exp merge).
* ``bruteforce`` -> `VariantCacheCarbonEncoder.forward_bruteforce`: runs ordinary
  full-sequence causal layers with the variant values spliced in, clamping the
  non-variant positions back to the frozen reference after each layer. Exactly
  the frozen-reference assumption, so it matches ``efficient`` up to numerics —
  the correctness oracle. Intentionally materializes the full activations the
  cache exists to avoid, so it's for validation / small batches only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .carbon_layers import (CarbonDecoderLayer, CarbonRMSNorm,
                            CarbonRotaryEmbedding, apply_rotary_pos_emb,
                            build_causal_mask, repeat_kv, rotate_half)


def _rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 unsqueeze_dim: int = 1) -> torch.Tensor:
    """Apply RoPE to a single tensor (e.g. reference keys) of shape
    (batch, heads, seq, head_dim). ``cos``/``sin`` are (batch, seq, head_dim)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class VariantCacheCarbonEncoder(nn.Module):
    """Carbon decoder stack with a variant-cache forward pass.

    Holds its own copy of the decoder layers, final RMSNorm, and rotary table
    (attribute names match the checkpoint so weights load straight in). The
    standard `CarbonDecoderLayer` is reused as-is for the reference and
    brute-force passes; the efficient pass reaches into each layer's submodules
    (`self_attn.{q,k,v,o}_proj`, `input_layernorm`, `post_attention_layernorm`,
    `mlp`) to do the manual cross/self attention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (config.num_attention_heads //
                                     config.num_key_value_heads)
        self.head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5

        self.layers = nn.ModuleList([
            CarbonDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CarbonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = CarbonRotaryEmbedding(config=config)

        # Recompute the reference stream's layer activations in the backward pass
        # instead of storing them. Off by default (inference never needs it); set
        # by the trainer. See `forward_reference`.
        self.reference_checkpointing = False

    # ------------------------------------------------------------------ #
    # Reference pass: cache the (pre-norm) residual stream entering each   #
    # layer. That cached input is the K/V source the variant update        #
    # cross-attends to — in a pre-norm block the keys/values are projected  #
    # from input_layernorm(layer_input), so the reference K/V for variant   #
    # layer i come from the reference input to layer i.                     #
    # ------------------------------------------------------------------ #
    def forward_reference(
        self,
        hidden_states: torch.Tensor,  # (ref_seqlen, hidden) reference embeddings
        attention_mask: torch.Tensor,  # (ref_seqlen,) or (1, ref_seqlen)
    ) -> List[torch.Tensor]:
        hidden = hidden_states.unsqueeze(0)  # (1, ref, hidden)
        ref_seqlen = hidden.shape[1]
        device, dtype = hidden.device, hidden.dtype

        position_ids = torch.arange(ref_seqlen, device=device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden, position_ids)
        causal_mask = build_causal_mask(
            attention_mask.reshape(1, ref_seqlen), (1, ref_seqlen), dtype, device)

        ckpt = self.reference_checkpointing and torch.is_grad_enabled()

        layer_inputs: List[torch.Tensor] = []
        for layer_module in self.layers:
            layer_inputs.append(hidden)  # pre-norm input entering this layer
            if ckpt:
                # `eager_attention_forward` saves, per layer, a (1, h, T, T) score
                # tensor plus a float32 softmax copy — at T=3334 over 28 layers that
                # is ~30 GB and it is the whole reason a grad-enabled reference pass
                # blows up. Recomputing the layer costs ~1.4x time and is exact.
                #
                # use_reentrant=False so this composes with the ordinary autograd
                # graph the variant branch builds on top of `layer_inputs`.
                hidden = torch.utils.checkpoint.checkpoint(
                    layer_module, hidden,
                    position_embeddings=(cos, sin),
                    attention_mask=causal_mask,
                    use_reentrant=False)
            else:
                hidden = layer_module(hidden,
                                      position_embeddings=(cos, sin),
                                      attention_mask=causal_mask)
        return layer_inputs

    @staticmethod
    def _attention_with_lse(
        q: torch.Tensor,  # (N, h, Lq, d)
        k: torch.Tensor,  # (N, h, Lk, d)
        v: torch.Tensor,  # (N, h, Lk, d)
        bias: torch.Tensor,  # additive, broadcastable to (N, h, Lq, Lk)
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention returning the normalized output and the
        per-query log-sum-exp of the (scaled, biased) scores.

        Used for the *self* branch, where q/k/v all share batch dim N. The cross
        branch has batch-1 reference K/V and goes through
        `_cross_attention_with_lse` instead — see that method for why.

        With ``bias`` built from ``finfo.min`` for disallowed keys, a fully
        masked row produces a finite (garbage) output with an astronomically
        negative lse, so the log-sum-exp merge gives it weight 0 — which is how
        an empty cross branch (variant at position 0) is handled without NaNs.
        """
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (N, h, Lq, Lk)
        scores = scores + bias
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # (N, h, Lq, 1)
        probs = torch.exp(scores - lse)
        out = torch.matmul(probs, v)  # (N, h, Lq, d)
        return out, lse

    @staticmethod
    def _cross_attention_with_lse(
        q: torch.Tensor,  # (N, h, Lq, d) variant queries
        k: torch.Tensor,  # (1, h, Lk, d) reference keys   — batch 1
        v: torch.Tensor,  # (1, h, Lk, d) reference values — batch 1
        bias: torch.Tensor,  # (1, 1, Lq, Lk) additive, shared across all N
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same contract as `_attention_with_lse`, but with the N variant
        scenarios folded into the *query* axis so the batch-1 reference K/V are
        never broadcast.

        Why this exists: `torch.matmul` on batch dims (N, h) x (1, h) expands the
        reference K/V to (N, h, Lk, d), and autograd **saves the expanded
        tensors**. That is 2 * N * h * Lk * d per layer — ~380 MB per scenario
        per layer at Lk=3334 — and it dominates training memory, which defeats
        the whole point of keeping the reference at batch 1. Reshaping the
        queries to (1, h, N*Lq, d) makes both matmul operands batch-1, so
        nothing expands and only the (h, N, Lq, Lk) probabilities are saved.

        The head dim must stay in the batch position (that is what lines up with
        the reference K/V), so N folds into the sequence axis, not the batch
        axis. Row ``n*Lq + q`` of the folded query is scenario n's query q, so
        the (1, 1, Lq, Lk) bias broadcasts correctly over an (h, N, Lq, Lk)
        *view* of the scores without ever being materialized.

        Inference is unaffected — under no_grad nothing is saved, so the
        broadcast was already free.
        """
        N, h, Lq, d = q.shape
        Lk = k.shape[-2]

        q_folded = q.transpose(0, 1).reshape(h, N * Lq, d).unsqueeze(0)
        scores = torch.matmul(q_folded, k.transpose(-1, -2)) * scale  # (1,h,N*Lq,Lk)
        scores = scores.view(h, N, Lq, Lk) + bias
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # (h, N, Lq, 1)
        probs = torch.exp(scores - lse)
        out = torch.matmul(probs.view(1, h, N * Lq, Lk), v)  # (1, h, N*Lq, d)
        out = out.view(h, N, Lq, d).transpose(0, 1)  # (N, h, Lq, d)
        return out, lse.transpose(0, 1)  # (N, h, Lq, 1)

    def forward(
        self,
        variant_cache: torch.Tensor,  # (N, cache_seqlen, hidden) variant embeddings
        variant_cache_indices: torch.Tensor,  # (cache_seqlen,) positions in the ref seq
        hidden_states: torch.Tensor,  # (ref_seqlen, hidden) reference embeddings
        attention_mask: torch.Tensor,  # (ref_seqlen,) or (1, ref_seqlen)
        output_all_encoded_layers: Optional[bool] = True,
    ) -> List[torch.Tensor]:
        # Per-layer frozen reference residual stream (the cross-attn K/V source).
        reference_layer_inputs = self.forward_reference(
            hidden_states=hidden_states, attention_mask=attention_mask)

        device, dtype = variant_cache.device, variant_cache.dtype
        N, cache_seqlen = variant_cache.shape[:2]
        ref_seqlen = reference_layer_inputs[0].shape[1]
        h, kvh = self.num_attention_heads, self.num_key_value_heads
        d = self.head_dim
        groups = self.num_key_value_groups
        scale = self.scaling
        neg = torch.finfo(dtype).min

        pos_q = variant_cache_indices.to(device=device).long()  # (cs,)
        pos_ref = torch.arange(ref_seqlen, device=device)        # (ref,)

        # RoPE tables for the variant positions and the reference positions.
        cos_q, sin_q = self.rotary_emb(variant_cache, pos_q.unsqueeze(0))
        cos_r, sin_r = self.rotary_emb(variant_cache, pos_ref.unsqueeze(0))

        # Cross-attention mask (1, 1, cs, ref): variant query at position p may
        # attend to reference key j iff j <= p (causal), j is real (not pad), and
        # j is not itself a variant position (those are covered by the self
        # branch — excluding them here avoids double counting).
        keep = attention_mask.to(device=device).bool().reshape(ref_seqlen).clone()
        keep[pos_q] = False
        causal_c = pos_ref[None, :] <= pos_q[:, None]           # (cs, ref)
        allow_c = causal_c & keep[None, :]
        cross_mask = torch.zeros(cache_seqlen, ref_seqlen, dtype=dtype,
                                 device=device).masked_fill(~allow_c, neg)
        cross_mask = cross_mask[None, None]                     # (1, 1, cs, ref)

        # Self-attention mask (1, 1, cs, cs): causal among variant tokens.
        causal_s = pos_q[None, :] <= pos_q[:, None]             # (cs, cs)
        self_mask = torch.zeros(cache_seqlen, cache_seqlen, dtype=dtype,
                                device=device).masked_fill(~causal_s, neg)
        self_mask = self_mask[None, None]                       # (1, 1, cs, cs)

        var = variant_cache  # (N, cs, hidden) running variant residual stream
        all_encoder_layers: List[torch.Tensor] = []
        for layer_module, ref_input in zip(self.layers, reference_layer_inputs):
            attn = layer_module.self_attn

            residual = var
            var_normed = layer_module.input_layernorm(var)  # (N, cs, hidden)
            ref_normed = layer_module.input_layernorm(
                ref_input.to(device=device, dtype=dtype))   # (1, ref, hidden)

            # Variant Q/K/V (batch N) from the running variant residual stream.
            q_v = attn.q_proj(var_normed).view(N, cache_seqlen, h, d).transpose(1, 2)
            k_v = attn.k_proj(var_normed).view(N, cache_seqlen, kvh, d).transpose(1, 2)
            v_v = attn.v_proj(var_normed).view(N, cache_seqlen, kvh, d).transpose(1, 2)

            # Reference K/V (batch 1) from the cached layer input, this layer's
            # projections; kept at batch 1 and broadcast over N in the matmul.
            k_r = attn.k_proj(ref_normed).view(1, ref_seqlen, kvh, d).transpose(1, 2)
            v_r = attn.v_proj(ref_normed).view(1, ref_seqlen, kvh, d).transpose(1, 2)

            # RoPE: variant Q/K at the variant positions, reference K at 0..ref-1.
            q_v, k_v = apply_rotary_pos_emb(q_v, k_v, cos_q, sin_q)
            k_r = _rope_single(k_r, cos_r, sin_r)

            # Expand grouped KV heads to full head count.
            k_v, v_v = repeat_kv(k_v, groups), repeat_kv(v_v, groups)  # (N, h, cs, d)
            k_r, v_r = repeat_kv(k_r, groups), repeat_kv(v_r, groups)  # (1, h, ref, d)

            out_c, lse_c = self._cross_attention_with_lse(q_v, k_r, v_r,
                                                          cross_mask, scale)
            out_s, lse_s = self._attention_with_lse(q_v, k_v, v_v, self_mask, scale)

            # Merge the two softmaxes by their log-sum-exps == single softmax
            # over the union of the two key sets.
            lse = torch.logaddexp(lse_c, lse_s)
            attn_out = (torch.exp(lse_c - lse) * out_c +
                        torch.exp(lse_s - lse) * out_s)  # (N, h, cs, d)

            attn_out = attn_out.transpose(1, 2).reshape(N, cache_seqlen, h * d)
            attn_out = attn.o_proj(attn_out)
            var = residual + attn_out

            # Pre-norm MLP block.
            residual = var
            var = layer_module.post_attention_layernorm(var)
            var = layer_module.mlp(var)
            var = residual + var

            if output_all_encoded_layers:
                all_encoder_layers.append(var)

        if not output_all_encoded_layers:
            all_encoder_layers.append(var)
        return all_encoder_layers

    def forward_bruteforce(
        self,
        variant_cache: torch.Tensor,       # (N, cache_seqlen, hidden)
        variant_cache_indices: torch.Tensor,  # (cache_seqlen,)
        hidden_states: torch.Tensor,       # (ref_seqlen, hidden)
        attention_mask: torch.Tensor,      # (ref_seqlen,) or (1, ref_seqlen)
        output_all_encoded_layers: Optional[bool] = True,
    ) -> List[torch.Tensor]:
        """Oracle: ordinary full-sequence causal layers with the variant values
        spliced in, clamping non-variant positions back to the frozen reference
        after each layer. Matches `forward` up to numerics."""
        reference_layer_inputs = self.forward_reference(
            hidden_states=hidden_states, attention_mask=attention_mask)

        device, dtype = variant_cache.device, variant_cache.dtype
        N = variant_cache.shape[0]
        ref_seqlen = reference_layer_inputs[0].shape[1]
        idx = variant_cache_indices.to(device=device).long()

        # Variants don't change which positions are real, so they share the
        # reference's padding mask (broadcast over the N scenarios).
        attn_mask_full = attention_mask.to(device=device).reshape(
            1, ref_seqlen).expand(N, ref_seqlen)
        position_ids = torch.arange(ref_seqlen, device=device).unsqueeze(0)
        emb0 = reference_layer_inputs[0].to(device=device, dtype=dtype)
        cos, sin = self.rotary_emb(emb0, position_ids)
        causal_mask = build_causal_mask(attn_mask_full, (N, ref_seqlen), dtype,
                                        device)

        var = variant_cache  # (N, cs, hidden) running variant values
        all_encoder_layers: List[torch.Tensor] = []
        for layer_module, ref_input in zip(self.layers, reference_layer_inputs):
            full = ref_input.to(device=device, dtype=dtype).expand(
                N, ref_seqlen, var.shape[-1]).clone()
            full[:, idx, :] = var
            full_out = layer_module(full,
                                    position_embeddings=(cos, sin),
                                    attention_mask=causal_mask)
            var = full_out[:, idx, :]  # keep only the variant positions
            if output_all_encoded_layers:
                all_encoder_layers.append(var)

        if not output_all_encoded_layers:
            all_encoder_layers.append(var)
        return all_encoder_layers


@dataclass
class VariantCacheOutput:
    """Variant-position outputs from the cache forward.

    last_hidden_state : (N, cache_seqlen, hidden) post-final-norm hidden states
        at the variant positions (comparable to a normal forward's
        ``hidden_states[-1]`` sliced to those positions).
    logits : (N, cache_seqlen, vocab) or None.
    """
    last_hidden_state: torch.Tensor
    logits: Optional[torch.Tensor] = None


class VariantCacheCarbonForCausalLM(nn.Module):
    """Carbon causal LM that recomputes a set of variant positions through the
    variant cache, against a single frozen reference window.

    Drop-in-ish for measuring how close the frozen-reference approximation is to
    a normal forward: pass the reference ``input_ids`` plus the ``variant_positions``
    (and optionally substituted ``variant_input_ids``); get back the variant
    positions' post-norm hidden states and logits.

    backend
        ``'efficient'``  -> VariantCacheCarbonEncoder.forward (log-sum-exp merge)
        ``'bruteforce'`` -> VariantCacheCarbonEncoder.forward_bruteforce (oracle)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         getattr(config, "pad_token_id", None))
        self.encoder = VariantCacheCarbonEncoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)
        if getattr(config, "tie_word_embeddings", True):
            self.tie_weights()
        self.set_backend(getattr(config, "variant_cache_backend", "efficient"))

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def set_backend(self, backend: str) -> None:
        if backend not in ("efficient", "bruteforce"):
            raise ValueError(
                f"backend must be 'efficient' or 'bruteforce', got {backend!r}.")
        self.backend = backend

    def forward(
        self,
        input_ids: torch.LongTensor,         # (ref_seqlen,) or (1, ref_seqlen)
        variant_positions: torch.LongTensor,  # (cache_seqlen,)
        attention_mask: Optional[torch.Tensor] = None,  # (ref_seqlen,) optional
        variant_input_ids: Optional[torch.LongTensor] = None,  # (N, cs) or (cs,)
        output_logits: bool = True,
    ) -> VariantCacheOutput:
        input_ids = input_ids.reshape(-1)            # (ref_seqlen,)
        ref_seqlen = input_ids.shape[0]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(ref_seqlen, dtype=torch.long,
                                        device=device)

        variant_positions = variant_positions.to(device=device).long()
        # Default: substitute the reference's own tokens (pure approximation-error
        # probe — the cache should then reproduce the normal forward exactly up to
        # the frozen-reference assumption).
        if variant_input_ids is None:
            variant_input_ids = input_ids.index_select(0, variant_positions)
        variant_input_ids = variant_input_ids.to(device=device).long()
        if variant_input_ids.dim() == 1:
            variant_input_ids = variant_input_ids.unsqueeze(0)  # (1, cs)

        ref_emb = self.embed_tokens(input_ids)               # (ref, hidden)
        variant_cache = self.embed_tokens(variant_input_ids)  # (N, cs, hidden)

        run = (self.encoder.forward_bruteforce
               if self.backend == "bruteforce" else self.encoder.forward)
        var_hidden = run(variant_cache, variant_positions, ref_emb,
                         attention_mask, output_all_encoded_layers=False)[-1]
        var_hidden = self.encoder.norm(var_hidden)  # (N, cs, hidden) post-norm

        logits = self.lm_head(var_hidden) if output_logits else None
        return VariantCacheOutput(last_hidden_state=var_hidden, logits=logits)

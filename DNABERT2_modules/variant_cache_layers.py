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
from DNABERT2_modules.bert_layers import (BertEmbeddings, BertGatedLinearUnitMLP,
                                          BertLayer, BertOnlyMLMHead,
                                          BertUnpadAttention,
                                          _get_alibi_head_slopes)


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
        self.attention_head_size = config.hidden_size // self.num_attention_heads

        # Unlike BertEncoder, this module never materializes a dense ALiBi tensor:
        # the reference pass (`forward_reference`) uses FlashAttention's in-kernel
        # ALiBi from `alibi_slopes`, and the variant update builds a small custom
        # bias on the fly (with arbitrary per-token positions), so there is no
        # dense `self.alibi` buffer to maintain here.

    def _dense_alibi_bias(self, seqlen: int, attention_mask: torch.Tensor,
                          device, dtype) -> torch.Tensor:
        """Dense (batch, heads, seqlen, seqlen) ALiBi+padding bias for the
        triton/torch attention paths over a *contiguous* sequence (positions
        0..seqlen-1). Mirrors BertEncoder's construction; used by the reference
        pass and the brute-force backend, both of which run full sequences where
        the token's array index equals its ALiBi position.
        """
        ext = attention_mask.to(device=device, dtype=torch.float32)
        ext = ext.unsqueeze(1).unsqueeze(2)            # (batch, 1, 1, seqlen)
        ext = (1.0 - ext) * -10000.0
        slopes = self.alibi_slopes.to(device=device)   # (heads,)
        pos = torch.arange(seqlen, device=device)
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()          # (seqlen, seqlen)
        alibi = (-slopes.view(-1, 1, 1) * rel.unsqueeze(0)).unsqueeze(0)  # (1, h, s, s)
        return (ext + alibi).to(dtype)

    # Run the reference sequence through the BERT stack and cache, for every
    # layer, the (padded) residual stream that *enters* that layer. Those cached
    # inputs are exactly the K/V source the variant update cross-attends to: in a
    # standard layer the keys/values are projected from the layer's input, so the
    # reference K/V for variant-layer `i` is the reference input to layer `i`
    # (embeddings for i==0, output of layer i-1 otherwise).
    def forward_reference(
        self,
        hidden_states: torch.Tensor, # (reference_seqlen, hidden)
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:

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
            # triton / torch paths need a dense (batch, heads, seq, seq) bias.
            alibi_slopes = None
            alibi_attn_mask = self._dense_alibi_bias(
                seqlen, attention_mask, hidden_states.device, hidden_states.dtype)

        # layer_inputs[i] = padded reference residual stream entering layer i.
        layer_inputs: List[torch.Tensor] = []
        for layer_module in self.layer:
            layer_inputs.append(
                pad_input(hidden_states, indices, batch, seqlen))
            hidden_states = layer_module(hidden_states,
                                         cu_seqlens,
                                         seqlen,
                                         None,
                                         indices,
                                         attn_mask=attention_mask,
                                         bias=alibi_attn_mask,
                                         alibi_slopes=alibi_slopes)

        return layer_inputs
    
    @staticmethod
    def _attention_with_lse(
        q: torch.Tensor,  # (N, h, Lq, d)
        k: torch.Tensor,  # (B, h, Lk, d), B in {1, N}
        v: torch.Tensor,  # (B, h, Lk, d), B in {1, N}
        bias: torch.Tensor,  # broadcastable to (N, h, Lq, Lk)
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plain-math scaled dot-product attention returning the normalized
        output and the per-query log-sum-exp of the (biased, scaled) scores.

        `k`/`v` may keep batch dim 1: matmul broadcasts them over the N queries,
        so the reference K/V are never expanded to batch N (avoids the OOM).
        Returns (out: (N, h, Lq, d), lse: (N, h, Lq, 1)).
        """
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (N, h, Lq, Lk)
        scores = scores + bias
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # (N, h, Lq, 1)
        probs = torch.exp(scores - lse)
        out = torch.matmul(probs, v)  # (N, h, Lq, d)
        return out, lse

    def forward(self, variant_cache: torch.Tensor, # (batch, cache_seqlen, hidden), embeddings of the variant cache tokens
                variant_cache_indices: torch.Tensor, # (cache_seqlen,), indices of the variant cache tokens in the original input sequence)
                hidden_states: torch.Tensor, # (reference_seqlen, hidden)
                attention_mask: torch.Tensor,
                output_all_encoded_layers: Optional[bool] = True,
                subset_mask: Optional[torch.Tensor] = None,
                output_layer: Optional[int] = None,
            ) -> List[torch.Tensor]:
            if subset_mask is not None or output_layer is not None:
                raise NotImplementedError(
                    "subset_mask / output_layer are not supported by the variant "
                    "cache forward yet.")

            # Cache, per layer, the reference residual stream that enters it; this
            # is the (fixed) K/V source each variant layer cross-attends to.
            reference_layer_inputs = self.forward_reference(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )  # [(1, reference_seqlen, hidden)] * num_layers

            # Assume the reference hidden states are fixed. Each layer computes an
            # update to the residual stream of the *variant* tokens only. A variant
            # query at original position p attends jointly to
            #   {reference keys at non-variant positions} ∪ {variant keys},
            # with ALiBi bias -slope*|p - pos(key)|. We never concatenate those two
            # key sets (that would force expanding the reference to batch N); instead
            # we run two attentions and merge them with the log-sum-exp trick, which
            # is exactly equivalent to a single softmax over the union.
            device = variant_cache.device
            dtype = variant_cache.dtype
            cache_seqlen = variant_cache.shape[1]
            ref_seqlen = reference_layer_inputs[0].shape[1]
            h = self.num_attention_heads
            d = self.attention_head_size
            scale = 1.0 / math.sqrt(d)

            slopes = self.alibi_slopes.to(device=device)  # (h,) fp32, positive
            pos_q = variant_cache_indices.to(device=device).long()  # (cache_seqlen,)
            pos_k_ref = torch.arange(ref_seqlen, device=device)     # (ref_seqlen,)

            # Cross-attention bias (1, h, cache_seqlen, ref_seqlen): ALiBi from the
            # variant positions to the reference positions, with reference padding
            # and the variant's own positions masked out (those positions are
            # represented by the variant keys in the self branch, so excluding them
            # here avoids double counting).
            rel_c = (pos_q[:, None] - pos_k_ref[None, :]).abs().to(slopes.dtype)
            cross_bias = -slopes.view(h, 1, 1) * rel_c.view(1, cache_seqlen, ref_seqlen)
            keep = attention_mask.to(device=device).bool().reshape(ref_seqlen).clone()
            keep[pos_q] = False
            neg_inf = torch.finfo(cross_bias.dtype).min
            cross_bias = cross_bias.masked_fill(
                ~keep.view(1, 1, ref_seqlen), neg_inf).unsqueeze(0)

            # Self-attention bias (1, h, cache_seqlen, cache_seqlen): ALiBi between
            # variant tokens, using their custom original-sequence positions.
            rel_s = (pos_q[:, None] - pos_q[None, :]).abs().to(slopes.dtype)
            self_bias = (-slopes.view(h, 1, 1) *
                         rel_s.view(1, cache_seqlen, cache_seqlen)).unsqueeze(0)

            var = variant_cache  # (N, cache_seqlen, hidden)
            all_encoder_layers: List[torch.Tensor] = []
            for layer_module, ref_input in zip(self.layer, reference_layer_inputs):
                self_attn = layer_module.attention.self

                # Variant Q/K/V (batch N) from the running variant residual stream.
                qkv_v = self_attn.Wqkv(var)
                qkv_v = rearrange(qkv_v, 'n s (three h d) -> three n h s d',
                                  three=3, h=h)
                q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]  # (N, h, cache_seqlen, d)

                # Reference K/V (batch 1) from the cached layer input; projected with
                # this layer's Wqkv. Kept at batch 1 and broadcast in the matmul.
                qkv_r = self_attn.Wqkv(ref_input.to(device=device, dtype=dtype))
                qkv_r = rearrange(qkv_r, 'b s (three h d) -> three b h s d',
                                  three=3, h=h)
                k_r, v_r = qkv_r[1], qkv_r[2]  # (1, h, ref_seqlen, d)

                out_c, lse_c = self._attention_with_lse(q_v, k_r, v_r, cross_bias, scale)
                out_s, lse_s = self._attention_with_lse(q_v, k_v, v_v, self_bias, scale)

                # Merge the two softmaxes by their log-sum-exps == single softmax
                # over the concatenated keys.
                lse = torch.logaddexp(lse_c, lse_s)
                attn = (torch.exp(lse_c - lse) * out_c +
                        torch.exp(lse_s - lse) * out_s)  # (N, h, cache_seqlen, d)

                # Remaining standard-BERT-layer plumbing: dense + residual + LN, MLP.
                # These run on the flat (nnz, hidden) layout the BertLayer expects
                # (the GLU MLP slices dim 1 = the feature dim), so flatten the
                # (N, cache_seqlen) leading dims and restore them after.
                n_var = var.shape[0]
                attn = rearrange(attn, 'n h s d -> (n s) (h d)')
                var_flat = rearrange(var, 'n s d -> (n s) d')
                attn_out = layer_module.attention.output(attn, var_flat)
                var = rearrange(layer_module.mlp(attn_out), '(n s) d -> n s d', n=n_var)
                if output_all_encoded_layers:
                    all_encoder_layers.append(var)

            if not output_all_encoded_layers:
                all_encoder_layers.append(var)
            return all_encoder_layers

    def _run_full_layer(self, layer_module, hidden_padded: torch.Tensor,
                        attention_mask: torch.Tensor,
                        alibi_attn_mask: Optional[torch.Tensor],
                        alibi_slopes: Optional[torch.Tensor]) -> torch.Tensor:
        """Run one standard BertLayer over a padded (batch, seqlen, hidden)
        tensor, returning the padded output. Used by the brute-force backend."""
        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_padded.shape[:2]
        unpadded, indices, cu_seqlens, _ = unpad_input(
            hidden_padded, attention_mask_bool)
        out = layer_module(unpadded,
                           cu_seqlens,
                           seqlen,
                           None,
                           indices,
                           attn_mask=attention_mask,
                           bias=alibi_attn_mask,
                           alibi_slopes=alibi_slopes)
        return pad_input(out, indices, batch, seqlen)

    def forward_bruteforce(
        self,
        variant_cache: torch.Tensor,       # (batch, cache_seqlen, hidden)
        variant_cache_indices: torch.Tensor,  # (cache_seqlen,)
        hidden_states: torch.Tensor,       # (reference_seqlen, hidden)
        attention_mask: torch.Tensor,
        output_all_encoded_layers: Optional[bool] = True,
        subset_mask: Optional[torch.Tensor] = None,
        output_layer: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Brute-force equivalent of `forward`.

        Instead of the log-sum-exp merge, this runs *ordinary* full-sequence
        BERT layers on the full variant sequence (the reference with the variant
        substitutions), and after each layer artificially clamps the non-variant
        positions back to the cached reference residual stream. That clamping is
        exactly the frozen-reference assumption, so this produces the same result
        as `forward` (up to numerics) — it's the correctness oracle and a simple
        alternative backend. It is intentionally inefficient: it materializes the
        full (batch, reference_seqlen, hidden) activations the cache exists to
        avoid, so use it for validation / small batches, not production.
        """
        if subset_mask is not None or output_layer is not None:
            raise NotImplementedError(
                "subset_mask / output_layer are not supported by the variant "
                "cache forward yet.")

        # Per-layer frozen reference residual stream (input to each layer).
        reference_layer_inputs = self.forward_reference(
            hidden_states=hidden_states, attention_mask=attention_mask)

        device = variant_cache.device
        dtype = variant_cache.dtype
        N = variant_cache.shape[0]
        ref_seqlen = reference_layer_inputs[0].shape[1]
        idx = variant_cache_indices.to(device=device).long()

        # Reference mask (1, ref) broadcast to the variant batch. Variants don't
        # change which positions are real, so they share the reference's mask.
        attn_mask_full = attention_mask.to(device=device).reshape(1, ref_seqlen).expand(N, ref_seqlen)

        if self.attn_impl == 'flash':
            alibi_slopes = self.alibi_slopes.to(device=device)
            alibi_attn_mask = None
        else:
            alibi_slopes = None
            alibi_attn_mask = self._dense_alibi_bias(
                ref_seqlen, attn_mask_full, device, dtype)

        var = variant_cache  # (N, cache_seqlen, hidden) — running variant values
        all_encoder_layers: List[torch.Tensor] = []
        for layer_module, ref_input in zip(self.layer, reference_layer_inputs):
            # Full sequence = frozen reference everywhere, variant values spliced
            # in at the variant positions. .expand is a view; .clone makes it
            # writable for the in-place splice.
            full = ref_input.to(device=device, dtype=dtype).expand(
                N, ref_seqlen, var.shape[-1]).clone()
            full[:, idx, :] = var
            full_out = self._run_full_layer(
                layer_module, full, attn_mask_full, alibi_attn_mask, alibi_slopes)
            var = full_out[:, idx, :]  # keep only the variant positions
            if output_all_encoded_layers:
                all_encoder_layers.append(var)

        if not output_all_encoded_layers:
            all_encoder_layers.append(var)
        return all_encoder_layers


class VariantCacheBertForMaskedLM(nn.Module):
    """DNABERT-2 masked-LM that runs through the variant-cache encoder.

    Drop-in replacement for BertForMaskedLM in the variant-MLM scripts: same
    ``(input_ids, attention_mask, labels)`` call and ``MaskedLMOutput`` return,
    so swapping it in is just a ``--backend`` flag.

    Each row of the batch is an independent reference window. The masked tokens
    (``labels > 0``) are the "variant" positions whose hidden states the cache
    recomputes; every other token is frozen to the reference pass. The exact
    model and this model therefore receive identical inputs and differ *only* by
    the frozen-reference approximation, which is exactly the benign-ness
    comparison we want.

    Because each window is a distinct reference, windows are processed one at a
    time (the cache's amortization — one reference, many variants — doesn't apply
    to per-window MLM), so this is slower than the exact model here; that's fine
    for a diagnostic.

    backend
        ``'efficient'``  -> VariantCacheBertEncoder.forward (log-sum-exp merge)
        ``'bruteforce'`` -> VariantCacheBertEncoder.forward_bruteforce (oracle)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = VariantCacheBertEncoder(config)
        self.cls = BertOnlyMLMHead(config, self.embeddings.word_embeddings.weight)
        self.set_backend(getattr(config, 'variant_cache_backend', 'efficient'))

    def set_backend(self, backend: str) -> None:
        if backend not in ('efficient', 'bruteforce'):
            raise ValueError(
                f"backend must be 'efficient' or 'bruteforce', got {backend!r}.")
        self.backend = backend

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> MaskedLMOutput:
        if labels is None:
            raise ValueError(
                "VariantCacheBertForMaskedLM needs `labels` to locate the variant "
                "(masked) positions; labels > 0 marks them.")

        batch, seqlen = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Reference embeddings for the whole batch (one full sequence per row).
        emb = self.embeddings(input_ids, token_type_ids)  # (batch, seqlen, hidden)
        vocab_size = self.cls.predictions.decoder.weight.shape[0]
        logits = emb.new_zeros((batch, seqlen, vocab_size))

        run = (self.encoder.forward_bruteforce if self.backend == 'bruteforce'
               else self.encoder.forward)

        for b in range(batch):
            var_pos = torch.nonzero(labels[b] > 0, as_tuple=False).flatten()
            if var_pos.numel() == 0:
                continue
            ref_emb = emb[b]  # (seqlen, hidden)
            am = attention_mask[b].reshape(1, seqlen)
            # Variant embeddings are the reference embeddings at the masked
            # positions (same input as the exact model) — the cache just
            # recomputes them against the frozen context.
            variant_cache = ref_emb.index_select(0, var_pos).unsqueeze(0)  # (1, cs, hidden)
            var_hidden = run(variant_cache, var_pos, ref_emb, am,
                             output_all_encoded_layers=False)[-1]  # (1, cs, hidden)
            var_logits = self.cls(var_hidden)[0]  # (cs, vocab)
            logits[b].index_copy_(0, var_pos, var_logits.to(logits.dtype))

        loss = None
        tmask = labels > 0
        if tmask.any():
            loss = nn.CrossEntropyLoss()(logits[tmask], labels[tmask])

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=None,
                              attentions=None)
"""
CARBON_modules/variant_lora_layers.py
-------------------------------------
Variant-only LoRA on top of the Carbon variant cache.

A separate model class rather than a flag on `VariantCacheCarbonForCausalLM`,
because the whole point is a *structural* guarantee: the reference stream must
be provably untouched by the adapters. That guarantee is what buys the reference
cache below, and a runtime flag would leave it as something you have to re-check
every time the forward pass is edited.

The idea
--------
`VariantCacheCarbonEncoder.forward` already computes two streams with the same
weights:

  * the **reference** stream, in `forward_reference` — one batch-1 pass over the
    reference window, producing the per-layer residual stream that the variant
    queries cross-attend to;
  * the **variant** stream — the N haplotypes' tokens at the variant positions,
    recomputed layer by layer.

Only the second depends on the individual. So we attach LoRA deltas to the
variant call sites *only*. `forward_reference` is inherited from the parent
untouched and the adapters live in a sibling `ModuleList`, so no code path
reachable from `forward_reference` can see them.

What that buys
--------------
1. **The reference stream becomes a constant.** With the base frozen, a window's
   `reference_layer_inputs` never change over a run: compute once, reuse for
   every step and every haplotype chunk (see `ReferenceCache`). Today the
   reference pass is recomputed once per `hap_chunk` per window per step, with
   gradient checkpointing at ~1.4x on top because a grad-enabled reference pass
   otherwise blows up memory. All of that disappears — no reference backward at
   all.

2. **bf16 base with fp32 optimizer state.** Full fine-tuning in bf16 caps this
   benchmark at the baseline: torch AdamW keeps moments in the parameter dtype,
   so with no fp32 master weights the ~1e-2 relative updates fall under bf16's
   3.9e-3 relative epsilon and are rounded away. That failure mode needs the
   parameters to be *updated*. Here the base is frozen, so it can sit in bf16
   for the memory and bandwidth, while `LoRADelta` keeps its parameters in fp32
   and casts them at use — the fp32 master-weight scheme, for free.

What it does NOT buy
--------------------
It cannot make the model "attend to the variants more" in the reference stream.
In this approximation reference positions never attend to variant positions at
all: `forward_reference` runs on the pure reference sequence, so every
downstream reference token is computed as though each variant carried the
reference allele. That is the cache's defining approximation, measured at ~0.043
bits/SNP against the exact forward, and no adapter on the variant path can
recover it.

Also note LoRA *reduces* capacity. In the regime measured so far the model
underfits (train/val gap ~0.01, never overfitting), so at a fixed window count
expect this to score at best level with full fine-tuning. The reason to want it
is throughput per window, which is what lets the window count go up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .variant_cache_layers import (VariantCacheCarbonEncoder,
                                   VariantCacheCarbonForCausalLM,
                                   VariantCacheOutput)


# Every adaptable projection, and which of the two streams it is applied to in
# `VariantCacheCarbonEncoder.forward`. k_proj/v_proj appear in BOTH streams (the
# variant K/V come from `var_normed`, the reference K/V from `ref_normed`), which
# is why adapting them is a real modelling decision rather than a free choice —
# see `LoRAConfig.targets`.
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
ALL_TARGETS = ATTN_TARGETS + MLP_TARGETS

# q/o + the MLP: the projections that are *only* ever applied to the variant
# stream, so adapting them raises no question about the two key sets being
# projected differently. See LoRAConfig.targets.
DEFAULT_TARGETS = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = DEFAULT_TARGETS
    # Adapting k_proj/v_proj on the variant side only means variant keys and
    # reference keys reach the log-sum-exp merge through different effective
    # projections. The merge stays a valid softmax over the union of the two key
    # sets, but the sets are no longer commensurate. That may be exactly what
    # makes variant positions distinguishable, or it may break the merge's
    # calibration; it is off by default so the first result is not confounded.
    def __post_init__(self):
        bad = [t for t in self.targets if t not in ALL_TARGETS]
        if bad:
            raise ValueError(f"unknown LoRA target(s) {bad}; pick from {ALL_TARGETS}")
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class LoRADelta(nn.Module):
    """``B @ A @ x * (alpha/r)``, with the parameters held in fp32.

    The parameters stay fp32 no matter what dtype the base model is cast to, and
    are cast to the activation dtype inside `forward`. That is deliberate: the
    cast is differentiable, so gradients accumulate into the fp32 master copy and
    AdamW's moments are fp32 even when the base and the activations are bf16.
    This is the piece full fine-tuning cannot have without a custom optimizer,
    and it is why bf16 is safe here when it was not there.

    ``lora_B`` is zero-initialised, so a freshly built model is *exactly* the
    base model — the zero-shot number must match `VariantCacheCarbonForCausalLM`
    bit for bit, which `test_variant_lora.py` asserts.
    """

    def __init__(self, in_features: int, out_features: int, cfg: LoRAConfig):
        super().__init__()
        self.scaling = cfg.scaling
        self.lora_A = nn.Parameter(torch.empty(cfg.r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, cfg.r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        return F.linear(F.linear(self.dropout(x), a), b) * self.scaling


class VariantLayerAdapters(nn.Module):
    """The adapters for one decoder layer, keyed by base-projection name."""

    def __init__(self, config, cfg: LoRAConfig):
        super().__init__()
        hidden = config.hidden_size
        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        q_out = config.num_attention_heads * head_dim
        kv_out = config.num_key_value_heads * head_dim
        inter = config.intermediate_size
        shapes = {
            "q_proj": (hidden, q_out), "k_proj": (hidden, kv_out),
            "v_proj": (hidden, kv_out), "o_proj": (q_out, hidden),
            "gate_proj": (hidden, inter), "up_proj": (hidden, inter),
            "down_proj": (inter, hidden),
        }
        self.deltas = nn.ModuleDict({
            name: LoRADelta(*shapes[name], cfg) for name in cfg.targets})

    def delta(self, name: str, x: torch.Tensor) -> torch.Tensor | None:
        """The delta for one projection, or None when it is not adapted."""
        if name not in self.deltas:
            return None
        return self.deltas[name](x)


class VariantLoRACarbonEncoder(VariantCacheCarbonEncoder):
    """Variant-cache encoder whose adapters apply to the variant stream only.

    `forward_reference` is inherited verbatim and the adapters are held in
    `self.adapters`, a sibling of `self.layers`. Nothing reachable from
    `forward_reference` — which calls the stock `CarbonDecoderLayer` — can reach
    them, so the reference stream is base-weights-only by construction rather
    than by convention.
    """

    def __init__(self, config, lora: LoRAConfig | None = None):
        super().__init__(config)
        self.lora_config = lora or LoRAConfig()
        self.adapters = nn.ModuleList([
            VariantLayerAdapters(config, self.lora_config)
            for _ in range(config.num_hidden_layers)])

    def _delta(self, layer_idx: int, name: str,
               x: torch.Tensor) -> torch.Tensor | None:
        """Route the parent's adapter hook to this layer's `VariantLayerAdapters`.

        The parent's `forward` / `_variant_layer` (which also carry the
        variant-branch gradient-checkpointing support and the
        ``reference_layer_inputs`` shortcut) are inherited unchanged; the hook is
        called only from `_variant_layer`, never from `forward_reference`, so the
        reference stream stays base-weights-only by construction.
        """
        return self.adapters[layer_idx].delta(name, x)


class VariantLoRACarbonForCausalLM(VariantCacheCarbonForCausalLM):
    """Carbon variant cache with variant-only LoRA and a frozen base.

    Differences from the parent that matter to a caller:

    * ``freeze_base()`` (called by the loader) leaves only ``lora_*`` trainable.
    * ``forward`` accepts ``reference_layer_inputs`` to skip the reference pass.
    * ``encode_reference`` produces those under ``no_grad``.
    * the ``bruteforce`` backend is rejected — see ``set_backend``.
    """

    def __init__(self, config, lora: LoRAConfig | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.lora_config = lora or LoRAConfig()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         getattr(config, "pad_token_id", None))
        self.encoder = VariantLoRACarbonEncoder(config, self.lora_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", True):
            self.tie_weights()
        self.backend = "efficient"
        self.base_frozen = True

    def set_backend(self, backend: str) -> None:
        # `forward_bruteforce` runs the stock decoder layers, which cannot see the
        # adapters — it would silently score the *base* model and still look like
        # a valid oracle. Refuse rather than mislead.
        if backend != "efficient":
            raise ValueError(
                f"VariantLoRACarbonForCausalLM supports backend='efficient' only "
                f"(got {backend!r}). The bruteforce oracle runs the stock decoder "
                f"layers, which never see the variant adapters, so it would "
                f"silently score the base model.")
        self.backend = backend

    # ── parameters ────────────────────────────────────────────────────────────

    def freeze_base(self, freeze: bool = True) -> None:
        """Freeze the base, or train it alongside the adapters.

        ``freeze=False`` is strictly more expressive than full fine-tuning, not
        redundant with it. The base weights are *shared* by both streams —
        `forward_reference` runs the whole decoder layer — so moving theta moves
        the reference and variant streams together, while a delta moves the
        variant stream alone. Writing the model as f(theta, delta):

            full fine-tuning  = { f(theta, 0)     }   both streams, locked together
            frozen-base LoRA  = { f(theta_0, d)   }   variant stream only
            freeze=False      = { f(theta, d)     }   strictly contains both

        The cost is that the reference stream stops being a constant of the
        window, so `ReferenceCache` is invalid (it refuses to attach) and the
        reference pass needs its gradient again — set
        ``encoder.reference_checkpointing`` as the full model does. bf16 is also
        unsafe again, since theta is once more something AdamW updates.
        """
        for name, p in self.named_parameters():
            p.requires_grad = True if not freeze else ("lora_" in name)
        self.base_frozen = freeze

    def checkpoint_state_dict(self) -> dict:
        """What has to be saved to reproduce this model from the checkpoint.

        Adapters alone when the base is frozen (~17 MB); everything when it is
        not, because then the base carries part of the learned function and
        saving only the adapters would silently restore a different model.
        """
        return self.lora_state_dict() if self.base_frozen else self.state_dict()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self) -> dict:
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": train, "total": total,
                "fraction": train / total if total else 0.0}

    def lora_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if "lora_" in k}

    def cast_base(self, dtype: torch.dtype) -> "VariantLoRACarbonForCausalLM":
        """Cast everything to ``dtype``, then put the adapters back in fp32.

        `nn.Module.to(dtype=...)` would take the LoRA parameters with it, which
        would throw away the fp32 optimizer state this class exists to keep.
        """
        if dtype != torch.float32 and not getattr(self, "base_frozen", True):
            raise ValueError(
                f"base_dtype={dtype} with a trainable base reintroduces exactly "
                f"the failure bf16 causes in full fine-tuning: AdamW keeps its "
                f"moments in the parameter dtype, so updates ~1e-2 relative fall "
                f"under bf16's ~3.9e-3 relative epsilon and are rounded away. "
                f"bf16 is safe here only while the base is frozen.")
        self.to(dtype=dtype)
        for name, p in self.named_parameters():
            if "lora_" in name:
                p.data = p.data.float()
        return self

    # ── reference stream ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_reference(self, input_ids: torch.LongTensor,
                         attention_mask: Optional[torch.Tensor] = None
                         ) -> List[torch.Tensor]:
        """The frozen per-layer reference residual stream for one window.

        Under ``no_grad``: with the base frozen nothing here needs a graph, which
        is also why gradient checkpointing of the reference pass is unnecessary
        in this class.
        """
        input_ids = input_ids.reshape(-1)
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long,
                                        device=device)
        return self.encoder.forward_reference(
            hidden_states=self.embed_tokens(input_ids),
            attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        variant_positions: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        variant_input_ids: Optional[torch.LongTensor] = None,
        output_logits: bool = True,
        reference_layer_inputs: Optional[List[torch.Tensor]] = None,
    ) -> VariantCacheOutput:
        input_ids = input_ids.reshape(-1)
        ref_seqlen = input_ids.shape[0]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(ref_seqlen, dtype=torch.long, device=device)

        variant_positions = variant_positions.to(device=device).long()
        if variant_input_ids is None:
            variant_input_ids = input_ids.index_select(0, variant_positions)
        variant_input_ids = variant_input_ids.to(device=device).long()
        if variant_input_ids.dim() == 1:
            variant_input_ids = variant_input_ids.unsqueeze(0)

        ref_emb = self.embed_tokens(input_ids)
        variant_cache = self.embed_tokens(variant_input_ids)

        var_hidden = self.encoder.forward(
            variant_cache, variant_positions, ref_emb, attention_mask,
            output_all_encoded_layers=False,
            reference_layer_inputs=reference_layer_inputs)[-1]
        var_hidden = self.encoder.norm(var_hidden)
        logits = self.lm_head(var_hidden) if output_logits else None
        return VariantCacheOutput(last_hidden_state=var_hidden, logits=logits)


class ReferenceCache:
    """Per-window store of the frozen reference stream.

    Correct **only** while the base weights are unchanged, which is what
    `freeze_base` guarantees. Because silently serving a stale stream would look
    like a modelling result rather than a bug, `get` re-checks a fingerprint of a
    base parameter and raises if it moved.

    One entry costs ``n_layers * ref_len * hidden * itemsize``: at Carbon-500M
    (28 layers, 1024 hidden) and a ~168-token window that is ~19 MB in fp32 or
    ~9.6 MB in bf16, so 400 windows fit in ~3.9 GB of host RAM in bf16.
    ``store_device='cpu'`` keeps them off the GPU and pays a transfer per use;
    the default follows the model.
    """

    def __init__(self, model: VariantLoRACarbonForCausalLM, *,
                 store_device: torch.device | str | None = None,
                 store_dtype: torch.dtype | None = None,
                 max_entries: int | None = None):
        self.model = model
        self.store_device = torch.device(store_device) if store_device else None
        self.store_dtype = store_dtype
        self.max_entries = max_entries
        if not getattr(model, "base_frozen", True):
            raise ValueError(
                "ReferenceCache requires a frozen base: with the base trainable "
                "the reference stream changes every step, so a cached one is "
                "stale after the first. Use freeze_base(True), or run without "
                "the cache.")
        self._entries: dict = {}
        self._order: list = []
        self.hits = 0
        self.misses = 0
        self._fingerprint = self._base_fingerprint()

    def _base_fingerprint(self) -> float:
        p = self.model.encoder.layers[0].self_attn.q_proj.weight
        return float(p.detach().flatten()[:256].float().sum())

    def _check_base_unchanged(self) -> None:
        now = self._base_fingerprint()
        if now != self._fingerprint:
            raise RuntimeError(
                "ReferenceCache: base weights changed since the cache was built, "
                "so every cached reference stream is stale. This cache is only "
                "valid with a frozen base — check that freeze_base() ran and that "
                "the optimizer was given model.trainable_parameters().")

    def get(self, key, input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        hit = self._entries.get(key)
        if hit is not None:
            self.hits += 1
        else:
            self._check_base_unchanged()
            self.misses += 1
            ref = self.model.encode_reference(input_ids, attention_mask)
            hit = [t.to(device=self.store_device or t.device,
                        dtype=self.store_dtype or t.dtype) for t in ref]
            self._entries[key] = hit
            self._order.append(key)
            if self.max_entries and len(self._order) > self.max_entries:
                self._entries.pop(self._order.pop(0), None)
        dev = input_ids.device
        want = self.model.embed_tokens.weight.dtype
        return [t.to(device=dev, dtype=want) for t in hit]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for entry in self._entries.values() for t in entry)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0,
                "mb": self.nbytes() / 2**20}

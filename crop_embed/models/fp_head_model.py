#General file for head models with a (#fingerprint x embedding dim) + (Batch x #fingerprint_gather_ind) input and a (Batch x n_traits) output. 
# The FPHeadModel is the main class, and the rest are helper functions and classes.

"""
crop_embed/heads.py
-------------------
Head models that map per-window embeddings (B, n_windows, D) to per-sample
trait predictions (B, n_traits). New heads just need to implement
`forward(window_emb) -> Tensor`.

Implemented:
  - LinearHead    : mean-pool over windows → Linear
  - MLPHead       : mean-pool over windows → 2-layer GELU MLP
  - AttentionHead : K learnable queries cross-attend to windows → MLP

`window_position_features(dataset)` is a helper for the AttentionHead
positional-encoding option.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_embed.dataset import UniqueWindowDataset

def window_position_features(
    dataset: UniqueWindowDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-window (chrom, center_bp) arrays aligned to dataset.partitioner.windows
    order — the same axis that AttentionHead/LinearHead see as `n_windows`.

    Returns
    -------
    chroms    : LongTensor[n_windows] — integer chromosome
    positions : LongTensor[n_windows] — window-center genomic position (bp)
    """
    windows = dataset.partitioner.windows
    chroms    = torch.tensor([w.chrom                  for w in windows], dtype=torch.long)
    positions = torch.tensor([(w.start + w.end) // 2   for w in windows], dtype=torch.long)
    return chroms, positions

def _sinusoidal_pe(
    positions: torch.Tensor, d_model: int, max_position: int
) -> torch.Tensor:
    """Standard sinusoidal positional encoding for integer (bp) positions."""
    pos      = positions.float().unsqueeze(1)                              # (n, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(max_position) / d_model)
    )
    pe = torch.zeros(positions.size(0), d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

class LinearModel(nn.Module):
    def __init__(self, emb_dim: int, n_traits: int) -> None:
        super().__init__()
        self.linear = nn.Linear(emb_dim, n_traits)

    def forward(self, seq_emb: torch.Tensor) -> torch.Tensor:
        # seq_emb: (B, D)
        return self.linear(seq_emb)

class _ResidualBlock(nn.Module):
    """LayerNorm → Linear → GELU → Dropout with a residual skip (in/out dim equal)."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class _LearnedStandardizer(nn.Module):
    """
    Per-dimension learned de-meaning + rescaling: out = (x - mean) / scale.

    Both `mean` and `scale` are learnable, so the layer can drift away from its
    initialization during training. The scale is parametrized as exp(log_scale)
    to stay strictly positive (no sign flips / divide-by-zero as it trains).

    Why: summed fingerprint embeddings have a large, n_gather-dependent mean and
    inflated magnitude. Feeding those straight into the head means the first
    gradients mostly fight that offset/scale instead of learning trait structure.
    Warm-starting from real stats puts the input at ~zero-mean / unit-scale.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))   # scale = exp(0) = 1

    @torch.no_grad()
    def warm_start(self, embeddings: torch.Tensor) -> None:
        """
        Initialize mean/scale from a (N, D) tensor of representative embeddings —
        ideally the *summed* per-sample vectors this layer will actually see, so
        the statistics match its input distribution.
        """
        if embeddings.dim() != 2 or embeddings.size(1) != self.mean.numel():
            raise ValueError(
                f"expected (N, {self.mean.numel()}) embeddings, got {tuple(embeddings.shape)}"
            )
        emb = embeddings.float()
        self.mean.copy_(emb.mean(dim=0))
        std = emb.std(dim=0).clamp_min(self.eps)
        self.log_scale.copy_(std.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * torch.exp(-self.log_scale)

class MLPModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_traits: int,
        hidden_dim: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        )
        self.output = nn.Linear(hidden_dim, n_traits)
    
    def forward(self, seq_emb: torch.Tensor) -> torch.Tensor:
        # seq_emb: (B, D)
        x = self.input_proj(seq_emb)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

class FPSumHeadModel(nn.Module):
    """
    Parametrized head model that gathers and sums over fingerprint embeddings and then applies an arbitrary
    model to predict traits. The model can be a simple linear layer, an MLP, or any other architecture that takes in a (B, D) tensor and outputs (B, n_traits).

    The summed embeddings pass through a learned de-mean + rescale layer
    (`_LearnedStandardizer`) before the model, which conditions the otherwise
    large/biased sum so early gradients are meaningful. Disable with
    `normalize=False`. Pass `warm_start_embeddings` (an (N, D) tensor of
    representative summed embeddings) to initialize that layer from real stats.
    """
    def __init__(
        self,
        model: nn.Module,
        emb_dim: int,
        normalize: bool = True,
        warm_start_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        if normalize:
            self.norm: nn.Module = _LearnedStandardizer(emb_dim)
            if warm_start_embeddings is not None:
                self.norm.warm_start(warm_start_embeddings)
        else:
            if warm_start_embeddings is not None:
                raise ValueError("warm_start_embeddings is only valid when normalize=True")
            self.norm = nn.Identity()

    def forward(self, fp_emb: torch.Tensor, gather_ind: torch.Tensor) -> torch.Tensor:
        # fp_emb: (n_fps, D), gather_ind: (B, n_gather)
        # embedding_bag fuses the gather and the per-row sum, so the (B, n_gather, D)
        # intermediate that fp_emb[gather_ind].sum(1) would build never materializes.
        # A 2D index tensor is treated as one equal-length bag per row -> (B, D).
        summed = F.embedding_bag(gather_ind, fp_emb, mode="mean")   # (B, D)
        return self.model(self.norm(summed))
    
    def forward_postsum(self, summed_emb: torch.Tensor) -> torch.Tensor:
        """
        Alternate forward method that skips the embedding_bag sum 
        and just applies the model to a pre-summed (B, D) tensor. This is for
        debugging / ablations that want to feed in pre-summed embeddings (e.g. train_head.py warm-start).
        """
        return self.model(self.norm(summed_emb))
    
class FPRefDeltaSumHeadModel(FPSumHeadModel):
    """
    Variant of FPSumHeadModel that pools each window's *delta from its reference*
    instead of its absolute embedding.

    For a fingerprint (chrom, w_start, w_end, alt_positions), the reference is the
    variant-free window (chrom, w_start, w_end, ()) — see crop_embed.fingerprint.
    `ref_index[i]` holds the cache row of fingerprint i's reference, so the forward
    gathers `fp_emb[ref_index]` (the per-row `fp_ref_emb`) and subtracts it before
    the embedding_bag pool:

        delta = fp_emb - fp_emb[ref_index]          # reference-subtracted table
        summed = embedding_bag(gather_ind, delta)   # pool the deltas

    A reference fingerprint maps to itself, so its delta is exactly zero and it
    contributes nothing to the (mode="sum") pool — pure-reference windows drop
    out, and the head sees only how each window departs from reference.

    Everything else — the `model`, the learned standardizer, and
    `forward_postsum` — is inherited unchanged from FPSumHeadModel. Note the
    standardizer warm-start (`warm_start_embeddings`) must be fit on summed
    *deltas* to match this model's input distribution; build them by pre-summing
    `subtract_reference(cache, ref_index)` the same way train_head pre-sums the
    raw cache.

    Build `ref_index` once from the fingerprint list that indexes the cache rows
    (the cache's `unique_fingerprints`, equivalently `dataset.unique_fingerprints`)
    via `FPRefDeltaSumHeadModel.build_ref_index(...)`.
    """

    def __init__(
        self,
        model: nn.Module,
        emb_dim: int,
        ref_index: torch.Tensor,
        normalize: bool = True,
        warm_start_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            model, emb_dim,
            normalize=normalize,
            warm_start_embeddings=warm_start_embeddings,
        )
        # (n_fps,) long; row i -> cache row of fingerprint i's reference window.
        # Registered as a buffer so it follows .to(device) and round-trips in the
        # state_dict for reconstruction.
        self.register_buffer("ref_index", ref_index.long())

    @staticmethod
    def build_ref_index(
        unique_fingerprints: list,
        *,
        strict: bool = True,
    ) -> torch.Tensor:
        """
        Map each fingerprint row to the row of its reference (variant-free) window.

        Parameters
        ----------
        unique_fingerprints : the fingerprint list that indexes the cache rows, in
            cache-row order (cache `unique_fingerprints` / `dataset.unique_fingerprints`).
        strict : if True (default), raise when any window's reference fingerprint
            is absent from the cache. If False, those rows map to themselves
            (zero delta), so the window simply drops out of the pool.

        Returns
        -------
        ref_index : LongTensor[n_fps]
        """
        fps = [(c, s, e, tuple(a)) for (c, s, e, a) in unique_fingerprints]
        fp_to_idx = {fp: i for i, fp in enumerate(fps)}

        ref_index = torch.empty(len(fps), dtype=torch.long)
        missing = 0
        for i, (chrom, w_start, w_end, _alts) in enumerate(fps):
            ref_row = fp_to_idx.get((chrom, w_start, w_end, ()))
            if ref_row is None:
                missing += 1
                ref_index[i] = i          # self -> zero delta
            else:
                ref_index[i] = ref_row

        if missing and strict:
            raise ValueError(
                f"{missing}/{len(fps)} fingerprints have no reference (variant-free) "
                "window in the cache, so their baseline can't be subtracted. Pass "
                "strict=False to map those to themselves (zero delta / dropped from "
                "the pool) instead."
            )
        return ref_index

    @staticmethod
    def subtract_reference(
        fp_emb: torch.Tensor, ref_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Reference-subtracted embedding table: `fp_emb - fp_emb[ref_index]`.

        Exposed as a static helper so callers that pre-sum the cache (e.g.
        train_head.py) can pool the deltas with the same embedding_bag they
        already use on the raw cache.
        """
        return fp_emb - fp_emb[ref_index]

    def forward(
        self,
        fp_emb: torch.Tensor,
        gather_ind: torch.Tensor,
        ref_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # fp_emb: (n_fps, D), gather_ind: (B, n_gather)
        # Subtract each window's reference embedding, then pool exactly as
        # FPSumHeadModel does. The delta table backprops into fp_emb.
        #
        # `ref_index` defaults to the stored global buffer (cache-row order). End-
        # to-end training passes a *batch-local* index instead, because there
        # fp_emb is only the batch's unique windows, not the full cache.
        idx    = self.ref_index if ref_index is None else ref_index
        delta  = fp_emb - fp_emb[idx]                               # (n_fps, D)
        summed = F.embedding_bag(gather_ind, delta, mode="mean")    # (B, D)
        return self.model(self.norm(summed))


class FPGatherHeadModel(nn.Module):
    """
    Head model that gathers the per-sample fingerprint embeddings into the full
    (B, n_gather, D) sequence and hands it to a model that consumes the whole set
    rather than a pooled vector (e.g. AttentionHead). 

    Unlike FPSumHeadModel, this deliberately materializes the (B, n_gather, D)
    tensor — that's the point, the downstream model needs every embedding. This
    is a gather (fp_emb[gather_ind]), not a scatter: it reads rows of fp_emb into
    a new (B, n_gather, D) layout, it doesn't write into fp_emb.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, fp_emb: torch.Tensor, gather_ind: torch.Tensor) -> torch.Tensor:
        # fp_emb: (n_fps, D), gather_ind: (B, n_gather)
        gathered = fp_emb[gather_ind]   # (B, n_gather, D); backprops into fp_emb
        return self.model(gathered)


class AttentionHead(nn.Module):
    """
    Cross-attention head: K learnable query vectors attend to the (B, n_windows, D)
    window embeddings, producing (B, K, D). That stack is flattened and passed
    through a 2-layer GELU MLP to n_traits.

    Why this shape: each query can specialize to a different combination of
    windows (e.g. cluster-by-trait, additive-vs-epistatic, chromosome-specific)
    instead of being forced into a single mean pool.

    Parameters
    ----------
    emb_dim         : window embedding dimension (must match the embedder).
    n_traits        : number of regression outputs.
    n_queries       : K — how many distinct "summaries" of the genome to learn.
    n_heads         : attention heads in the cross-attention layer.
    hidden_dim      : MLP hidden width. Defaults to emb_dim.
    positional      : if True, add a learnable per-chromosome embedding and a
                      fixed sinusoidal position embedding to each window
                      before attention. Off by default — the head treats
                      windows as a bag without it.
    window_chroms   : LongTensor[n_windows]. Required when positional=True.
    window_positions: LongTensor[n_windows] in bp. Required when positional=True.
    max_position    : scale used for sinusoidal wavelengths; should exceed the
                      longest chromosome.

    Use `crop_embed.heads.window_position_features(dataset)` to get the
    chroms/positions arrays in the right order.
    """

    def __init__(
        self,
        emb_dim: int,
        n_traits: int,
        n_queries: int = 8,
        n_heads: int = 4,
        hidden_dim: int | None = None,
        positional: bool = False,
        window_chroms: torch.Tensor | None = None,
        window_positions: torch.Tensor | None = None,
        max_position: int = int(5e8),
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or emb_dim

        self.positional = positional
        if positional:
            if window_chroms is None or window_positions is None:
                raise ValueError(
                    "positional=True requires window_chroms and window_positions; "
                    "see crop_embed.heads.window_position_features(dataset)."
                )
            n_chroms = int(window_chroms.max().item()) + 1
            self.chrom_emb = nn.Embedding(n_chroms, emb_dim)
            self.register_buffer("window_chroms", window_chroms.long())
            self.register_buffer(
                "pos_emb", _sinusoidal_pe(window_positions, emb_dim, max_position)
            )

        self.norm    = nn.LayerNorm(emb_dim)
        self.queries = nn.Parameter(torch.randn(n_queries, emb_dim) * 0.02)
        self.attn    = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.mlp     = nn.Sequential(
            nn.Linear(n_queries * emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_traits),
        )

    def forward(self, window_emb: torch.Tensor) -> torch.Tensor:
        # window_emb: (B, n_windows, D)
        if self.positional:
            window_emb = window_emb + self.chrom_emb(self.window_chroms) + self.pos_emb
        x = self.norm(window_emb)
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)   # (B, K, D)
        pooled, _ = self.attn(q, x, x, need_weights=False)        # (B, K, D)
        return self.mlp(pooled.flatten(1))

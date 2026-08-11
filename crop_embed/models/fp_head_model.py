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


def window_position_features_from_cache(
    unique_fingerprints: list,
    sample_fp_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-window (chrom, center_bp) in window-column order, reconstructed from a
    cache alone (``unique_fingerprints`` + ``sample_fp_index``) when no partitioner
    is available — the fast ``CachedWindows`` / ``from_cached_windows`` path.

    Every fingerprint in window column j shares that window's (chrom, start, end)
    and differs only in its alt positions (see ``build_sample_fp_index``), so any
    sample's cache row for column j is a valid representative. Row 0 is used.

    Returns (chroms LongTensor[n_windows], positions LongTensor[n_windows]) — the
    same arrays ``window_position_features`` yields, ready for AttentionHead."""
    rep_rows = sample_fp_index[0].tolist()          # one cache row per window column
    chroms, positions = [], []
    for r in rep_rows:
        chrom, start, end, _ = unique_fingerprints[r]
        chroms.append(chrom)
        positions.append((start + end) // 2)
    return (torch.tensor(chroms, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long))

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


class WindowPositionalEncoding(nn.Module):
    """Additive genomic positional encoding for windows. Each window has a
    chromosome and a within-chromosome bp position; this maps those to an additive
    (n_windows, emb_dim) vector added to the (standardized) window embeddings before
    attention, gated by a learnable scalar so the head can dial positional strength.

    Swappable ``mode`` (this is the experimentation surface):
      none        — no-op (returns None).
      chrom       — per-chromosome learned embedding only.
      sinusoidal  — chrom + fixed log-spaced sinusoidal encoding of bp (Vaswani).
      learned     — chrom + a learned embedding per within-genome bp bucket.
      fourier     — chrom + a learned linear map of log-spaced sin/cos bp features
                    (learnable-ish generalization of sinusoidal; more expressive).

    RoPE / relative schemes need rotary inside the attention (not an additive term),
    so they live with a custom attention layer, not here.
    """

    VALID = ("none", "chrom", "sinusoidal", "learned", "fourier")

    def __init__(
        self,
        emb_dim: int,
        chroms: torch.Tensor,
        positions: torch.Tensor,
        mode: str = "sinusoidal",
        *,
        n_buckets: int = 128,
        n_fourier: int = 64,
        min_wavelength: float = 1e2,
        max_wavelength: float = 5e7,
    ) -> None:
        super().__init__()
        if mode not in self.VALID:
            raise ValueError(f"unknown pos mode {mode!r}; use one of {self.VALID}")
        self.mode = mode
        self.register_buffer("chroms", chroms.long())
        self.register_buffer("positions", positions.long())
        if mode == "none":
            return
        self.gate = nn.Parameter(torch.ones(()))
        n_chroms = int(chroms.max().item()) + 1
        self.chrom_emb = nn.Embedding(n_chroms, emb_dim)
        nn.init.normal_(self.chrom_emb.weight, std=0.5)

        if mode == "sinusoidal":
            self.register_buffer("pe", _sinusoidal_pe(positions, emb_dim, int(max_wavelength)))
        elif mode == "learned":
            maxpos = int(positions.max().item()) + 1
            bucket = (positions.float() / maxpos * n_buckets).long().clamp_(max=n_buckets - 1)
            self.register_buffer("bucket", bucket)
            self.bucket_emb = nn.Embedding(n_buckets, emb_dim)
            nn.init.normal_(self.bucket_emb.weight, std=0.5)
        elif mode == "fourier":
            freqs = torch.exp(torch.linspace(
                math.log(1.0 / max_wavelength), math.log(1.0 / min_wavelength), n_fourier))
            ang = positions.float().unsqueeze(1) * (2 * math.pi * freqs).unsqueeze(0)  # (n, F)
            self.register_buffer("fourier_feats", torch.cat([ang.sin(), ang.cos()], dim=1))
            self.fourier_proj = nn.Linear(2 * n_fourier, emb_dim)

    def forward(self) -> torch.Tensor | None:
        """(n_windows, emb_dim) additive encoding, or None for mode 'none'."""
        if self.mode == "none":
            return None
        pe = self.chrom_emb(self.chroms)
        if self.mode == "sinusoidal":
            pe = pe + self.pe
        elif self.mode == "learned":
            pe = pe + self.bucket_emb(self.bucket)
        elif self.mode == "fourier":
            pe = pe + self.fourier_proj(self.fourier_feats)
        return self.gate * pe


@torch.no_grad()
def window_means(
    cache: torch.Tensor,
    sample_fp_index: torch.Tensor,
    sample_idx: torch.Tensor | None = None,
    chunk: int = 16,
) -> torch.Tensor:
    """
    Per-window mean embedding across samples: ``m_j = mean_s cache[sample_fp_index[s, j]]``.

    For each window position j, average that window's embedding over the chosen
    samples — weighting each haplotype by how many samples carry it (samples that
    share a fingerprint share its cache row). The result, subtracted per window,
    strips the dominant shared per-window component so each window vector becomes
    the per-sample deviation. Feed it to ``AttentionHead(window_mean=...)``.

    Parameters
    ----------
    cache           : (n_fps, D) embedding table (FixedWindowEmbedder.cache).
    sample_fp_index : (n_samples, n_windows) sample→window→cache-row map.
    sample_idx      : samples to average over. Pass the TRAIN indices to avoid
                      leaking val embeddings into the centering; None uses all.
    chunk           : samples per accumulation step, so the (S, n_windows, D)
                      gather never fully materializes.

    Returns
    -------
    (n_windows, D) float tensor on ``cache.device``.
    """
    device = cache.device
    if sample_idx is None:
        sample_idx = torch.arange(sample_fp_index.shape[0], device=device)
    sample_idx = sample_idx.to(device)
    acc = torch.zeros(sample_fp_index.shape[1], cache.shape[1],
                      device=device, dtype=torch.float32)
    for i in range(0, len(sample_idx), chunk):
        b = sample_idx[i:i + chunk]
        acc += cache[sample_fp_index[b]].float().sum(0)
    return acc / len(sample_idx)


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
        emb = embeddings.float().to(self.mean.device)
        self.mean.copy_(emb.mean(dim=0))
        std = emb.std(dim=0).clamp_min(self.eps)
        self.log_scale.copy_(std.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * torch.exp(-self.log_scale)

class _RMSStandardizer(nn.Module):
    """
    Per-dimension de-meaning + a SINGLE scalar rescale: out = (x - mean) / scale.

    Same shape API and warm-start contract as _LearnedStandardizer (learnable,
    drifts during training, scale = exp(log_scale) to stay positive), with one
    difference: `mean` is per-dim (D,) but `log_scale` is a scalar, so the whole
    vector is divided by one number — its root-mean-square magnitude — rather than
    each dim by its own std.

    Why a scalar scale: per-dim whitening divides dim j by its own across-sample
    std σ_j. Under mean pooling, averaging over thousands of windows collapses every
    σ_j to ~1e-5, so the per-dim divisor (i) needs an absolute eps floor that then
    sits right in the signal and clamps ~a quarter of the dims, and (ii) amplifies
    the lowest-variance, least-informative dims by up to 1/σ_j — ill-conditioning
    the head and, because the standardizer backward multiplies the gradient into the
    windows by 1/scale, inflating the embedder gradient by the same factor (the very
    blow-up sum pooling is avoided to dodge). A single RMS scale lifts the vector to
    ~unit magnitude with no per-dim amplification, so it is scale-invariant across
    pool modes and gradient-safe for end-to-end. Near-constant dims simply stay small
    instead of being whitened up to unit variance.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(()))    # scalar; scale = exp(0) = 1

    @torch.no_grad()
    def warm_start(self, embeddings: torch.Tensor) -> None:
        """
        Initialize mean (per-dim) and the scalar scale from a (N, D) tensor of
        representative embeddings — ideally the *pooled* per-sample vectors this
        layer will actually see. The scale is the RMS magnitude of the de-meaned
        embeddings over all elements, so the de-meaned vector lands at ~unit RMS.
        """
        if embeddings.dim() != 2 or embeddings.size(1) != self.mean.numel():
            raise ValueError(
                f"expected (N, {self.mean.numel()}) embeddings, got {tuple(embeddings.shape)}"
            )
        emb = embeddings.float().to(self.mean.device)
        self.mean.copy_(emb.mean(dim=0))
        rms = (emb - self.mean).pow(2).mean().sqrt().clamp_min(self.eps)   # scalar
        self.log_scale.copy_(rms.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * torch.exp(-self.log_scale)

class _ScalarScale(nn.Module):
    """Multiply by a single learnable scalar. A *linear* (additivity-preserving)
    alternative to LayerNorm for lifting tiny centered window vectors to a usable
    magnitude without the per-window nonlinear renormalization LayerNorm imposes —
    so mean/attention over the scaled windows still tracks the linear mean signal."""

    def __init__(self, init: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class MLPModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_traits: int,
        hidden_dim: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.0,
        bottleneck_dim: int | None = None,
    ) -> None:
        super().__init__()
        # Optional learnable linear bottleneck: a plain Linear(emb_dim -> bottleneck_dim)
        # with NO activation — a *learned* dimensionality reduction (the trainable analog
        # of an SVD preprocessing step) applied before the MLP proper. Decoupled from the
        # block width: the residual stack then runs at `hidden_dim`, which defaults to the
        # bottleneck (or emb_dim when there's no bottleneck), so you can reduce-then-expand.
        if bottleneck_dim is not None:
            self.bottleneck: nn.Module | None = nn.Linear(emb_dim, bottleneck_dim)
            proj_in = bottleneck_dim
        else:
            self.bottleneck = None
            proj_in = emb_dim
        hidden_dim = hidden_dim or proj_in
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        )
        self.output = nn.Linear(hidden_dim, n_traits)

    def forward(self, seq_emb: torch.Tensor) -> torch.Tensor:
        # seq_emb: (B, D)
        x = seq_emb if self.bottleneck is None else self.bottleneck(seq_emb)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

class FPSumHeadModel(nn.Module):
    """
    Parametrized head model that gathers and sums over fingerprint embeddings and then applies an arbitrary
    model to predict traits. The model can be a simple linear layer, an MLP, or any other architecture that takes in a (B, D) tensor and outputs (B, n_traits).

    The summed embeddings pass through a learned de-mean + rescale layer before
    the model, which conditions the otherwise large/biased sum so early gradients
    are meaningful. `standardizer` picks the layer: "perdim" (`_LearnedStandardizer`,
    per-dim std rescale) or "rms" (`_RMSStandardizer`, a single scalar RMS rescale —
    prefer it for mean pooling, where per-dim stds collapse to ~eps). Disable either
    with `normalize=False`. Pass `warm_start_embeddings` (an (N, D) tensor of
    representative pooled embeddings) to initialize the layer from real stats.

    `freeze_standardizer` makes the warm-started mean/scale non-trainable. Strongly
    recommended (and the train_head.py default) when warm-starting under mean pooling:
    lifting the ~1e-5 signal to unit scale needs a ~1/scale ≈ 2e4 amplification, and
    a trainable mean/log_scale jitters by ~lr each AdamW step in *absolute* units — so
    that jitter gets multiplied by the amplification into a huge swing in the layer's
    output, swamping the signal. Freezing pins the stats so the input stays at unit
    scale. (For sum pooling the amplification is ~1, so it matters far less there.)
    """
    def __init__(
        self,
        model: nn.Module,
        emb_dim: int,
        normalize: bool = True,
        warm_start_embeddings: torch.Tensor | None = None,
        pool: str = "sum",
        standardizer: str = "perdim",
        freeze_standardizer: bool = False,
    ) -> None:
        super().__init__()
        if pool not in ("sum", "mean"):
            raise ValueError(f"pool must be 'sum' or 'mean', got {pool!r}")
        if standardizer not in ("perdim", "rms"):
            raise ValueError(f"standardizer must be 'perdim' or 'rms', got {standardizer!r}")
        # How windows are pooled into the (B, D) per-sample vector: "sum" (magnitude
        # grows with #windows) or "mean" (window-count invariant). Plain string, so it
        # is NOT in state_dict — reconstruction must pass it (saved in head_config).
        # forward_postsum callers (train_head.py) must pre-pool with this SAME mode, or
        # the head trains/infers on a different scale than it was built for.
        self.pool = pool
        # Which standardizer the .norm slot holds. Also a plain string (the chosen
        # layer's parameter SHAPES differ — per-dim vs scalar log_scale — so a
        # reconstructing caller must pass the same value before load_state_dict).
        self.standardizer = standardizer
        # Whether the standardizer's stats are frozen. requires_grad is NOT part of the
        # state_dict, so a reconstructing caller must pass this again to match.
        self.freeze_standardizer = freeze_standardizer
        self.model = model
        if normalize:
            std_cls = _RMSStandardizer if standardizer == "rms" else _LearnedStandardizer
            self.norm: nn.Module = std_cls(emb_dim)
            if warm_start_embeddings is not None:
                self.norm.warm_start(warm_start_embeddings)
            if freeze_standardizer:
                for p in self.norm.parameters():
                    p.requires_grad_(False)
        else:
            if warm_start_embeddings is not None:
                raise ValueError("warm_start_embeddings is only valid when normalize=True")
            self.norm = nn.Identity()

    def forward(self, fp_emb: torch.Tensor, gather_ind: torch.Tensor) -> torch.Tensor:
        # fp_emb: (n_fps, D), gather_ind: (B, n_gather)
        # embedding_bag fuses the gather and the per-row pool, so the (B, n_gather, D)
        # intermediate that fp_emb[gather_ind].sum(1) would build never materializes.
        # A 2D index tensor is treated as one equal-length bag per row -> (B, D).
        pooled = F.embedding_bag(gather_ind, fp_emb, mode=self.pool)   # (B, D)
        return self.model(self.norm(pooled))

    def forward_postsum(self, pooled_emb: torch.Tensor) -> torch.Tensor:
        """
        Alternate forward that skips the embedding_bag pool and just applies the model
        to a pre-pooled (B, D) tensor. The caller MUST pre-pool with the same `pool`
        mode this head was built with (see self.pool) so the scale matches `forward`.
        Used by train_head.py, which pre-pools the frozen cache once up front.
        """
        return self.model(self.norm(pooled_emb))
    
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

    A reference fingerprint maps to itself, so its delta is exactly zero. Under
    `pool="sum"` such a window contributes nothing, so pure-reference windows drop
    out of the pool entirely and the head sees only how each window departs from
    reference. Under `pool="mean"` a zero-delta window still counts toward the bag
    size, so it pulls the averaged delta toward zero rather than dropping out.

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
        pool: str = "sum",
        standardizer: str = "perdim",
        freeze_standardizer: bool = False,
    ) -> None:
        super().__init__(
            model, emb_dim,
            normalize=normalize,
            warm_start_embeddings=warm_start_embeddings,
            pool=pool,
            standardizer=standardizer,
            freeze_standardizer=freeze_standardizer,
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
        pooled = F.embedding_bag(gather_ind, delta, mode=self.pool)  # (B, D)
        return self.model(self.norm(pooled))


class FPCenteredSumHeadModel(FPSumHeadModel):
    """
    Variant of FPSumHeadModel that pools each window's *deviation from its
    across-sample mean* instead of its absolute embedding — the symmetric
    counterpart to FPRefDeltaSumHeadModel.

    Where refdelta subtracts each window's variant-free *reference* embedding
    (zeroing reference windows, privileging them), this subtracts window j's mean
    over samples, m_j: every window — reference or not — becomes its per-sample
    deviation, and no window is forced to exactly zero. `window_center[i]` holds
    the m_j of fingerprint i's window (fingerprints in the same window share it),
    so the forward gathers it and subtracts before the embedding_bag pool:

        delta  = fp_emb - window_center            # per-fp deviation table
        summed = embedding_bag(gather_ind, delta)  # pool the deviations

    IMPORTANT — interaction with the standardizer. Under sum/mean pooling this
    subtraction is a *constant per-sample shift* (every sample has the same window
    set), so a learned standardizer (normalize=True) absorbs it and the head is
    mathematically identical to the absolute FPSumHeadModel. It only differs from
    absolute under ``normalize=False`` (the standardizer-free regime), where the
    centering replaces what the standardizer would otherwise do. (The same caveat
    applies to FPRefDeltaSumHeadModel.) Build `window_center` from the TRAINING
    samples (see `build_window_center`) to avoid val leakage.
    """

    def __init__(
        self,
        model: nn.Module,
        emb_dim: int,
        window_center: torch.Tensor,
        normalize: bool = True,
        warm_start_embeddings: torch.Tensor | None = None,
        pool: str = "sum",
        standardizer: str = "perdim",
        freeze_standardizer: bool = False,
    ) -> None:
        super().__init__(
            model, emb_dim,
            normalize=normalize,
            warm_start_embeddings=warm_start_embeddings,
            pool=pool,
            standardizer=standardizer,
            freeze_standardizer=freeze_standardizer,
        )
        # (n_fps, D): row i is the across-sample mean of fingerprint i's window.
        # Buffer so it follows .to(device) and round-trips in the state_dict.
        self.register_buffer("window_center", window_center.float())

    @staticmethod
    def build_window_center(
        cache: torch.Tensor,
        sample_fp_index: torch.Tensor,
        sample_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Per-fingerprint window-center table (n_fps, D): for each fingerprint, the
        mean embedding of its window across `sample_idx` samples (pass the TRAIN
        indices to avoid val leakage; None = all). Built from `window_means` plus
        the fingerprint→window-column map implied by `sample_fp_index`.
        """
        device = cache.device
        n_fps = cache.shape[0]
        n_samples, n_windows = sample_fp_index.shape
        wmean = window_means(cache, sample_fp_index, sample_idx)        # (n_windows, D)
        # Each fingerprint lives in exactly one window column, so scattering the
        # column index over the (n_samples, n_windows) map is consistent.
        cols = torch.arange(n_windows, device=device).expand(n_samples, n_windows)
        fp_to_col = torch.empty(n_fps, dtype=torch.long, device=device)
        fp_to_col[sample_fp_index.reshape(-1)] = cols.reshape(-1)
        return wmean[fp_to_col]                                         # (n_fps, D)

    @staticmethod
    def subtract_center(fp_emb: torch.Tensor, window_center: torch.Tensor) -> torch.Tensor:
        """Centered embedding table: `fp_emb - window_center`. Exposed so callers
        that pre-pool the cache (train_head.py) pool the deviations the same way."""
        return fp_emb - window_center

    def forward(
        self,
        fp_emb: torch.Tensor,
        gather_ind: torch.Tensor,
        window_center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Subtract each window's across-sample mean, then pool exactly as
        # FPSumHeadModel does. `window_center` defaults to the stored buffer;
        # end-to-end training would pass a batch-local table.
        center = self.window_center if window_center is None else window_center
        delta  = fp_emb - center                                     # (n_fps, D)
        pooled = F.embedding_bag(gather_ind, delta, mode=self.pool)  # (B, D)
        return self.model(self.norm(pooled))


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
    through an MLP head to n_traits (2-layer by default; deeper via mlp_layers).

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
    mlp_layers      : residual GELU blocks in the head on top of the attention
                      (0 = the original input-proj → output 2-layer MLP).
    mlp_dropout     : dropout in that head's projection and residual blocks.
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
        pos_encoding: str = "none",
        window_chroms: torch.Tensor | None = None,
        window_positions: torch.Tensor | None = None,
        max_position: int = int(5e8),
        input_standardize: bool = False,
        layernorm: bool = True,
        output_standardize: bool = False,
        standardizer: str = "rms",
        window_mean: torch.Tensor | None = None,
        window_scale: float | None = None,
        mlp_layers: int = 0,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or emb_dim

        # Per-window mean (n_windows, D) subtracted from each window before attention:
        # window j's embedding minus its across-sample mean m_j. Unlike refdelta (which
        # zeroes reference windows and privileges them), this is symmetric — a window
        # that's reference for this sample becomes a small "everyone-else-varies-here"
        # vector, not exactly zero. It removes the dominant shared per-window component
        # so each window vector IS the per-sample deviation (where the signal lives).
        # No-op for sum/mean pooling heads (a constant shift the standardizer absorbs);
        # it matters here because attention sees windows individually. Build m_j from
        # the TRAINING samples (see the trainer) to avoid val leakage.
        self.center_windows = window_mean is not None
        if self.center_windows:
            self.register_buffer("window_mean", window_mean.clone().float())

        # Swappable additive genomic positional encoding (see WindowPositionalEncoding).
        self.pos_encoding = pos_encoding
        if pos_encoding != "none":
            if window_chroms is None or window_positions is None:
                raise ValueError(
                    f"pos_encoding={pos_encoding!r} requires window_chroms and "
                    "window_positions; see window_position_features_from_cache(...).")
            self.pos_enc: nn.Module = WindowPositionalEncoding(
                emb_dim, window_chroms, window_positions, mode=pos_encoding,
                max_wavelength=float(max_position))
        else:
            self.pos_enc = nn.Identity()

        # Per-dimension de-mean/rescale of the window embeddings BEFORE attention,
        # warm-startable from training stats (see warm_start_input). This is the
        # piece FPSumHeadModel relies on: absolute window embeddings carry a large
        # common per-dim offset and a tiny across-sample signal, and LayerNorm —
        # which normalizes each window vector across D, not each dim across the
        # population — doesn't surface it. The standardizer whitens per-dim so the
        # signal isn't denormalized away. Off by default (back-compat).
        # 'rms' (scalar RMS rescale) vs 'perdim' (per-dim std). Default RMS: the
        # attended aggregate is pool-like, so its per-dim across-sample stds collapse
        # to ~1e-5 and per-dim whitening amplifies the lowest-variance dims by ~1/σ,
        # exploding the output (val_loss 1e4) and collapsing training. A single RMS
        # scalar lifts to ~unit magnitude without that per-dim amplification.
        _std_cls = _RMSStandardizer if standardizer == "rms" else _LearnedStandardizer
        self.in_std: nn.Module = _std_cls(emb_dim) if input_standardize else nn.Identity()
        # Per-window normalization, applied before attention. Three modes, so the
        # effect of LayerNorm specifically can be isolated:
        #   window_scale set -> learnable scalar upscale (linear; keeps additivity)
        #   layernorm        -> LayerNorm over D per window (default; nonlinear)
        #   neither          -> Identity
        if window_scale is not None:
            self.norm: nn.Module = _ScalarScale(window_scale)
        elif layernorm:
            self.norm = nn.LayerNorm(emb_dim)
        else:
            self.norm = nn.Identity()
        self.queries = nn.Parameter(torch.randn(n_queries, emb_dim) * 0.02)
        self.attn    = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        # Per-dim de-mean/rescale of the ATTENDED summary (flattened (K*D)) before the
        # MLP — the post-aggregation analog of FPSumHeadModel's standardizer. This is
        # where the amplification has to happen: the across-sample signal only emerges
        # after aggregation, so whitening it here (warm_start_output) lifts it from
        # ~1e-5 to ~unit scale where the MLP can learn. Off by default.
        self.out_std: nn.Module = (
            _std_cls(n_queries * emb_dim) if output_standardize else nn.Identity()
        )
        # Head on top of the flattened attended summary (K*D → n_traits). Reuses
        # MLPModel: input projection → `mlp_layers` residual GELU blocks → output.
        # mlp_layers=0 (default) is input-proj → output, i.e. the original 2-layer
        # GELU MLP; raise it (and mlp_dropout) for a deeper head over the attention.
        self.mlp = MLPModel(
            n_queries * emb_dim, n_traits,
            hidden_dim=hidden_dim, n_layers=mlp_layers, dropout=mlp_dropout,
        )

    @torch.no_grad()
    def warm_start_input(self, window_embeddings: torch.Tensor) -> None:
        """Fit the input standardizer from a (N, D) sample of window embeddings
        (no-op unless input_standardize=True). Pass the training-set window
        embeddings so its per-dim mean/scale match the attention input."""
        if hasattr(self.in_std, "warm_start"):
            self.in_std.warm_start(window_embeddings)

    @torch.no_grad()
    def warm_start_output(self, attended_feats: torch.Tensor) -> None:
        """Fit the output standardizer from a (N, K*D) sample of attended summaries
        (no-op unless output_standardize=True). Produce the feats by running
        `attend()` over the training samples first."""
        if hasattr(self.out_std, "warm_start"):
            self.out_std.warm_start(attended_feats)

    @torch.no_grad()
    def init_attention_as_mean(self) -> None:
        """Initialize the cross-attention so its output is the UNIFORM MEAN of the
        (normalized) windows: zero the query/key projections (so every score is 0 →
        softmax is uniform), set the value projection and the output projection to
        identity. Every query then emits mean_j x_j, so the flattened output is that
        mean (K copies for K>1). This starts attention *at* the working mean-pool —
        the only starting point from which it reliably trains on these numerically
        nasty windows (random-query init collapses) — and it can only deviate from
        uniform as it learns. With K>1 the queries differentiate once training breaks
        their symmetry (their distinct query vectors get non-zero projections)."""
        D = self.queries.shape[1]
        self.attn.in_proj_weight.zero_()                       # Q, K = 0  -> uniform softmax
        self.attn.in_proj_weight[2 * D:3 * D].copy_(torch.eye(D))   # V = identity
        if self.attn.in_proj_bias is not None:
            self.attn.in_proj_bias.zero_()
        self.attn.out_proj.weight.copy_(torch.eye(D))
        self.attn.out_proj.bias.zero_()

    def attend(self, window_emb: torch.Tensor) -> torch.Tensor:
        """Everything up to (but not through) the MLP: returns the flattened
        attended summary (B, K*D). Exposed so warm_start_output can fit on it."""
        if self.center_windows:
            window_emb = window_emb - self.window_mean  # subtract per-window across-sample mean
        window_emb = self.in_std(window_emb)            # per-dim whiten (broadcasts over B, windows)
        x = self.norm(window_emb)                                # VALUES: positional-free
        # Positional goes on the KEYS only, not the values: attention uses genomic
        # position to decide WHERE to look, but the pooled output carries no positional
        # mass. Injecting it into the values instead makes any attention re-weighting
        # move a large positional term through the frozen out_std (~1e4 amplification)
        # and explode — pooling only the (signal) values keeps out_std valid.
        k = x if self.pos_encoding == "none" else x + self.pos_enc()   # (…broadcasts over B)
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)   # (B, K, D)
        pooled, _ = self.attn(q, k, x, need_weights=False)        # query, key=k(+pos), value=x
        return pooled.flatten(1)

    def forward(self, window_emb: torch.Tensor) -> torch.Tensor:
        # window_emb: (B, n_windows, D)
        return self.mlp(self.out_std(self.attend(window_emb)))

"""
crop_embed/train.py
-------------------
Minimum viable end-to-end training stack: sample IDs → window embeddings →
head model → masked loss → backprop. The embedder and head are passed in as
parameters so any DNA encoder (DNABERT-2, PlantCaduceus, ...) and any head
(LinearHead, MLP, attention-pool, ...) plug in unchanged.

Usage
-----
    from crop_embed import UniqueWindowDataset, SNPWindowPartitioner, WindowEmbedder
    from crop_embed.train import LinearHead, train
    from DNABERT2_modules import load_dnabert2

    model, tokenizer = load_dnabert2()
    embedder = WindowEmbedder(model, tokenizer, max_length=512)
    head     = LinearHead(emb_dim=768, n_traits=Y.shape[1])

    # Y: Tensor[n_samples, n_traits] aligned to dataset.samples order, NaN = missing.
    train(embedder, head, dataset, Y, batch_size=32, steps=500)

For head-only training on precomputed embeddings, swap WindowEmbedder for
CachedWindowEmbedder — it has no parameters, so the optimizer updates only
the head:

    from crop_embed import CachedWindowEmbedder
    embedder = CachedWindowEmbedder.from_checkpoint("checkpoints/v1/sativas413.ckpt.pt")
    train(embedder, head, dataset, Y, ...)
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from crop_embed.dataset import UniqueWindowDataset
from crop_embed.embedder import WindowEmbedder


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


class LinearHead(nn.Module):
    """Reference head: mean-pool over windows → Linear → n_traits."""

    def __init__(self, emb_dim: int, n_traits: int) -> None:
        super().__init__()
        self.linear = nn.Linear(emb_dim, n_traits)

    def forward(self, window_emb: torch.Tensor) -> torch.Tensor:
        # window_emb: (B, n_windows, D)
        return self.linear(window_emb.mean(dim=1))


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

    Use `crop_embed.train.window_position_features(dataset)` to get the
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
                    "see crop_embed.train.window_position_features(dataset)."
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


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over non-NaN target entries. Empty mask returns a zero with grad."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()


def train(
    embedder: WindowEmbedder,
    head: nn.Module,
    dataset: UniqueWindowDataset,
    targets: torch.Tensor,
    *,
    loss_fn=masked_mse,
    batch_size: int = 32,
    lr: float = 1e-4,
    steps: int = 1000,
    precision: str = "fp32",
    device: str | torch.device | None = None,
    log_every: int = 50,
) -> None:
    """
    Run gradient descent on whichever parameters of (embedder + head) currently
    have `requires_grad=True`. Frozen params are skipped (no optimizer state).

    For two-phase fine-tuning, call this twice:
      1. Freeze the embedder (`for p in embedder.parameters(): p.requires_grad_(False)`),
         call train(...) — only the head updates.
      2. Unfreeze the embedder, call train(...) again — full end-to-end.
    Adam state for the head resets at the phase boundary; for most settings this
    costs only a few warmup steps.

    Parameters
    ----------
    embedder   : WindowEmbedder — wraps a DNA encoder + tokenizer + pooling.
    head       : nn.Module mapping (B, n_windows, D) → (B, n_traits).
    dataset    : UniqueWindowDataset; rows of `dataset.samples` define sample order.
    targets    : Tensor[n_samples, n_traits] aligned to `dataset.samples` order.
                 NaN entries are masked out of the loss.
    loss_fn    : (pred, target) → scalar tensor. Default: masked_mse.
    precision  : "fp32" | "bf16" | "fp16". fp16 uses GradScaler; bf16 doesn't.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    embedder.to(device).train()
    head.to(device).train()
    targets = targets.to(device)

    # Mixed precision
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    if amp_dtype is None:
        autocast_ctx_factory = nullcontext
    else:
        autocast_ctx_factory = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None

    # Only optimize params that are currently trainable. Lets the caller freeze
    # the embedder by setting requires_grad=False externally.
    params = [
        p for p in list(embedder.parameters()) + list(head.parameters())
        if p.requires_grad
    ]
    if not params:
        raise ValueError("train(): no parameters have requires_grad=True.")
    opt = torch.optim.AdamW(params, lr=lr)

    n_samples = len(dataset.samples)
    for step in range(steps):
        idx = torch.randint(n_samples, (batch_size,))
        sequences, fingerprints, inverse = dataset.gather_batch(idx)

        with autocast_ctx_factory():
            emb     = embedder(sequences, fingerprints)     # (N_unique, D)
            windows = emb[inverse.to(device)]               # (B, n_windows, D)
            y_hat   = head(windows)                         # (B, n_traits)
            loss    = loss_fn(y_hat, targets[idx])

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if step % log_every == 0:
            print(f"step {step:5d}  loss={loss.item():.4f}  "
                  f"unique_windows={len(sequences):,}")

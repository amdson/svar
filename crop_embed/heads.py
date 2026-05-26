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


class LinearHead(nn.Module):
    """Reference head: mean-pool over windows → Linear → n_traits."""

    def __init__(self, emb_dim: int, n_traits: int) -> None:
        super().__init__()
        self.linear = nn.Linear(emb_dim, n_traits)

    def forward(self, window_emb: torch.Tensor) -> torch.Tensor:
        # window_emb: (B, n_windows, D)
        return self.linear(window_emb.mean(dim=1))


class MLPHead(nn.Module):
    """
    Mean-pool over windows → 2-layer GELU MLP → n_traits.

    A step up from LinearHead when the relationship between pooled-embedding
    coordinates and traits isn't well captured by a single affine map.

    Parameters
    ----------
    emb_dim    : window embedding dimension (must match the embedder).
    n_traits   : number of regression outputs.
    hidden_dim : MLP hidden width. Defaults to emb_dim.
    dropout    : dropout probability applied between the two linear layers.
    """

    def __init__(
        self,
        emb_dim: int,
        n_traits: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_traits),
        )

    def forward(self, window_emb: torch.Tensor) -> torch.Tensor:
        # window_emb: (B, n_windows, D)
        return self.mlp(window_emb.mean(dim=1))


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

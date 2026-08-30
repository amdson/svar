"""
crop_embed/partitioner.py
-------------------------
SNPWindowPartitioner: assigns each SNP to exactly one genomic window.

Two window modes; **disjoint is the default**.

``disjoint`` (default) — window *spans* never overlap, so every SNP inside a
window's sequence is a SNP that window owns.

    Greedy, left-to-right per chromosome. Open a window at the leftmost
    unassigned SNP p, but clamp the left edge to the previous window's right edge
    so spans abut rather than overlap:

        w_start = max(p - half_window, prev_window_end)
        w_end   = w_start + 2*half_window
        (claim every SNP in [w_start, w_end); the next window opens at the first
         SNP >= w_end)

    Requires ``buffer == 0``: once windows are contiguous a right-edge buffer
    would leave boundary SNPs unclaimed. An individual window whose opening SNP is
    >= 2*half_window past the previous window's end is not clamped, and is then
    exactly [p-half_window, p+half_window) — identical to the ``overlap`` window
    for that SNP. This is a per-window property, not a global one: once one clamp
    fires every later window on that chromosome shifts, so the two modes give
    different window *counts* and a different window *order* even on sparse data.
    Measured on rice chr1 at half_window=500 (1 SNP per 8.7 kb, 41% of adjacent
    pairs closer than 2*half_window): 3,642 disjoint vs 3,738 overlap windows, of
    which 2,967 spans (81%) coincide.

    In dense regions windows are shifted right to abut, so they are no longer
    centred on their opening SNP and a dense region becomes a contiguous tiling of
    2*half_window-wide bins. On arabidopsis chr4 at half_window=500 that is 16,278
    windows averaging 22.9 owned SNPs, against 27,484 windows averaging 13.3 owned
    plus 10.6 foreign under ``overlap``.

``overlap`` (legacy) — window spans are [p-half_window, p+half_window] and
consecutive spans overlap by up to half_window.

    Each SNP is still *assigned* to exactly one window, but a window's genomic
    span contains SNPs owned by earlier windows — measured on arabidopsis chr4 at
    half_window=500, 44% of the segregating sites inside a window's span belong to
    a different window, and all of them sit upstream of the window's first owned
    SNP. Anything that reconstructs a window's sequence therefore pins those sites
    at the reference allele, producing a haplotype the individual does not carry
    and withholding exactly the upstream context that carries linkage information.
    That is why the default changed.

    Kept because caches, embeddings and published figures were built this way and
    have to stay reproducible — NOT because it is a reasonable default.

**Cache compatibility.** The two modes produce different windows *and a different
window order*, so a cache built under one mode is meaningless under the other.
Embedding caches are addressed by (crop, backbone, half_window) with no mode in
the path, so this is not caught by the filename: pass the mode a cache was built
with explicitly, and see ``train_pipeline/embed_windows.py``'s metadata check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from crop_embed.data.vcf import SNPRecord


WINDOW_MODES = ("disjoint", "overlap")


@dataclass(frozen=True)
class Window:
    chrom: int
    start: int   # genomic start, 0-based inclusive (may be negative near chr start)
    end: int     # genomic end,   0-based exclusive
    index: int   # position in SNPWindowPartitioner.windows


class SNPWindowPartitioner:
    """
    Parameters
    ----------
    snps_by_chrom : {chrom_int: [SNPRecord]}, each list sorted by pos.
                    Typically produced by load_snps_from_vcf.
    half_window   : half the window size in bases.  The window spans
                    2 * half_window bases total.
    buffer        : minimum distance (in bases) from a SNP to either window
                    edge.  Must be < half_window.

    Attributes
    ----------
    windows           : list[Window] — all windows, ordered chrom then start.
    snp_to_window_idx : dict[(chrom, pos), int] — maps each SNP's (chrom, pos)
                        to the index of its assigned window in self.windows.
    window_snp_indices: dict[int, list[int]] — maps window index to list of
                        SNP indices (into snps_by_chrom[chrom]) assigned to it.
    """

    def __init__(
        self,
        snps_by_chrom: dict[int, list[SNPRecord]],
        half_window: int,
        buffer: int = 0,
        *,
        mode: str = "disjoint",
    ) -> None:
        if mode not in WINDOW_MODES:
            raise ValueError(f"unknown window mode {mode!r}; use one of {WINDOW_MODES}")
        if buffer >= half_window:
            raise ValueError(
                f"buffer ({buffer}) must be less than half_window ({half_window})"
            )
        if mode == "disjoint" and buffer != 0:
            raise ValueError(
                f"mode='disjoint' requires buffer=0 (got {buffer}): windows are "
                f"contiguous, so a right-edge buffer would leave boundary SNPs "
                f"unassigned. Use mode='overlap' if you need a buffer."
            )

        self.half_window = half_window
        self.buffer = buffer
        self.mode = mode
        self.snps_by_chrom = snps_by_chrom

        self.windows: list[Window] = []
        self.snp_to_window_idx: dict[tuple[int, int], int] = {}
        self.window_snp_indices: dict[int, list[int]] = {}

        self._build_disjoint() if mode == "disjoint" else self._build_overlap()

    def _build_disjoint(self) -> None:
        """Contiguous, non-overlapping spans — see the module docstring."""
        hw = self.half_window
        for chrom in sorted(self.snps_by_chrom):
            snps = self.snps_by_chrom[chrom]   # already sorted by pos
            n = len(snps)
            i = 0
            prev_end: int | None = None        # per-chromosome

            while i < n:
                p = snps[i].pos
                w_start = p - hw
                if prev_end is not None and w_start < prev_end:
                    w_start = prev_end          # abut previous window; never overlap
                w_end = w_start + 2 * hw

                win_idx = len(self.windows)
                self.windows.append(Window(chrom, w_start, w_end, win_idx))
                self.window_snp_indices[win_idx] = []

                # Claim the whole span [w_start, w_end) — matches the bisect range
                # build_sample_fp_index uses, so span membership == assignment.
                # The opening SNP is always inside: even when clamped,
                # w_end = prev_end + 2*hw > (p - hw) + 2*hw > p.
                while i < n and snps[i].pos < w_end:
                    self.snp_to_window_idx[(chrom, snps[i].pos)] = win_idx
                    self.window_snp_indices[win_idx].append(i)
                    i += 1

                prev_end = w_end

    def _build_overlap(self) -> None:
        """Legacy: spans [p-hw, p+hw] that may overlap — see the module docstring."""
        for chrom in sorted(self.snps_by_chrom):
            snps = self.snps_by_chrom[chrom]   # already sorted by pos
            n    = len(snps)
            lo   = 0
            
            while lo < n:
                p        = snps[lo].pos
                w_start  = p - self.half_window
                w_end    = p + self.half_window
                cutoff   = w_end - self.buffer  # last pos that fits within buffer

                win_idx = len(self.windows)
                self.windows.append(Window(chrom, w_start, w_end, win_idx))
                self.window_snp_indices[win_idx] = []

                while lo < n and snps[lo].pos <= cutoff:
                    self.snp_to_window_idx[(chrom, snps[lo].pos)] = win_idx
                    self.window_snp_indices[win_idx].append(lo)
                    lo += 1

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self) -> Iterator[Window]:
        return iter(self.windows)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def snps_per_window_stats(self) -> dict:
        counts = [len(v) for v in self.window_snp_indices.values()]
        if not counts:
            return {}
        import statistics
        return {
            "n_windows":  len(counts),
            "total_snps": sum(counts),
            "mean":       round(statistics.mean(counts), 2),
            "median":     statistics.median(counts),
            "max":        max(counts),
            "min":        min(counts),
        }


def make_partitioner(snps_by_chrom, half_window, buffer=0, *, mode="disjoint"):
    """Kept as the name the ``--window-mode`` CLIs call; the mode now lives on
    :class:`SNPWindowPartitioner` itself."""
    return SNPWindowPartitioner(snps_by_chrom, half_window, buffer, mode=mode)

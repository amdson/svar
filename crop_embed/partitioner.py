"""
crop_embed/partitioner.py
-------------------------
SNPWindowPartitioner: assigns each SNP to exactly one genomic window such that
every SNP sits at least `buffer` bases from both window edges.

Algorithm (greedy, left-to-right per chromosome)
-------------------------------------------------
Sort SNPs by position.  While there are unassigned SNPs:
  1. Take the leftmost unassigned SNP at position p.
  2. Open a window  [p - half_window, p + half_window]
  3. Assign all SNPs whose position falls in [p - half_window + buffer, p + half_window - buffer]
     (The first SNP is `buffer` from the left edge; the last is at least
      `buffer` from the right edge.)
  4. Advance to the next unassigned SNP.

This clusters nearby SNPs into a shared window while guaranteeing the buffer
invariant, and never places any SNP at an edge where context is truncated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from crop_embed.data.vcf import SNPRecord


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
        buffer: int,
    ) -> None:
        if buffer >= half_window:
            raise ValueError(
                f"buffer ({buffer}) must be less than half_window ({half_window})"
            )

        self.half_window = half_window
        self.buffer = buffer
        self.snps_by_chrom = snps_by_chrom

        self.windows: list[Window] = []
        self.snp_to_window_idx: dict[tuple[int, int], int] = {}
        self.window_snp_indices: dict[int, list[int]] = {}

        self._build()

    def _build(self) -> None:
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


class NonOverlappingSNPWindowPartitioner(SNPWindowPartitioner):
    """A non-overlapping variant of :class:`SNPWindowPartitioner`: window spans
    ``[start, end)`` are **disjoint**, so every SNP falls in exactly one window and
    is embedded in exactly one reconstructed sequence / fingerprint.

    Why
    ---
    The base partitioner assigns each SNP to one window for *tiling*, but a window's
    embedded content is every SNP inside its ``[p-half_window, p+half_window)`` span,
    and consecutive spans overlap by up to ``half_window``. So a SNP near a window
    edge is embedded in **two** windows (empirically ~1.8 windows/SNP on dense data).
    When a reference method defines strictly disjoint windows, that double-embedding
    is a mismatch. This variant removes it.

    Algorithm (greedy, left-to-right per chromosome)
    ------------------------------------------------
    Same greedy opening as the base class — open a window at the leftmost unassigned
    SNP ``p`` — but the window's left edge is **clamped to the previous window's right
    edge** so spans never overlap::

        w_start = max(p - half_window, prev_window_end)
        w_end   = w_start + 2*half_window          # fixed 2*half_window width
        (claim every SNP in [w_start, w_end); the next window opens at the first SNP >= w_end)

    * When SNPs are **>= 2*half_window apart**, ``p - half_window >= prev_end`` so no
      clamp happens and the window is exactly ``[p-half_window, p+half_window)`` —
      **byte-identical windows to the base partitioner** (verified).
    * When a following SNP falls **within** a placed window, the next window is
      **shifted right** to abut it. Shifted windows are no longer centered on their
      opening SNP (the price of disjointness), and a dense region becomes a
      contiguous tiling of ``2*half_window``-wide bins.

    ``buffer`` must be 0: a right-edge buffer would leave boundary SNPs unclaimed
    once windows are forced contiguous. All other attributes / the iteration and
    lookup interface are inherited, so this is a drop-in for
    ``build_sample_fp_index`` / ``UniqueWindowDataset``.
    """

    def _build(self) -> None:
        if self.buffer != 0:
            raise ValueError(
                "NonOverlappingSNPWindowPartitioner requires buffer=0: windows are "
                "contiguous, so a right-edge buffer would leave boundary SNPs "
                "unassigned. Got buffer=%r." % self.buffer)
        hw = self.half_window
        for chrom in sorted(self.snps_by_chrom):
            snps = self.snps_by_chrom[chrom]   # already sorted by pos
            n = len(snps)
            i = 0
            prev_end: int | None = None

            while i < n:
                p = snps[i].pos
                w_start = p - hw
                if prev_end is not None and w_start < prev_end:
                    w_start = prev_end          # abut previous window; never overlap
                w_end = w_start + 2 * hw

                win_idx = len(self.windows)
                self.windows.append(Window(chrom, w_start, w_end, win_idx))
                self.window_snp_indices[win_idx] = []

                # Claim the whole span [w_start, w_end) (matches the bisect range
                # build_sample_fp_index uses, so span membership == assignment).
                while i < n and snps[i].pos < w_end:
                    self.snp_to_window_idx[(chrom, snps[i].pos)] = win_idx
                    self.window_snp_indices[win_idx].append(i)
                    i += 1

                prev_end = w_end

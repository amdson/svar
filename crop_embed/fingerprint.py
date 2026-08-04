"""
crop_embed/fingerprint.py
-------------------------
A fingerprint uniquely identifies the DNA sequence of a window for a given
sample.  Because the reference is invariant, the sequence is fully determined
by the window's genomic coordinates plus which SNPs within that range carry
the alt allele for the sample in question.

Fingerprint type
----------------
    tuple[int, int, int, tuple[int, ...]]
    (chrom, window_start, window_end, alt_positions)

    alt_positions : sorted tuple of 0-based genomic positions of SNPs within
                    [window_start, window_end) that carry the alt allele for
                    this sample.  Empty tuple = pure reference in this window.

This is hashable and can be used directly as a dict key or set member.
"""

from __future__ import annotations

import bisect

import numpy as np
import torch

from crop_embed.data.vcf import SNPRecord
from crop_embed.partitioner import SNPWindowPartitioner, Window

# Type alias for readability
Fingerprint = tuple[int, int, int, tuple[int, ...]]


def make_fingerprint(
    window: Window,
    snps_in_range: list[SNPRecord],
    sample_idx: int,
) -> Fingerprint:
    """
    Compute the fingerprint for one (window, sample) pair.

    Parameters
    ----------
    window        : the Window object (chrom, start, end, index)
    snps_in_range : all SNPRecords whose pos falls in [window.start, window.end).
                    These are the SNPs that can affect the window's sequence.
    sample_idx    : index into each SNPRecord's gt_alts bytes

    Returns
    -------
    Fingerprint tuple
    """
    alt_positions = tuple(
        rec.pos
        for rec in snps_in_range
        if rec.gt_alts[sample_idx]
    )
    return (window.chrom, window.start, window.end, alt_positions)


def build_sample_window_map(
    partitioner: SNPWindowPartitioner,
    n_samples: int,
) -> dict[tuple[int, int, int], Fingerprint]:
    """
    Precompute fingerprints for every (sample_idx, window_idx) pair.

    Returns
    -------
    sample_window_to_fp : {(sample_idx, window_idx): Fingerprint}

    Also returns the set of unique fingerprints as a by-product so callers
    can build the deduplicated index without a second pass.

    Returns
    -------
    sample_window_to_fp : dict[(sample_idx, window_idx) → Fingerprint]
    unique_fingerprints : list[Fingerprint] — stable order, deduplicated
    fp_to_idx           : dict[Fingerprint → int] — index into unique_fingerprints
    """
    sample_window_to_fp: dict[tuple[int, int, int], Fingerprint] = {}
    fp_to_idx: dict[Fingerprint, int] = {}
    unique_fingerprints: list[Fingerprint] = []

    for window in partitioner:
        chrom = window.chrom
        snps  = partitioner.snps_by_chrom[chrom]

        # Binary-search to find all SNPs in [window.start, window.end)
        positions = [r.pos for r in snps]
        lo = bisect.bisect_left(positions,  window.start)
        hi = bisect.bisect_left(positions,  window.end)
        snps_in_range = snps[lo:hi]

        for sample_idx in range(n_samples):
            fp  = make_fingerprint(window, snps_in_range, sample_idx)
            key = (sample_idx, window.index)
            sample_window_to_fp[key] = fp

            if fp not in fp_to_idx:
                fp_to_idx[fp] = len(unique_fingerprints)
                unique_fingerprints.append(fp)

    return sample_window_to_fp, unique_fingerprints, fp_to_idx


def build_sample_fp_index(
    partitioner: SNPWindowPartitioner,
    n_samples: int,
) -> tuple[list[Fingerprint], "torch.Tensor"]:
    """
    Memory-frugal replacement for build_sample_window_map: produce the two
    artefacts the dataset actually needs — the deduplicated ``unique_fingerprints``
    list and the ``(n_samples, n_windows)`` index tensor — WITHOUT ever
    materialising the per-(sample × window) fingerprint dict (hundreds of
    millions of entries on soy → ~130 GB / >1 h). Peak extra memory here is one
    ``(n_samples, k)`` uint8 matrix per window (k = SNPs in the window, usually 1).

    Per window it stacks the per-SNP alt-flag byte vectors (``gt_alts``, already
    one byte per sample) into an ``(n_samples, k)`` matrix and dedups with a
    single ``np.unique(axis=0)``. Because a fingerprint embeds ``(chrom, start,
    end)``, windows never share fingerprints, so the global list is simply each
    window's unique rows concatenated in window order. Unique rows are emitted in
    first-appearance-by-sample order, so the output is byte-identical to what
    ``build_sample_window_map`` + the old index construction produced.

    Returns
    -------
    unique_fingerprints : list[Fingerprint]                — cache-row order
    sample_fp_index     : LongTensor[n_samples, n_windows] — indexes the list
    """
    sample_fp_index = torch.empty((n_samples, len(partitioner)), dtype=torch.long)
    unique_fingerprints: list[Fingerprint] = []
    positions_by_chrom: dict[int, list[int]] = {}

    for window in partitioner:
        chrom = window.chrom
        snps  = partitioner.snps_by_chrom[chrom]

        positions = positions_by_chrom.get(chrom)
        if positions is None:
            positions = [r.pos for r in snps]
            positions_by_chrom[chrom] = positions

        # Same binary-search slice build_sample_window_map uses.
        lo = bisect.bisect_left(positions, window.start)
        hi = bisect.bisect_left(positions, window.end)
        snps_in_range = snps[lo:hi]

        offset = len(unique_fingerprints)

        if not snps_in_range:
            # No SNPs in range → every sample is pure reference: one fingerprint.
            unique_fingerprints.append((chrom, window.start, window.end, ()))
            sample_fp_index[:, window.index] = offset
            continue

        # (n_samples, k) alt-flag matrix, one column per SNP in range.
        alt = np.empty((n_samples, len(snps_in_range)), dtype=np.uint8)
        for j, rec in enumerate(snps_in_range):
            alt[:, j] = np.frombuffer(rec.gt_alts, dtype=np.uint8)

        uniq, first_idx, inv = np.unique(
            alt, axis=0, return_index=True, return_inverse=True
        )
        inv = np.asarray(inv).reshape(-1)   # np>=2 may return shape (n, 1)

        # Re-order the (lexicographically sorted) unique rows into first-
        # appearance-by-sample order so emitted indices match the old builder.
        order = np.argsort(first_idx, kind="stable")
        rank  = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])

        snp_pos = [rec.pos for rec in snps_in_range]
        for u in order:
            row = uniq[u]
            alt_positions = tuple(
                snp_pos[c] for c in range(len(snp_pos)) if row[c]
            )
            unique_fingerprints.append((chrom, window.start, window.end, alt_positions))

        sample_fp_index[:, window.index] = torch.from_numpy(
            (offset + rank[inv]).astype(np.int64)
        )

    return unique_fingerprints, sample_fp_index

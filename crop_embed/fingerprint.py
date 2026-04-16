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

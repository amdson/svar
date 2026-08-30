"""
crop_embed/data/vcf.py
----------------------
Load and parse VCF files into the internal SNP representation used by the
rest of the pipeline.

Internal format
---------------
snps_by_chrom : dict[int, list[SNPRecord]]
    Keyed by integer chromosome number (1-based).
    Each list is sorted by genomic position and contains namedtuples:
        SNPRecord(pos, ref_byte, alt_byte, gt_alts)
    where gt_alts is a bytes object of length n_samples:
        gt_alts[i] == 1  →  sample i carries the alt allele
        gt_alts[i] == 0  →  sample i carries the ref allele (or missing)
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import pysam


class SNPRecord(NamedTuple):
    pos: int        # 0-based genomic position
    ref_byte: int   # ord of reference nucleotide
    alt_byte: int   # ord of alt nucleotide
    gt_alts: bytes  # per-sample alt flags, length == n_samples


def load_snps_from_vcf(
    vcf_path: str,
    samples: list[str] | None = None,
    chroms: set[int] | None = None,
) -> tuple[dict[int, list[SNPRecord]], list[str]]:
    """
    Single-pass VCF read.

    Parameters
    ----------
    vcf_path : path to a biallelic VCF (plain or bgzipped)
    samples  : sample IDs to include; None means all samples in the VCF header
    chroms   : if given, keep only records on these (parsed-int) chromosomes;
               non-matching records are skipped before the per-sample genotype
               is decoded, so a single-chromosome subset reads much faster.

    Returns
    -------
    snps_by_chrom : {chrom_int: [SNPRecord]}, each list sorted by pos
    samples       : ordered list of sample IDs used (matches gt_alts index)
    """
    snps_by_chrom: dict[int, list[SNPRecord]] = defaultdict(list)

    with pysam.VariantFile(vcf_path) as vcf:
        all_samples = list(vcf.header.samples)
        if samples is None:
            samples = all_samples

        # Pre-compute indices so we avoid repeated list.index() calls
        sample_indices = [all_samples.index(s) for s in samples]

        for rec in vcf.fetch():
            if not rec.alts:
                continue

            chrom_int = _parse_chrom(rec.chrom)
            if chroms is not None and chrom_int not in chroms:
                continue
            ref_byte  = ord(rec.ref[0].upper())
            alt_byte  = ord(rec.alts[0][0].upper())

            # Compact per-sample genotype: 1 = alt, 0 = ref/missing
            gt_alts = bytes(
                1 if (_first_allele(rec.samples[all_samples[si]]["GT"]) == 1) else 0
                for si in sample_indices
            )

            snps_by_chrom[chrom_int].append(
                SNPRecord(rec.start, ref_byte, alt_byte, gt_alts)
            )

    # Sort each chromosome's SNPs by position (required for sliding-window logic)
    for lst in snps_by_chrom.values():
        lst.sort(key=lambda r: r.pos)

    return dict(snps_by_chrom), samples


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_chrom(chrom: str) -> int:
    if chrom.isdigit():
        return int(chrom)
    return int(chrom.lstrip("chr").lstrip("Chr"))


def _first_allele(gt) -> int:
    """Return the first allele index from a pysam GT tuple, or 0 if missing."""
    if gt is None or len(gt) == 0 or gt[0] is None:
        return 0
    return gt[0]

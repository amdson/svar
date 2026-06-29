"""
crop_embed/data/pgen.py
-----------------------
Load a plink2 .pgen/.pvar/.psam fileset into the SAME internal SNP representation
as crop_embed.data.vcf.load_snps_from_vcf — but far faster than parsing a large
text VCF, since genotypes come straight from the binary .pgen via pgenlib.
(On the 20k-accession soy SoySNP50K set, the VCF parse is ~11 min; this is seconds.)

Equivalence
-----------
For a fileset produced by `plink2 … --make-pgen --export vcf` (REF aligned to the
genome), this yields SNPRecords byte-identical to load_snps_from_vcf() on the
exported VCF. Why: that VCF writes heterozygotes ref-first ("0/1"), so the VCF
loader's first-allele rule (gt_alts == 1 iff the first GT allele is alt) marks
exactly the homozygous-ALT samples — which is the pgen hardcall == 2. Hets (1),
hom-ref (0) and missing (-9) all map to 0. (scripts/test_pgen_vs_vcf.py checks this.)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from crop_embed.data.vcf import SNPRecord, _parse_chrom


def _read_psam(psam_path: str) -> list[str]:
    """Sample IDs from a .psam, in file order (the IID column)."""
    with open(psam_path) as f:
        header = f.readline().lstrip("#").split()
        iid = header.index("IID")
        return [ln.split()[iid] for ln in f if ln.strip()]


def _read_pvar(pvar_path: str) -> list[tuple[int, int, int, int]]:
    """Per-variant (chrom_int, pos0, ref_byte, alt_byte), in file (=.pgen) order."""
    out: list[tuple[int, int, int, int]] = []
    with open(pvar_path) as f:
        for ln in f:
            if ln.startswith("#"):  # ## meta lines and the #CHROM column header
                continue
            parts = ln.split()
            chrom, pos, ref, alt = parts[0], parts[1], parts[3], parts[4]
            out.append((
                _parse_chrom(chrom),
                int(pos) - 1,                       # .pvar is 1-based; SNPRecord.pos is 0-based
                ord(ref[0].upper()),
                ord(alt.split(",")[0][0].upper()),  # first ALT (filesets here are biallelic)
            ))
    return out


def load_snps_from_pgen(
    prefix: str,
    samples: list[str] | None = None,
) -> tuple[dict[int, list[SNPRecord]], list[str]]:
    """
    Parameters
    ----------
    prefix  : path to the fileset, with or without a trailing ``.pgen``; the
              companion ``.pvar`` and ``.psam`` must sit alongside it.
    samples : sample IDs to include, in this order; None = all, in .psam order.

    Returns
    -------
    snps_by_chrom : {chrom_int: [SNPRecord]}, each list sorted by pos
    samples       : ordered list of sample IDs used (matches gt_alts index)
    """
    try:
        import pgenlib
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "load_snps_from_pgen needs the 'Pgenlib' package (pip install Pgenlib)."
        ) from e

    prefix = str(prefix)
    if prefix.endswith(".pgen"):
        prefix = prefix[:-5]

    all_samples = _read_psam(prefix + ".psam")
    if samples is None:
        samples = all_samples
    # Indices of the requested samples within the .pgen, in requested order, so
    # gt_alts is ordered to match `samples` (same contract as load_snps_from_vcf).
    take = np.fromiter((all_samples.index(s) for s in samples),
                       dtype=np.intp, count=len(samples))

    variants = _read_pvar(prefix + ".pvar")

    reader = pgenlib.PgenReader(str.encode(prefix + ".pgen"))
    n_all = reader.get_raw_sample_ct()
    if reader.get_variant_ct() != len(variants):
        reader.close()
        raise ValueError(
            f".pgen has {reader.get_variant_ct()} variants but .pvar has {len(variants)}"
        )

    snps_by_chrom: dict[int, list[SNPRecord]] = defaultdict(list)
    buf = np.empty(n_all, dtype=np.int8)  # ALT hardcalls: 0/1/2, -9 = missing
    for v_idx, (chrom, pos0, ref_b, alt_b) in enumerate(variants):
        reader.read(v_idx, buf)
        gt_alts = (buf[take] == 2).astype(np.uint8).tobytes()
        snps_by_chrom[chrom].append(SNPRecord(pos0, ref_b, alt_b, gt_alts))
    reader.close()

    # Sort each chromosome's SNPs by position (load_snps_from_vcf does the same).
    for lst in snps_by_chrom.values():
        lst.sort(key=lambda r: r.pos)

    return dict(snps_by_chrom), list(samples)

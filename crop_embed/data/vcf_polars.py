"""
crop_embed/data/vcf_polars.py
-----------------------------
A fast, vectorized VCF reader — a drop-in alternative to the pysam path in
``crop_embed.data.vcf.load_snps_from_vcf`` for plain (uncompressed) biallelic
GT VCFs. polars parses the whole file in Rust in one streaming pass instead of
pysam's per-record Python loop, which is ~2 orders of magnitude faster on large
files (arabidopsis 1001g: ~18 s here vs ~45 min projected for pysam).

Assumptions (hold for the soy/rice/arabidopsis/wheat filesets in this repo):
  * plain-text (not bgzipped) VCF;
  * FORMAT begins with GT (only the GT subfield is read);
  * the alt flag for a sample is "first GT allele index == 1" — matching
    ``crop_embed.data.vcf._first_allele(...) == 1``, so multiallelic alleles
    (index >= 2) and missing (".") both read as ref/0, exactly as pysam does.

Validated byte-for-byte identical to load_snps_from_vcf on the rice fileset
(27,882 SNPs x 383 samples): identical sample order, (chrom, pos) keys, and
per-sample alt flags. See scratchpad/validate_polars_vs_pysam.py.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

from crop_embed.data.vcf import SNPRecord

# Standard VCF fixed columns before the per-sample genotype columns.
_N_FIXED = 9


def load_alt_matrix(
    vcf_path: str,
    samples: list[str] | None = None,
    chroms: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Vectorized read of a plain biallelic GT VCF.

    Parameters
    ----------
    vcf_path : path to a plain-text VCF.
    samples  : sample IDs (columns) to keep, in this order; None = all, in header
               order.
    chroms   : if given, keep only these (parsed-int) chromosomes. The filter sits
               *before* the per-sample select in the lazy plan, so the ~1k regex
               extractions only run on surviving rows — on arabidopsis chr4 that is
               16% of the file's records and roughly a 4x saving in both time and
               peak memory.

    Returns
    -------
    chrom : int32[n_snps]    — integer chromosome (``chr``/``Chr`` prefix stripped)
    pos   : int64[n_snps]    — **0-based** position (VCF POS - 1, matching pysam rec.start)
    ref_byte : uint8[n_snps] — ord(REF[0].upper())
    alt_byte : uint8[n_snps] — ord(ALT[0].upper())
    alt   : uint8[n_snps, n_samples] — 1 iff sample carries the alt allele
    samples : list[str]      — column order of ``alt``

    Records with no alt (ALT ".") are dropped, matching load_snps_from_vcf.
    """
    lf = pl.scan_csv(vcf_path, separator="\t", comment_prefix="##",
                     has_header=True, infer_schema_length=0)   # all columns Utf8
    cols = lf.collect_schema().names()
    chrom_col = cols[0]                                        # "#CHROM"
    all_samples = cols[_N_FIXED:]
    if samples is None:
        samples = list(all_samples)
    else:
        known = set(all_samples)
        missing = [s for s in samples if s not in known]
        if missing:
            raise KeyError(f"{len(missing)} requested samples absent from VCF header "
                           f"(e.g. {missing[:3]})")

    # First GT allele index == 1 → alt. Regex pulls the leading allele index
    # (handles multi-digit and both '/' and '|' separators); missing "." → null → 0.
    def alt_flag(col: str) -> pl.Expr:
        return (pl.col(col).str.extract(r"^([0-9]+)") == "1").fill_null(False).cast(pl.UInt8).alias(col)

    chrom_int = (pl.col(chrom_col).str.replace(r"^(chr|Chr)", "")
                 .cast(pl.Int32, strict=False))
    keep = pl.col("ALT") != "."                               # skip no-alt records
    if chroms is not None:
        keep = keep & chrom_int.is_in(sorted(int(c) for c in chroms))

    df = (
        lf.filter(keep)
        .select(
            chrom_int.alias("__chrom"),
            pl.col("POS").cast(pl.Int64).alias("__pos"),
            pl.col("REF").alias("__ref"),
            pl.col("ALT").alias("__alt"),
            *[alt_flag(s) for s in samples],
        )
        .collect(engine="streaming")
    )

    chrom = df["__chrom"].to_numpy()
    pos = df["__pos"].to_numpy() - 1                           # 1-based VCF POS → 0-based
    ref_byte = np.frombuffer(
        "".join(s[0] for s in df["__ref"].to_list()).upper().encode("ascii"), dtype=np.uint8)
    alt_byte = np.frombuffer(
        "".join(s[0] for s in df["__alt"].to_list()).upper().encode("ascii"), dtype=np.uint8)
    alt = df.select(samples).to_numpy()                       # (n_snps, n_samples) uint8
    return chrom, pos, ref_byte, alt_byte, alt, list(samples)


def load_snps_from_vcf(
    vcf_path: str,
    samples: list[str] | None = None,
    chroms: set[int] | None = None,
) -> tuple[dict[int, list[SNPRecord]], list[str]]:
    """Drop-in polars replacement for ``crop_embed.data.vcf.load_snps_from_vcf``.

    Returns ``(snps_by_chrom, samples)`` in the identical internal format
    (per-chromosome lists of SNPRecord, sorted by 0-based pos), built via the
    vectorized :func:`load_alt_matrix`.
    """
    chrom, pos, ref_byte, alt_byte, alt, samples = load_alt_matrix(
        vcf_path, samples, chroms)
    gt_rows = [row.tobytes() for row in alt]                  # per-SNP length-n_samples bytes

    snps_by_chrom: dict[int, list[SNPRecord]] = defaultdict(list)
    for i in range(len(pos)):
        snps_by_chrom[int(chrom[i])].append(
            SNPRecord(int(pos[i]), int(ref_byte[i]), int(alt_byte[i]), gt_rows[i]))
    for lst in snps_by_chrom.values():
        lst.sort(key=lambda r: r.pos)                         # VCFs are pos-sorted; ensure it
    return dict(snps_by_chrom), samples

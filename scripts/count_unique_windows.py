"""
scripts/count_unique_windows.py
-------------------------------
Count the number of UNIQUE WINDOWS implied by a VCF + window size.

A "unique window" is a distinct Fingerprint — (chrom, start, end, alt_positions)
— across all (sample, window) pairs (crop_embed/fingerprint.py). Two pairs share
a window iff they'd produce the same DNA sequence. This count equals
`len(UniqueWindowDataset)`, but is computed straight from the VCF: no FASTA and no
(n_samples × n_windows) index, so it scales to large panels (e.g. the 20k-accession
soy SoySNP50K set, where building the full dataset would need ~6 GB).

    python scripts/count_unique_windows.py --vcf-path <vcf> --half-window 500
    python scripts/count_unique_windows.py --vcf-path <vcf> --half-window 500 --samples 200
    python scripts/count_unique_windows.py --vcf-path <vcf> --half-window 500 --method exact

`fast` (default) and `exact` give identical counts; `exact` is the literal
make_fingerprint()/set reference (O(n_windows × n_samples)); `fast` counts the
same thing per window with numpy. --validate runs both and asserts they agree.
"""
from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import DEFAULT_VCF_PATH
from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.data.pgen import load_snps_from_pgen, _read_psam
from crop_embed.partitioner import SNPWindowPartitioner
from crop_embed.fingerprint import make_fingerprint


def _snps_in_window(snps, positions, window):
    """The SNPRecords whose pos falls in [window.start, window.end) — same
    binary-search slice build_sample_window_map uses."""
    lo = bisect.bisect_left(positions, window.start)
    hi = bisect.bisect_left(positions, window.end)
    return snps[lo:hi]


def count_exact(snps_by_chrom, partitioner, n_samples: int) -> int:
    """Reference count: distinct make_fingerprint() over every (sample, window).

    This is exactly what build_sample_window_map() dedups into
    `unique_fingerprints`, so it equals len(UniqueWindowDataset). O(W × S).
    """
    pos_cache = {c: [r.pos for r in s] for c, s in snps_by_chrom.items()}
    unique = set()
    for window in partitioner:
        snps_in_range = _snps_in_window(
            snps_by_chrom[window.chrom], pos_cache[window.chrom], window
        )
        for sample_idx in range(n_samples):
            unique.add(make_fingerprint(window, snps_in_range, sample_idx))
    return len(unique)


def count_fast(snps_by_chrom, partitioner, n_samples: int) -> int:
    """Same count, computed per window with numpy.

    Within one window chrom/start/end are fixed, so distinct fingerprints ==
    distinct alt-position subsets == distinct columns of the in-range SNPs'
    gt_alts matrix. Fingerprints from different windows never collide, so the
    total is the sum over windows of that per-window distinct-column count.
    """
    total = 0
    pos_cache = {c: [r.pos for r in s] for c, s in snps_by_chrom.items()}
    for window in partitioner:
        snps_in_range = _snps_in_window(
            snps_by_chrom[window.chrom], pos_cache[window.chrom], window
        )
        k = len(snps_in_range)
        if k == 0:
            total += 1  # only the all-reference fingerprint, shared by every sample
            continue
        # (k, n_samples) uint8 alt-flag matrix; count distinct sample columns.
        mat = np.frombuffer(
            b"".join(r.gt_alts for r in snps_in_range), dtype=np.uint8
        ).reshape(k, n_samples)
        if k <= 64:
            # gt_alts is binary, so each sample's k flags pack into one uint64
            # key — distinct keys == distinct columns, via a fast 1-D unique.
            weights = (np.uint64(1) << np.arange(k, dtype=np.uint64))[:, None]
            keys = (mat.astype(np.uint64) * weights).sum(axis=0)
            total += np.unique(keys).size
        else:  # >64 SNPs in one window (rare): fall back to the 2-D unique
            total += np.unique(mat, axis=1).shape[1]
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcf-path", default=DEFAULT_VCF_PATH)
    p.add_argument("--pgen", default=None,
                   help="plink2 fileset prefix (.pgen/.pvar/.psam). If given, "
                        "genotypes load from the binary .pgen via pgenlib — far "
                        "faster than the text VCF — and --vcf-path is ignored.")
    p.add_argument("--half-window", type=int, default=500,
                   help="Half the window size in bp; window spans 2*half_window.")
    p.add_argument("--buffer", type=int, default=0)
    p.add_argument("--samples", type=int, default=None,
                   help="Use only the first N samples (default: all).")
    p.add_argument("--method", choices=["fast", "exact"], default="fast")
    p.add_argument("--validate", action="store_true",
                   help="Run both methods and assert the counts match.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.pgen:
        prefix = args.pgen[:-5] if args.pgen.endswith(".pgen") else args.pgen
        keep = _read_psam(prefix + ".psam")[: args.samples] if args.samples else None
        print(f"Loading SNPs from {prefix}.pgen …")
        snps_by_chrom, samples = load_snps_from_pgen(prefix, keep)
    else:
        # Resolve the sample subset from the header first (cheap), so the VCF is
        # parsed only once — important for big multi-GB VCFs.
        keep = None
        if args.samples is not None:
            import pysam
            with pysam.VariantFile(args.vcf_path) as vf:
                keep = list(vf.header.samples)[: args.samples]
        print(f"Loading SNPs from {args.vcf_path} …")
        snps_by_chrom, samples = load_snps_from_vcf(args.vcf_path, keep)
    n_samples = len(samples)
    n_snps = sum(len(v) for v in snps_by_chrom.values())
    print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {n_samples:,} samples")

    part = SNPWindowPartitioner(snps_by_chrom, half_window=args.half_window, buffer=args.buffer)
    n_windows = len(part)
    print(f"  {n_windows:,} windows (half_window={args.half_window}, buffer={args.buffer})")

    if args.validate:
        f = count_fast(snps_by_chrom, part, n_samples)
        e = count_exact(snps_by_chrom, part, n_samples)
        assert f == e, f"mismatch: fast={f} exact={e}"
        n_unique = f
        print(f"  validated: fast == exact == {n_unique:,}")
    else:
        counter = count_fast if args.method == "fast" else count_exact
        n_unique = counter(snps_by_chrom, part, n_samples)

    print(f"\nunique windows: {n_unique:,}")
    print(f"  (n_windows={n_windows:,}; mean distinct haplotypes/window "
          f"= {n_unique / n_windows:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

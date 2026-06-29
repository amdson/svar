"""
scripts/test_pgen_vs_vcf.py
---------------------------
Verify that loading genotypes from a plink2 .pgen (load_snps_from_pgen) is
equivalent to parsing the exported text VCF (load_snps_from_vcf), and that a
UniqueWindowDataset built either way has the same unique windows.

Checks, on a sample subset (use --all-samples for the full panel):
  1. snps_by_chrom byte-identical: same chroms, and every SNPRecord
     (pos, ref_byte, alt_byte, gt_alts) equal.
  2. Partitioner windows identical.
  3. The VCF-backed UniqueWindowDataset's unique_fingerprints == the pgen-loaded
     fingerprints (same set and count) — i.e. the two datasets are the same.
Also prints the load-time speedup.

    python scripts/test_pgen_vs_vcf.py \
        --vcf-path  /…/soysnp50k_a2_final.vcf \
        --pgen      /…/soysnp50k_a2_final \
        --fasta-path /…/Glycine_max.Glycine_max_v2.1.dna_sm.toplevel.fa \
        --half-window 500 --samples 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.data.pgen import load_snps_from_pgen, _read_psam
from crop_embed.partitioner import SNPWindowPartitioner
from crop_embed.fingerprint import build_sample_window_map
from crop_embed.dataset import UniqueWindowDataset


def assert_snps_equal(a: dict, b: dict) -> None:
    assert set(a) == set(b), f"chrom sets differ: {sorted(a)} vs {sorted(b)}"
    for chrom in a:
        la, lb = a[chrom], b[chrom]
        assert len(la) == len(lb), f"chrom {chrom}: {len(la)} vs {len(lb)} SNPs"
        for i, (ra, rb) in enumerate(zip(la, lb)):
            if ra != rb:
                raise AssertionError(
                    f"chrom {chrom} SNP {i} differs:\n  vcf ={ra}\n  pgen={rb}"
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcf-path", required=True)
    p.add_argument("--pgen", required=True, help="plink2 fileset prefix")
    p.add_argument("--fasta-path", required=True)
    p.add_argument("--half-window", type=int, default=500)
    p.add_argument("--buffer", type=int, default=0)
    p.add_argument("--samples", type=int, default=300,
                   help="Use the first N samples (default 300; keeps the VCF parse cheap).")
    p.add_argument("--all-samples", action="store_true", help="Use every sample (slow VCF parse).")
    p.add_argument("--no-dataset", action="store_true",
                   help="Skip building the VCF-backed UniqueWindowDataset (skips a 2nd VCF parse).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prefix = args.pgen[:-5] if args.pgen.endswith(".pgen") else args.pgen
    keep = None if args.all_samples else _read_psam(prefix + ".psam")[: args.samples]
    n_req = "all" if keep is None else len(keep)
    print(f"samples: {n_req}\n")

    # 1. load both ways (timed)
    t = time.time()
    snps_pgen, samples = load_snps_from_pgen(prefix, keep)
    t_pgen = time.time() - t
    t = time.time()
    snps_vcf, samples_vcf = load_snps_from_vcf(args.vcf_path, keep)
    t_vcf = time.time() - t

    n_snps = sum(len(v) for v in snps_vcf.values())
    print(f"loaded {n_snps:,} SNPs × {len(samples)} samples")
    print(f"  pgen: {t_pgen:6.2f}s   vcf: {t_vcf:6.2f}s   speedup: {t_vcf / max(t_pgen, 1e-9):.1f}×\n")

    # check 1: identical SNP representation
    assert samples == samples_vcf, "sample order differs"
    assert_snps_equal(snps_vcf, snps_pgen)
    print("[1] snps_by_chrom byte-identical (pos, ref, alt, gt_alts)  ✓")

    # check 2: identical windows
    part_vcf = SNPWindowPartitioner(snps_vcf, half_window=args.half_window, buffer=args.buffer)
    part_pgen = SNPWindowPartitioner(snps_pgen, half_window=args.half_window, buffer=args.buffer)
    w_vcf = [(w.chrom, w.start, w.end) for w in part_vcf]
    w_pgen = [(w.chrom, w.start, w.end) for w in part_pgen]
    assert w_vcf == w_pgen, "partitioner windows differ"
    print(f"[2] partitioner identical: {len(part_vcf):,} windows  ✓")

    # check 3: the VCF-backed dataset's unique windows == pgen-derived ones
    _, uniq_pgen, _ = build_sample_window_map(part_pgen, len(samples))
    if args.no_dataset:
        _, uniq_vcf, _ = build_sample_window_map(part_vcf, len(samples))
        ref_desc = "vcf fingerprints"
    else:
        ds = UniqueWindowDataset(args.vcf_path, args.fasta_path, part_vcf, samples=keep)
        uniq_vcf = ds.unique_fingerprints
        ref_desc = "UniqueWindowDataset(vcf)"
    assert len(uniq_vcf) == len(uniq_pgen), f"count differs: {len(uniq_vcf)} vs {len(uniq_pgen)}"
    assert set(uniq_vcf) == set(uniq_pgen), "unique fingerprint sets differ"
    print(f"[3] {ref_desc} unique windows == pgen: {len(uniq_pgen):,}  ✓")

    print("\nRESULT: PASS — pgen and vcf loads are equivalent")
    return 0


if __name__ == "__main__":
    sys.exit(main())

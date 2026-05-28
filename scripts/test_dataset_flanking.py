"""
test_dataset_flanking.py
------------------------
Sanity-check the windows in UniqueWindowDataset against the MSU6 flanking-seq
table and the reference FASTA, mirroring notebooks/vcf_sanity_check_crossmap.

Two checks, both using each SNP's *reference* window pulled through the
dataset's own extract_sequence path (alt_positions=() → no alt substitution):

  1. REF allele check : the base at the SNP's offset in the reference window
                        equals the VCF REF allele.
  2. Flanking check   : the 16 bp on each side of the SNP equal the X5p_MSU6 /
                        X3p_MSU6 columns of FLANKING_PATH.

Usage:
    python scripts/test_dataset_flanking.py \\
        --vcf-path ../rice_data/sativas413_msu7_final.vcf \\
        --half-window 500 --buffer 0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed import SNPWindowPartitioner, UniqueWindowDataset
from crop_embed.data.coords import FASTA_PATH, FLANKING_PATH
from crop_embed.data.vcf import load_snps_from_vcf

DEFAULT_VCF_PATH = "../rice_data/sativas413_msu7_final.vcf"
FLANK = 16


def _ref_window_sequence(dataset: UniqueWindowDataset, chrom: int, start: int, end: int) -> str:
    """Reference sequence for a window — extract_sequence with no alts."""
    return dataset.extract_sequence((chrom, start, end, ()))


def check_dataset_flanking(
    dataset: UniqueWindowDataset,
    vcf_path: str,
    flanking_seq: pd.DataFrame,
    flank: int = FLANK,
) -> dict:
    """
    Iterate VCF records, assign each SNP to its window via the dataset's
    partitioner, and check REF allele + flanking sequence against the
    reference window materialised by dataset.extract_sequence.

    Returns a dict with per-check dataframes and summary counts.
    """
    partitioner = dataset.partitioner

    ref_match, ref_mismatch = [], []
    flank_results = []
    skipped_no_window = 0
    skipped_no_flanking = 0
    skipped_edge = 0

    with pysam.VariantFile(vcf_path) as vcf:
        for rec in vcf.fetch():
            if not rec.alts:
                continue

            chrom_str = rec.chrom
            chrom_int = int(chrom_str) if chrom_str.isdigit() else int(chrom_str.lstrip("chr"))
            pos0      = rec.start          # 0-based, matches SNPRecord.pos
            vcf_ref   = rec.ref.upper()

            win_idx = partitioner.snp_to_window_idx.get((chrom_int, pos0))
            if win_idx is None:
                skipped_no_window += 1
                continue

            window  = partitioner.windows[win_idx]
            win_seq = _ref_window_sequence(dataset, chrom_int, window.start, window.end)
            offset  = pos0 - window.start

            # ── REF allele check ────────────────────────────────────────────
            base = win_seq[offset : offset + len(vcf_ref)]
            entry = {
                "snp_id":     rec.id,
                "chr":        chrom_int,
                "pos":        rec.pos,
                "vcf_ref":    vcf_ref,
                "window_base": base,
            }
            (ref_match if base == vcf_ref else ref_mismatch).append(entry)

            # ── Flanking check ──────────────────────────────────────────────
            if rec.id not in flanking_seq.index:
                skipped_no_flanking += 1
                continue
            if offset < flank or offset + flank + 1 > len(win_seq):
                skipped_edge += 1
                continue

            stored = flanking_seq.loc[rec.id]
            left   = win_seq[offset - flank : offset]
            right  = win_seq[offset + 1 : offset + 1 + flank]
            stored_left  = stored["X5p_MSU6"].upper()
            stored_right = stored["X3p_MSU6"].upper()
            flank_results.append({
                "snp_id":       rec.id,
                "chr":          chrom_int,
                "pos":          rec.pos,
                "left_match":   left  == stored_left,
                "right_match":  right == stored_right,
                "match_excl_N": _match_excl_n(left, stored_left)
                                and _match_excl_n(right, stored_right),
            })

    return {
        "ref_match":          pd.DataFrame(ref_match),
        "ref_mismatch":       pd.DataFrame(ref_mismatch),
        "flank_results":      pd.DataFrame(flank_results),
        "skipped_no_window":  skipped_no_window,
        "skipped_no_flanking": skipped_no_flanking,
        "skipped_edge":       skipped_edge,
    }


def _match_excl_n(a: str, b: str) -> bool:
    return all(x == y or x == "N" or y == "N" for x, y in zip(a, b))


def print_report(out: dict) -> None:
    ref_match    = out["ref_match"]
    ref_mismatch = out["ref_mismatch"]
    flank        = out["flank_results"]

    n_ref = len(ref_match) + len(ref_mismatch)
    print(f"\n── REF allele check ──────────────────────────────────────")
    print(f"REF matches window  : {len(ref_match):,} / {n_ref:,}  "
          f"({100 * len(ref_match) / max(n_ref, 1):.2f}%)")
    print(f"REF mismatches      : {len(ref_mismatch):,}")
    if len(ref_mismatch):
        print(ref_mismatch.head(10).to_string(index=False))

    print(f"\n── Flanking check ────────────────────────────────────────")
    n_f = len(flank)
    if n_f == 0:
        print("No flanking results.")
    else:
        both = flank["left_match"] & flank["right_match"]
        print(f"SNPs checked        : {n_f:,}")
        print(f"Both flanks match   : {both.sum():,} / {n_f:,}  ({100 * both.mean():.2f}%)")
        print(f"Both match excl N   : {flank['match_excl_N'].sum():,} / {n_f:,}  "
              f"({100 * flank['match_excl_N'].mean():.2f}%)")
        print(f"Left only           : {(flank['left_match'] & ~flank['right_match']).sum():,}")
        print(f"Right only          : {(~flank['left_match'] & flank['right_match']).sum():,}")
        print(f"Neither             : {(~flank['left_match'] & ~flank['right_match']).sum():,}")

        per_chr = flank.groupby("chr").apply(
            lambda g: pd.Series({
                "n_snps":     len(g),
                "both_match": (g["left_match"] & g["right_match"]).sum(),
                "pct_match":  f"{100 * (g['left_match'] & g['right_match']).mean():.1f}%",
            }),
            include_groups=False,
        )
        print("\nPer-chromosome match rate:")
        print(per_chr.to_string())

    print(f"\nSkipped — no window for SNP : {out['skipped_no_window']:,}")
    print(f"Skipped — no flanking entry : {out['skipped_no_flanking']:,}")
    print(f"Skipped — flank past window : {out['skipped_edge']:,}")


def build_dataset(vcf_path: str, fasta_path: str, half_window: int, buffer: int) -> UniqueWindowDataset:
    print("Loading SNPs from VCF …")
    snps_by_chrom, samples = load_snps_from_vcf(vcf_path)
    n_snps = sum(len(v) for v in snps_by_chrom.values())
    print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {len(samples)} samples")

    print("Building SNPWindowPartitioner …")
    partitioner = SNPWindowPartitioner(snps_by_chrom, half_window=half_window, buffer=buffer)
    stats = partitioner.snps_per_window_stats()
    print(f"  {stats['n_windows']:,} windows (mean {stats['mean']:.1f} SNPs/window)")

    print("Building UniqueWindowDataset …")
    dataset = UniqueWindowDataset(vcf_path, fasta_path, partitioner)
    print(f"  {len(dataset):,} unique windows; {len(dataset.samples)} samples")
    return dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vcf-path",     type=str, default=DEFAULT_VCF_PATH)
    p.add_argument("--fasta-path",   type=str, default=FASTA_PATH)
    p.add_argument("--flanking-path", type=str, default=FLANKING_PATH)
    p.add_argument("--half-window",  type=int, default=500)
    p.add_argument("--buffer",       type=int, default=0)
    p.add_argument("--flank",        type=int, default=FLANK,
                   help="Flanking half-window for the FLANKING_PATH comparison.")
    args = p.parse_args()

    dataset = build_dataset(args.vcf_path, args.fasta_path, args.half_window, args.buffer)

    print(f"\nLoading flanking-seq table from {args.flanking_path} …")
    flanking_seq = pd.read_csv(args.flanking_path, sep="\t").set_index("snp_id")
    print(f"  {len(flanking_seq):,} SNPs in flanking table")

    out = check_dataset_flanking(dataset, args.vcf_path, flanking_seq, flank=args.flank)
    print_report(out)


if __name__ == "__main__":
    main()

"""
presentation/count_windows.py
-----------------------------
Goal 1 (processing). Sweep the genomic window length L and record, per length:

  * n_windows           : windows the partitioner produces (L = 2*half_window)
  * n_unique_windows    : distinct fingerprints == len(UniqueWindowDataset)
  * mean_tokens         : mean tokenizer token count over a sample of windows
                          (Carbon "<dna>" + 6-mer; capped at --max-length)
  * variants_per_window : n_unique_windows / n_windows (mean distinct haplotypes)

CPU only — no model forward, just the VCF→partition→fingerprint pipeline plus a
tokenizer for the token-count curve. Writes presentation/window_sweep.csv, which
plot_window_sweep.py turns into a figure.

    python presentation/count_windows.py
    python presentation/count_windows.py --lengths 2 100 10000 1000000 --tok-sample 200
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH, DEFAULT_VCF_PATH
from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.partitioner import SNPWindowPartitioner
from crop_embed.dataset import UniqueWindowDataset

# A log-spaced default sweep from ~2 bp to 1e6 bp.
DEFAULT_LENGTHS = [2, 10, 30, 100, 300, 1_000, 3_000, 10_000,
                   30_000, 100_000, 300_000, 1_000_000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep window length; count windows/tokens/variants.")
    p.add_argument("--vcf-path", default=DEFAULT_VCF_PATH)
    p.add_argument("--fasta-path", default=str(FASTA_PATH))
    p.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                   help="Window lengths L (bp) to evaluate; half_window = L//2.")
    p.add_argument("--max-length", type=int, default=2048,
                   help="Tokenizer truncation length (matches the embed runs). NOTE: "
                        "Carbon's slow DNA tokenizer does not enforce this, so the "
                        "token-count curve reflects the true 6-mer count (~L/6).")
    p.add_argument("--tok-sample", type=int, default=400,
                   help="How many unique windows to tokenize per length (speed). "
                        "Set 0 to skip the token-count curve entirely.")
    p.add_argument("--model-path", default="HuggingFaceBio/Carbon-500M",
                   help="Tokenizer to count tokens with (no model weights loaded).")
    p.add_argument("--out", default=str(Path(__file__).parent / "window_sweep.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading SNPs from {args.vcf_path} …")
    snps_by_chrom, samples = load_snps_from_vcf(args.vcf_path)
    n_snps = sum(len(v) for v in snps_by_chrom.values())
    print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {len(samples)} samples")

    tok = None
    if args.tok_sample > 0:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer {args.model_path} …")
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    rows = []
    for L in args.lengths:
        hw = max(1, L // 2)
        part = SNPWindowPartitioner(snps_by_chrom, half_window=hw, buffer=0)
        ds = UniqueWindowDataset(args.vcf_path, args.fasta_path, part)
        n_windows = len(part)
        n_unique = len(ds)

        mean_tokens = float("nan")
        if tok is not None:
            stride = max(1, n_unique // args.tok_sample)
            counts = []
            for i in range(0, n_unique, stride):
                seq = ds[i]["sequence"]
                ids = tok(f"<dna>{seq}", add_special_tokens=False,
                          truncation=True, max_length=args.max_length)["input_ids"]
                counts.append(len(ids))
            mean_tokens = sum(counts) / len(counts)

        row = {
            "length": L,
            "half_window": hw,
            "n_windows": n_windows,
            "n_unique_windows": n_unique,
            "mean_tokens": round(mean_tokens, 2),
            "variants_per_window": round(n_unique / n_windows, 4),
        }
        rows.append(row)
        print(f"  L={L:>9,}  windows={n_windows:>7,}  unique={n_unique:>8,}  "
              f"tokens≈{mean_tokens:7.1f}  variants/win={row['variants_per_window']}")

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

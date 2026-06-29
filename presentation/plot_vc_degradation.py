"""
presentation/plot_vc_degradation.py
-----------------------------------
Goal 4 (plot). Variant-cache embedding fidelity vs. number of concurrent SNPs,
from the CSV written by model_dev/compare_variant_cache_embeddings.py --csv-out.

Generate the data first (needs CUDA — two Carbon-500M forwards per window):

    python -m model_dev.compare_variant_cache_embeddings \
        --half-window 500 --buffer 0 --max-length 2048 --limit 4000 --min-snps 1 \
        --csv-out presentation/vc_degradation.csv

Then:

    python presentation/plot_vc_degradation.py        # -> figs/04_vc_degradation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent


def main() -> int:
    p = argparse.ArgumentParser(description="Plot variant-cache degradation vs #SNPs.")
    p.add_argument("--csv", default=str(HERE / "vc_degradation.csv"))
    p.add_argument("--out", default=str(HERE / "figs" / "04_vc_degradation.png"))
    p.add_argument("--max-snp", type=int, default=10,
                   help="Right edge of the #SNP axis (todo asks for 1-10).")
    args = p.parse_args()

    if not Path(args.csv).exists():
        print(f"{args.csv} not found — run compare_variant_cache_embeddings.py "
              f"--csv-out {args.csv} first (needs GPU).")
        return 1

    df = pd.read_csv(args.csv)
    df = df[df["n_snp"].between(1, args.max_snp)]
    n_per_bin = df.groupby("n_snp").size()
    g = df.groupby("n_snp").mean(numeric_only=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, (axc, axe) = plt.subplots(1, 2, figsize=(13, 4.6))

    # cosine similarity to the true alt forward (higher = better)
    axc.plot(g.index, g["var_cos"], "o-", label="cache, per-SNP-token")
    axc.plot(g.index, g["cache_pool_cos"], "s-", label="cache, window-pooled")
    axc.plot(g.index, g["ref_pool_cos"], "^-", label="reference-only (baseline)")
    axc.set_xlabel("# SNPs in window")
    axc.set_ylabel("cosine vs true alt forward")
    axc.set_title("Variant-cache fidelity vs #SNPs")
    axc.legend(); axc.grid(alpha=.3)

    # relative L2 error of the pooled embedding (lower = better)
    axe.plot(g.index, g["cache_relerr"], "s-", label="cache pooled")
    axe.plot(g.index, g["ref_relerr"], "^-", label="reference-only")
    axe.set_xlabel("# SNPs in window")
    axe.set_ylabel("relative L2 error (pooled)")
    axe.set_title("Pooled-embedding error vs #SNPs")
    axe.legend(); axe.grid(alpha=.3)

    # annotate windows-per-bin so sparse high-SNP bins are visible
    for n, cnt in n_per_bin.items():
        axc.annotate(str(cnt), (n, g.loc[n, "var_cos"]), fontsize=7,
                     textcoords="offset points", xytext=(0, 6), ha="center",
                     color="gray")

    fig.suptitle("SNP-embedding degradation under the variant-cache approximation")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

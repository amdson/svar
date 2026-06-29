"""
presentation/plot_cache_vs_fullforward.py
------------------------------------------
Figure 13 (Claim B, direct). How close are the *window embeddings* produced
through the variant cache to those from the standard full-forward embedder?
Reads the per-window CSV that `train_pipeline/compare_embeddings.py --csv-out`
writes (rows aligned by fingerprint) and plots the relative per-window L2
difference, binned by #SNPs.

Generate the CSV first (CPU; the matched hw500 caches already exist):

    python train_pipeline/compare_embeddings.py \\
        checkpoints/manual/sativas413_carbon500m_vc_hw500.ckpt.pt \\
        checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt \\
        --csv-out presentation/cache_vs_fullforward.csv

Then:

    python presentation/plot_cache_vs_fullforward.py \\
        --csv presentation/cache_vs_fullforward.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="Plot cache vs full-forward window embeddings.")
    p.add_argument("--csv", default=str(HERE / "cache_vs_fullforward.csv"))
    p.add_argument("--out", default=str(HERE / "figs" / "13_vc_vs_fullforward.png"))
    args = p.parse_args()

    if not Path(args.csv).exists():
        print(f"{args.csv} not found — run compare_embeddings.py --csv-out first.")
        return 1

    df = pd.read_csv(args.csv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: distribution of relative per-window L2 difference.
    ax_h.hist(df["rel_diff"], bins=60, color="tab:blue", alpha=0.8)
    ax_h.set_xlabel("relative L2 difference  ||vc - full|| / ||full||")
    ax_h.set_ylabel("# windows")
    ax_h.set_title("Per-window cache vs. full-forward difference")
    ax_h.axvline(df["rel_diff"].mean(), color="k", ls="--",
                 label=f"mean = {df['rel_diff'].mean():.2e}")
    ax_h.legend(); ax_h.grid(alpha=.3)

    # Right: mean relative difference vs #SNPs (the cache's approximation grows
    # mildly with concurrent SNPs; reference windows are exact).
    g = df[df["n_snp"].between(0, 10)].groupby("n_snp")["rel_diff"].mean()
    ax_b.plot(g.index, g.values, "o-")
    ax_b.set_xlabel("# SNPs in window")
    ax_b.set_ylabel("mean relative L2 difference")
    ax_b.set_title("Difference vs. #SNPs")
    ax_b.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

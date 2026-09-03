"""
Gate 0 of the SIEVE fine-tuning sanity ladder: is there detectable cis-mutation
signal in the expression deviations at all, before any model touches them?

Focus pairs (line has >=1 cis mutation at the gene) are compared against
background pairs (same lines, genes with no cis mutation) — the matched null:
identical trans effects, batch structure, and measurement noise, differing only
in the presence of a cis mutation. Deviations are z-scored per gene by the
control-line deviation std so genes with different expression noise pool fairly.

Reports:
  * per-gene control noise floor (std of control deviations)
  * focus vs background |z| and z-variance, with and without the per-line
    offset correction (line's mean deviation over its background genes)
  * tail enrichment: P(|z| > 2) and P(|z| > 3), focus / background

Gate: focus z-variance (or tail rate) must exceed background by a margin that
survives the line-offset correction. If it doesn't, there is nothing for any
cis model to learn and the fine-tuning gates are moot.

    python -m model_dev.sieve_signal_gate
"""
from __future__ import annotations

import sys

import numpy as np

BASE = "/90daydata/small_grains/andrew.dickson/datasets/brachypodium_sieve/dataset/"


def main() -> int:
    import h5py
    import pandas as pd

    with h5py.File(BASE + "sieve_dataset.h5", "r") as f:
        dev = f["deviation"][:]                    # (genes, lines)
        gene_id = f["genes/gene_id"][:].astype(str)
        is_ctrl = f["lines/is_control"][:]
        line_id = f["lines/line_id"][:].astype(str)

    gene_ix = {g: i for i, g in enumerate(gene_id)}
    line_ix = {l: i for i, l in enumerate(line_id)}

    # noise floor: std of control-line deviations per gene (ddof=1)
    ctrl = dev[:, is_ctrl]
    sd = ctrl.std(axis=1, ddof=1)
    expressed = sd > 1e-3  # unexpressed/degenerate genes can't be scored
    print(f"genes: {len(gene_id):,}; scoreable (control sd > 1e-3): "
          f"{expressed.sum():,}")
    print(f"control-line deviation sd per gene: p50={np.median(sd[expressed]):.4f} "
          f"p90={np.percentile(sd[expressed], 90):.4f}  (log2 units)")

    pairs = pd.read_parquet(BASE + "sieve_pairs.parquet")
    pairs = pairs[pairs["gene_id"].isin(gene_ix) & pairs["line"].isin(line_ix)]
    gi = pairs["gene_id"].map(gene_ix).to_numpy()
    li = pairs["line"].map(line_ix).to_numpy()
    keep = expressed[gi]
    pairs, gi, li = pairs[keep], gi[keep], li[keep]
    z = dev[gi, li] / sd[gi]
    focus = (pairs["role"] == "focus").to_numpy()
    print(f"pairs scored: {focus.sum():,} focus, {(~focus).sum():,} background")

    # per-line offset from that line's background genes (trans/technical shift)
    n_lines = dev.shape[1]
    off_sum = np.zeros(n_lines)
    off_cnt = np.zeros(n_lines)
    np.add.at(off_sum, li[~focus], z[~focus])
    np.add.at(off_cnt, li[~focus], 1)
    offset = np.where(off_cnt > 0, off_sum / np.maximum(off_cnt, 1), 0.0)
    z_adj = z - offset[li]

    def report(tag, zz):
        f, b = zz[focus], zz[~focus]
        var_f, var_b = f.var(), b.var()
        print(f"\n[{tag}]")
        print(f"  z-variance: focus {var_f:.3f} vs background {var_b:.3f} "
              f"-> excess {var_f - var_b:+.3f} ({(var_f/var_b - 1)*100:+.1f}%)")
        for thr in (2.0, 3.0):
            pf, pb = (np.abs(f) > thr).mean(), (np.abs(b) > thr).mean()
            print(f"  P(|z| > {thr:.0f}): focus {pf:.4f} vs background {pb:.4f} "
                  f"-> enrichment x{pf/pb if pb > 0 else float('inf'):.2f}")
        # crude significance: variance-ratio z via bootstrap over lines
        rng = np.random.default_rng(0)
        lines_f = li[focus]
        uniq = np.unique(lines_f)
        boots = []
        for _ in range(200):
            take = rng.choice(uniq, size=len(uniq), replace=True)
            mask = np.isin(lines_f, take)
            boots.append(f[mask].var() - var_b)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  excess variance 95% CI (line bootstrap): [{lo:+.3f}, {hi:+.3f}]")

    report("raw z", z)
    report("line-offset corrected", z_adj)

    # where the signal sits: focus |z_adj| by n_cis_mut
    ncm = pairs["n_cis_mut"].to_numpy()
    print("\nfocus pairs by n_cis_mut (line-offset corrected):")
    for k in (1, 2, 3):
        m = focus & (ncm == k if k < 3 else ncm >= k)
        if m.sum():
            print(f"  n_cis_mut{'>=' if k >= 3 else '=':>2}{k}: n={m.sum():>7,}  "
                  f"var={z_adj[m].var():.3f}  P(|z|>2)={np.mean(np.abs(z_adj[m])>2):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

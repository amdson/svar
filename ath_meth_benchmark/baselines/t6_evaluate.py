#!/usr/bin/env python3
"""SIEVE evaluation — per-mutation contrasts on the SIEVE pair table.

SIEVE is a standalone Brachypodium benchmark (induced mutations, no LD,
isogenic background); these metrics apply to any model that produces a
predicted deviation per (gene, line) pair, whether fine-tuned within
Brachypodium or evaluated zero-shot from elsewhere.

Predictions: parquet/TSV with columns (gene_id, line, pred), where pred is
the model's predicted expression deviation for that (gene, line) pair —
typically model(mutant window) − model(reference window).

Metrics (focus pairs = pairs carrying >=1 cis mutation):
  pooled beta   — slope of observed deviation ~ predicted (EMPRES's metric;
                  their published bar is beta = 0.38)
  pooled r      — Pearson and Spearman over focus pairs
  sign conc.    — fraction of focus pairs where sign(pred) == sign(obs),
                  overall and restricted to |obs| above the control-noise
                  floor (95th percentile of |background deviation|)
  per-gene r    — Spearman across lines, genes with >= --min-per-gene focus
                  pairs only (sparse; reported as a secondary distribution)
  calibration   — Var(pred)/Var(obs) on focus pairs (EMPRES saw ~5-10x
                  compression; report explicitly)
Background pairs sanity check: median |pred| on wild-type pairs should be
~0; a model predicting effects everywhere fails calibration.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="sieve_pairs.parquet")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--min-per-gene", type=int, default=5)
    a = ap.parse_args()

    pairs = pd.read_parquet(a.pairs)
    pred = pd.read_parquet(a.pred) if a.pred.endswith(".parquet") else \
        pd.read_csv(a.pred, sep="\t")
    df = pairs.merge(pred, on=["gene_id", "line"], how="left")
    n_missing = int(df["pred"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing:,}/{len(df):,} pairs lack predictions")
        df = df.dropna(subset=["pred"])

    foc = df[df["role"] == "focus"]
    bg = df[df["role"] == "background"]
    y, p = foc["deviation"].to_numpy(), foc["pred"].to_numpy()

    beta = np.polyfit(p, y, 1)[0] if p.std() > 0 else float("nan")
    noise_floor = np.quantile(np.abs(bg["deviation"]), 0.95) if len(bg) else 0.0
    big = np.abs(y) > noise_floor
    sign_all = float((np.sign(p) == np.sign(y)).mean())
    sign_big = float((np.sign(p[big]) == np.sign(y[big])).mean()) if big.any() else float("nan")

    per_gene = []
    for g, sub in foc.groupby("gene_id"):
        if len(sub) >= a.min_per_gene and sub["pred"].std() > 0:
            per_gene.append(spearmanr(sub["pred"], sub["deviation"]).statistic)

    print(f"focus pairs: {len(foc):,}   background pairs: {len(bg):,}")
    print(f"pooled beta (obs ~ pred): {beta:.3f}   [EMPRES bar: 0.38]")
    print(f"pooled Pearson r: {pearsonr(p, y).statistic:.3f}   "
          f"Spearman: {spearmanr(p, y).statistic:.3f}")
    print(f"sign concordance: {sign_all:.3f} (all)  {sign_big:.3f} "
          f"(|obs| > control-noise p95 = {noise_floor:.3f}; n={int(big.sum()):,})")
    print(f"calibration Var(pred)/Var(obs): {p.var() / y.var():.3f}")
    print(f"per-gene Spearman (n>={a.min_per_gene} pairs): "
          f"{len(per_gene):,} genes, median "
          f"{np.median(per_gene) if per_gene else float('nan'):.3f}")
    if len(bg):
        print(f"background |pred| median: {np.abs(bg['pred']).median():.4f} "
              f"(should be ~0; wild-type pairs)")


if __name__ == "__main__":
    main()

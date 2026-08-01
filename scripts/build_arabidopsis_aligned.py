"""
scripts/build_arabidopsis_aligned.py
------------------------------------
Turn the replicate-keyed AraPheno matrix (`arabidopsis_pheno_matrix.csv`, keyed
by (IID, rep)) into the one-row-per-IID aligned matrix the training pipeline
consumes (`arabidopsis_pheno_aligned.csv`, keyed by IID — same contract as
soy_pheno_aligned.csv).

Collapse: mean across replicates per (IID, trait), NaN-preserving (an accession
with no measurement for a trait stays NaN). Also prints the top-N traits by
post-collapse coverage (accessions with a value), which is what to pass as
--traits for a baseline sweep.

    python scripts/build_arabidopsis_aligned.py            # writes aligned CSV, prints top-20
    python scripts/build_arabidopsis_aligned.py --top 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path.home() / "svar_scratch" / "datasets" / "arabidopsis"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="arabidopsis dataset dir (holds arabidopsis_pheno_matrix.csv)")
    ap.add_argument("--top", type=int, default=20, help="how many best-covered traits to report")
    args = ap.parse_args()

    src = args.dir / "arabidopsis_pheno_matrix.csv"
    out = args.dir / "arabidopsis_pheno_aligned.csv"
    df = pd.read_csv(src)

    trait_cols = [c for c in df.columns if c not in ("IID", "rep")]
    # Mean across replicates per accession; NaN where an accession never measured a trait.
    aligned = df.groupby("IID", as_index=True)[trait_cols].mean()
    aligned.index.name = "IID"
    aligned.to_csv(out)

    cov = aligned.notna().sum().sort_values(ascending=False)
    print(f"wrote {out}")
    print(f"  {aligned.shape[0]} accessions x {aligned.shape[1]} traits "
          f"(from {len(df)} (IID,rep) rows)")
    top = cov.head(args.top)
    print(f"\nTop {args.top} traits by coverage (accessions with a value):")
    for tid, n in top.items():
        print(f"  {str(tid):8s} {n}")
    print("\n--traits for a sweep:")
    print("  " + ",".join(str(t) for t in top.index))


if __name__ == "__main__":
    main()

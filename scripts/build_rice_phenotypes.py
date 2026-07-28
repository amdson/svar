#!/usr/bin/env python
"""
build_rice_phenotypes.py
------------------------
Rewrite the RiceDiversity 44K phenotype table (keyed by NSFTVID) into the
canonical **IID-keyed** form the training pipeline expects, so rice matches soy's
`soy_pheno_aligned.csv` contract (training/common/features.load_targets reindexes
a pheno CSV by the VCF sample IID).

Rice VCF sample IIDs look like ``081215-A05_1``; the trailing integer after the
last underscore is the NSFTVID join key (same rule as
crop_embed.data.preprocessing.align_targets_to_dataset). Trait column order is
preserved from the source table.

    python scripts/build_rice_phenotypes.py            # writes into the rice data dir
    python scripts/build_rice_phenotypes.py --out /path/rice_pheno_aligned.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crop_embed.data.coords import DEFAULT_PHENO_PATH, DEFAULT_VCF_PATH
from crop_embed.data.preprocessing import load_phenotypes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pheno", default=DEFAULT_PHENO_PATH, help="NSFTVID-keyed 34-trait table")
    ap.add_argument("--vcf", default=DEFAULT_VCF_PATH, help="rice VCF (source of IIDs)")
    ap.add_argument("--out", default=None, help="output CSV (default: <rice dir>/rice_pheno_aligned.csv)")
    args = ap.parse_args()

    pheno_df, trait_cols = load_phenotypes(args.pheno)          # NSFTVID + traits (canonical order)
    by_nsftvid = pheno_df.set_index("NSFTVID")

    import pysam
    with pysam.VariantFile(args.vcf) as vf:
        iids = list(vf.header.samples)
    nsftvids = [int(s.rsplit("_", 1)[-1]) for s in iids]

    aligned = by_nsftvid.reindex(nsftvids)[trait_cols].copy()
    aligned.insert(0, "IID", iids)

    out = args.out or str(Path(args.vcf).resolve().parent / "rice_pheno_aligned.csv")
    aligned.to_csv(out, index=False)
    n_match = aligned[trait_cols].notna().any(axis=1).sum()
    print(f"wrote {out}")
    print(f"  {len(aligned)} IIDs, {len(trait_cols)} traits, {int(n_match)} with phenotype data")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T2 family-aware gene splits (design doc T2, PhytoExpr-style).

Genes are split by PLAZA dicots-05 HOMFAM homologous gene family — never
individually — so paralogs cannot leak across train/eval. Genes absent from
PLAZA become singleton families. Families are shuffled (seeded) and assigned
to train/val/test by cumulative gene count at ~70/10/20.

Writes genes/family_id and genes/family_split into expression_dataset.h5
and prints the summary.
"""
import argparse

import h5py
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--homfam", required=True)
    ap.add_argument("--fractions", default="0.7,0.1,0.2")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    f_train, f_val, f_test = map(float, a.fractions.split(","))

    fam = pd.read_csv(a.homfam, sep="\t", comment="#", header=None,
                      names=["gf_id", "species", "gene_id"])
    fam = fam[fam["species"] == "ath"]
    gene2fam = dict(zip(fam["gene_id"].str.upper(), fam["gf_id"]))

    with h5py.File(a.h5, "r") as h5:
        genes = [g.decode() if isinstance(g, bytes) else g
                 for g in h5["genes/gene_id"][:]]

    fam_ids = [gene2fam.get(g.upper(), f"SINGLETON_{g}") for g in genes]
    n_plaza = sum(1 for f in fam_ids if not f.startswith("SINGLETON_"))
    print(f"{len(genes):,} genes; {n_plaza:,} in PLAZA families, "
          f"{len(genes) - n_plaza:,} singletons")

    fam_series = pd.Series(fam_ids, index=genes)
    sizes = fam_series.value_counts()
    rng = np.random.default_rng(a.seed)
    order = sizes.index.to_numpy()
    rng.shuffle(order)
    cum = np.cumsum(sizes[order].to_numpy()) / len(genes)
    fam_split = {}
    for f, c in zip(order, cum):
        fam_split[f] = ("train" if c <= f_train else
                        "val" if c <= f_train + f_val else "test")
    split = fam_series.map(fam_split)
    print("gene counts by family_split:", split.value_counts().to_dict())
    print("family counts by split:",
          pd.Series(fam_split).value_counts().to_dict())

    str_dt = h5py.string_dtype()
    with h5py.File(a.h5, "a") as h5:
        for name, data in [("family_id", fam_ids),
                           ("family_split", split.to_numpy(dtype=object))]:
            path = f"genes/{name}"
            if path in h5:
                del h5[path]
            h5.create_dataset(path, data=np.array(data, dtype=object), dtype=str_dt)
    print("written to", a.h5)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Family-aware gene splits for the SIEVE benchmark.

The Bd21-3 v1.1 annotation carries no Bradi synonyms, so PhytoExpr/EMPRES
family ids cannot be joined directly. v1 proxy: genes sharing the same
Best-hit-arabi-name (fallback: Best-hit-rice-name; fallback: singleton) from
the Phytozome annotation_info are treated as one family — Brachypodium
paralogs almost always share their Arabidopsis best hit, which is the
leakage this split exists to prevent. Whole families go to train/val/test
(~70/10/20 by gene count), mirroring expr_family_split.py.

Writes genes/family_id and genes/family_split into sieve_dataset.h5.
"""
import argparse
import gzip

import h5py
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--annotation-info", required=True)
    ap.add_argument("--fractions", default="0.7,0.1,0.2")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    f_train, f_val, f_test = map(float, a.fractions.split(","))

    ann = pd.read_csv(a.annotation_info, sep="\t", dtype=str).fillna("")
    ann["gene"] = ann["locusName"].str.replace("^BdiBd21-3\\.", "", regex=True)
    ann = ann.drop_duplicates("gene").set_index("gene")

    with h5py.File(a.h5, "r") as h5:
        genes = [g.decode() if isinstance(g, bytes) else g
                 for g in h5["genes/gene_id"][:]]

    at = ann["Best-hit-arabi-name"].reindex(genes).fillna("")
    rice = ann["Best-hit-rice-name"].reindex(genes).fillna("")
    fam = ["AT:" + a_ if a_ else ("OS:" + r if r else f"SINGLETON_{g}")
           for g, a_, r in zip(genes, at, rice)]
    n_at = sum(1 for f in fam if f.startswith("AT:"))
    n_os = sum(1 for f in fam if f.startswith("OS:"))
    print(f"{len(genes):,} genes: {n_at:,} arabi-hit families, "
          f"{n_os:,} rice-hit, {len(genes)-n_at-n_os:,} singletons")

    fam_series = pd.Series(fam, index=genes)
    sizes = fam_series.value_counts()
    rng = np.random.default_rng(a.seed)
    order = sizes.index.to_numpy()
    rng.shuffle(order)
    cum = np.cumsum(sizes[order].to_numpy()) / len(genes)
    fam_split = {f: ("train" if c <= f_train else
                     "val" if c <= f_train + f_val else "test")
                 for f, c in zip(order, cum)}
    split = fam_series.map(fam_split)
    print("gene counts by family_split:", split.value_counts().to_dict())

    str_dt = h5py.string_dtype()
    with h5py.File(a.h5, "a") as h5:
        for name, data in [("family_id", fam),
                           ("family_split", split.to_numpy(dtype=object))]:
            path = f"genes/{name}"
            if path in h5:
                del h5[path]
            h5.create_dataset(path, data=np.array(data, dtype=object), dtype=str_dt)
    print("written to", a.h5)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluation harness for the cross-accession expression benchmark (T1/T2).

Input predictions: a parquet/TSV with columns gene_id, ecotype_id, pred
(long format), or an .npy matrix aligned to the dataset's gene x accession
order. Metric (design doc §4): per-gene Spearman r across EVAL accessions
(acc_split == test by default), median over genes, never pooled over
(gene, accession) pairs.

Modes:
  T1  eval genes = all (or --primary cis-h2 >= threshold via sandwich parquet)
  T2  eval genes = family_split == test  (unseen families x unseen accessions)

Strata reported: cis-h2 bins (if sandwich parquet given), family_split,
chromosome split.
"""
import argparse

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--pred", required=True,
                    help="long-format parquet/tsv (gene_id, ecotype_id, pred) "
                         "or .npy matrix (genes x accessions, dataset order)")
    ap.add_argument("--mode", choices=["T1", "T2"], default="T1")
    ap.add_argument("--acc-eval", default="test", choices=["test", "val"])
    ap.add_argument("--sandwich", default=None,
                    help="t1_sandwich.parquet for cis-h2 strata")
    ap.add_argument("--h2-threshold", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with h5py.File(a.h5, "r") as h5:
        gene_id = np.array([x.decode() for x in h5["genes/gene_id"][:]])
        fam_split = (np.array([x.decode() for x in h5["genes/family_split"][:]])
                     if "genes/family_split" in h5 else None)
        acc_id = np.array([x.decode() for x in h5["accessions/ecotype_id"][:]])
        acc_split = np.array([x.decode() for x in h5["accessions/acc_split"][:]])
        Y = h5["deviation"][:]

    eval_acc = np.flatnonzero(acc_split == a.acc_eval)

    if a.pred.endswith(".npy"):
        P = np.load(a.pred)
        assert P.shape == Y.shape, f"pred shape {P.shape} != dataset {Y.shape}"
    else:
        rd = pd.read_parquet(a.pred) if a.pred.endswith(".parquet") else \
            pd.read_csv(a.pred, sep="\t", dtype={"ecotype_id": str})
        gi = {g: i for i, g in enumerate(gene_id)}
        ai = {s: i for i, s in enumerate(acc_id)}
        P = np.full(Y.shape, np.nan, dtype=np.float32)
        P[rd["gene_id"].map(gi), rd["ecotype_id"].map(ai)] = rd["pred"]

    genes = np.arange(len(gene_id))
    if a.mode == "T2":
        assert fam_split is not None, "run expr_family_split.py first"
        genes = genes[fam_split[genes] == "test"]

    sw = pd.read_parquet(a.sandwich).set_index("gene_id") if a.sandwich else None
    if sw is not None and a.mode == "T1":
        primary = set(sw.index[sw["cis_h2"] >= a.h2_threshold])
        genes_primary = np.array([g for g in genes if gene_id[g] in primary])
    else:
        genes_primary = None

    def per_gene_r(gsel):
        rs = []
        for g in gsel:
            p, y = P[g, eval_acc], Y[g, eval_acc]
            ok = ~np.isnan(p)
            if ok.sum() >= 10 and np.std(p[ok]) > 0 and np.std(y[ok]) > 0:
                rs.append(spearmanr(p[ok], y[ok]).statistic)
            else:
                rs.append(np.nan)
        return np.array(rs)

    r_all = per_gene_r(genes)
    lines = [f"# {a.mode} evaluation — {a.pred}",
             f"eval accessions: {a.acc_eval} (n={len(eval_acc)}); "
             f"genes evaluated: {len(genes):,}",
             f"median per-gene Spearman r: {np.nanmedian(r_all):.4f}  "
             f"(mean {np.nanmean(r_all):.4f}; "
             f"{int(np.isnan(r_all).sum())} genes undefined)"]
    if genes_primary is not None and len(genes_primary):
        r_p = per_gene_r(genes_primary)
        lines.append(f"primary set (cis-h2 >= {a.h2_threshold}, "
                     f"n={len(genes_primary):,}): median r {np.nanmedian(r_p):.4f}")
    if sw is not None:
        h2 = sw.reindex(gene_id[genes])["cis_h2"].to_numpy()
        for lo, hi in [(0, .05), (.05, .1), (.1, .25), (.25, .5), (.5, 1.01)]:
            m = (h2 >= lo) & (h2 < hi)
            if m.sum():
                lines.append(f"  cis-h2 [{lo},{hi}): n={int(m.sum()):,} "
                             f"median r {np.nanmedian(r_all[m]):.4f}")
    report = "\n".join(lines)
    print(report)
    if a.out:
        pd.DataFrame({"gene_id": gene_id[genes], "r": r_all}).to_parquet(a.out)


if __name__ == "__main__":
    main()

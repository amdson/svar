#!/usr/bin/env python3
"""Expression benchmark core (design doc dataset #2, tasks T1/T2 data side).

From GSE80744 gene-normalized counts (728 accessions, rosette leaf):
  - intersect accessions with the 1001G v3.1 VCF panel
  - targets: log2(normCount+1); per-gene across-accession mean/var; the
    training/eval target is the per-gene DEVIATION from the accession-panel
    mean (never absolute level, per design doc §5)
  - accession QC: flag outlier accessions by correlation of their expression
    profile to the panel mean profile (Kawakatsu 2016 excluded one)
  - TSS/TTS coordinates per gene from the Ensembl GFF3 (for +-window inputs)
  - splits: accession axis by admixture group (same defaults as the
    methylation benchmark: test=italy_balkan_caucasus, val=asia);
    gene position axis chr1-3/4/5 train/val/test (family-aware T2 splits are
    a later, separate artifact)

Output: expression_dataset.h5 + expr_report.md
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CHROMS = ["1", "2", "3", "4", "5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--psam", required=True)
    ap.add_argument("--acc-meta", required=True, help="accessions_1001g.csv")
    ap.add_argument("--gff3", required=True)
    ap.add_argument("--meth-accessions", required=True,
                    help="benchmark_accessions.tsv from the methylation build")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--acc-test-groups", default="italy_balkan_caucasus")
    ap.add_argument("--acc-val-groups", default="asia")
    ap.add_argument("--outlier-r", type=float, default=0.8,
                    help="flag accessions whose profile corr to panel mean is below this")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- expression matrix ----
    df = pd.read_csv(a.counts, sep="\t", index_col=0)
    df.columns = [c.lstrip("X") for c in df.columns]
    n_matrix_acc = df.shape[1]
    print(f"counts: {df.shape[0]:,} genes x {n_matrix_acc} accessions")

    vcf_ids = set()
    with open(a.psam) as fh:
        next(fh)
        for line in fh:
            vcf_ids.add(line.split()[0].strip())
    keep = [c for c in df.columns if c in vcf_ids]
    print(f"intersection with VCF panel: {len(keep)}")
    df = df[keep]

    # ---- accession metadata / splits ----
    acc_meta = {}
    with open(a.acc_meta) as fh:
        for r in csv.reader(fh):
            acc_meta[r[0]] = r[10]  # admixture_group
    groups = pd.Series({c: acc_meta.get(c, "") for c in keep})
    test_g = set(a.acc_test_groups.split(","))
    val_g = set(a.acc_val_groups.split(","))
    acc_split = groups.map(lambda g: "test" if g in test_g else
                           "val" if g in val_g else "train")

    meth = pd.read_csv(a.meth_accessions, sep="\t", dtype=str)
    in_meth = pd.Series([c in set(meth["ecotype_id"]) for c in keep], index=keep)

    # ---- targets ----
    X = np.log2(df.to_numpy(np.float64) + 1.0)

    # accession outlier QC: corr of each accession profile to the mean profile
    mean_prof = X.mean(axis=1)
    Xc = X - X.mean(axis=0, keepdims=True)
    mc = mean_prof - mean_prof.mean()
    r_acc = (Xc * mc[:, None]).sum(0) / (
        np.sqrt((Xc ** 2).sum(0) * (mc ** 2).sum()) + 1e-12)
    outlier = r_acc < a.outlier_r
    print(f"outlier accessions (r<{a.outlier_r}): "
          f"{[(keep[i], round(r_acc[i],3)) for i in np.flatnonzero(outlier)]}")

    ok = ~outlier
    Xok = X[:, ok]
    keep_ok = [c for c, o in zip(keep, ok) if o]
    gene_mean = Xok.mean(axis=1)
    gene_var = Xok.var(axis=1, ddof=1)
    dev = Xok - gene_mean[:, None]

    # ---- gene coordinates ----
    gf = pd.read_csv(a.gff3, sep="\t", comment="#", header=None,
                     names=["chrom", "src", "type", "start", "end", "score",
                            "strand", "frame", "attr"], dtype={"chrom": str})
    genes = gf[(gf["type"] == "gene") & gf["chrom"].isin(CHROMS)].copy()
    genes["gene_id"] = genes["attr"].str.extract(r"ID=gene:([^;]+)")
    genes = genes.set_index("gene_id")
    common = df.index.intersection(genes.index)
    print(f"genes with coordinates: {len(common):,} of {df.shape[0]:,}")

    gsub = genes.loc[common]
    tss = np.where(gsub["strand"] == "+", gsub["start"], gsub["end"])
    tts = np.where(gsub["strand"] == "+", gsub["end"], gsub["start"])
    gpos_split = np.where(gsub["chrom"] == "4", "val",
                          np.where(gsub["chrom"] == "5", "test", "train"))

    sel = df.index.get_indexer(common)
    str_dt = h5py.string_dtype()
    with h5py.File(out / "expression_dataset.h5", "w") as h5:
        gg = h5.create_group("genes")
        gg.create_dataset("gene_id", data=np.array(common, dtype=object), dtype=str_dt)
        gg.create_dataset("chrom", data=gsub["chrom"].to_numpy(dtype=object), dtype=str_dt)
        gg.create_dataset("strand", data=gsub["strand"].to_numpy(dtype=object), dtype=str_dt)
        gg.create_dataset("start", data=gsub["start"].to_numpy(np.int64))
        gg.create_dataset("end", data=gsub["end"].to_numpy(np.int64))
        gg.create_dataset("tss", data=tss.astype(np.int64))
        gg.create_dataset("tts", data=tts.astype(np.int64))
        gg.create_dataset("pos_split", data=gpos_split.astype(object), dtype=str_dt)
        gg.create_dataset("mean_log2", data=gene_mean[sel].astype(np.float32))
        gg.create_dataset("var_log2", data=gene_var[sel].astype(np.float32))
        ga = h5.create_group("accessions")
        ga.create_dataset("ecotype_id", data=np.array(keep_ok, dtype=object), dtype=str_dt)
        ga.create_dataset("admixture_group",
                          data=groups[keep_ok].to_numpy(dtype=object), dtype=str_dt)
        ga.create_dataset("acc_split",
                          data=acc_split[keep_ok].to_numpy(dtype=object), dtype=str_dt)
        ga.create_dataset("in_methylation_benchmark",
                          data=in_meth[keep_ok].to_numpy(bool))
        h5.create_dataset("log2_expr", data=Xok[sel].astype(np.float32),
                          chunks=(1024, len(keep_ok)), compression="lzf")
        h5.create_dataset("deviation", data=dev[sel].astype(np.float32),
                          chunks=(1024, len(keep_ok)), compression="lzf")
        h5.attrs["source"] = "GSE80744 UQ+gene-normalized counts k4 (Kawakatsu 2016)"
        h5.attrs["target"] = ("deviation = log2(normCount+1) - per-gene panel mean; "
                              "never predict absolute level (design doc §5)")

    n_by = acc_split[keep_ok].value_counts().to_dict()
    both = int(in_meth[keep_ok].sum())
    report = [
        "# Expression dataset build report\n",
        f"- GSE80744 accessions: {n_matrix_acc} in matrix header; "
        f"**{len(keep)} intersect the 1001G VCF panel**",
        f"- outliers excluded (profile corr < {a.outlier_r}): "
        f"{[(keep[i], round(float(r_acc[i]),3)) for i in np.flatnonzero(outlier)]}",
        f"- final accessions: **{len(keep_ok)}**; splits: {n_by}",
        f"- overlap with methylation benchmark (811): **{both}** accessions "
        "(joint multi-task subset)",
        f"- genes: {len(common):,} with Ensembl coordinates "
        f"(of {df.shape[0]:,} in matrix)",
        f"- gene split (chr1-3/4/5): "
        f"{pd.Series(gpos_split).value_counts().to_dict()}",
        "\nCaveats: batch/k4 normalization is Kawakatsu's published pipeline; "
        "cis-h2 ceilings and family-aware T2 gene splits are separate follow-ups; "
        "one condition, one tissue - accession main effect only.",
    ]
    (out / "expr_report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()

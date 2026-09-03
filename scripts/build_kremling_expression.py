#!/usr/bin/env python3
"""Kremling 2018 maize expression dataset -> kremling_dataset.h5.

Mirrors ath_meth_benchmark/build/expr_build.py's conventions (deviation
targets, TSS/TTS from GFF, h5 layout + report) with one addition: a TISSUE
axis. From the DESeq2-normalized FPM matrix (1,771 samples = 7 tissues x
~300 Goodman-panel lines, AGPv3.29 gene models):

  - sample name `seqid_TISSUEbatch_line_barcode` -> (tissue, line); L3Mid
    dropped (readme: order of magnitude fewer samples, unused in the paper)
  - log2(1+FPM), then average replicate samples per (tissue, line)
  - line names matched to the genotype taxa (282set_ prefix / Goodman-Buckler
    suffix stripped, non-alphanumerics ignored); only matched lines kept
  - deviation[t, g, l] = log2 expr - per-(tissue, gene) mean over lines with
    data (NaN where a line lacks that tissue) — never predict absolute level
  - splits: line axis random 70/15/15 seed 42 (relatedness-aware splits are a
    follow-up; the Goodman panel has structure); gene axis by chromosome
    (train 1-7, test 8+10, val 9)

Run after `make` in datasets/maize_kremling (needs the .psam for taxa):
    python scripts/build_kremling_expression.py
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA = Path("/90daydata/small_grains/andrew.dickson/datasets/maize_kremling")
EXPR = DATA / ("df_STAR_HTSeq_counts_B73_match_based_on_genet_dist_"
               "DESeq2_normed_fpm_rounded.txt")
GFF = DATA / "Zea_mays.AGPv3.29.gff3.gz"
PSAM = DATA / "kremling_agpv3.psam"
OUT = DATA / "expression"

TISSUES = ["GRoot", "GShoot", "Kern", "L3Base", "L3Tip", "LMAD", "LMAN"]
CHROMS = [str(c) for c in range(1, 11)]


# Expression-side stock names that differ from the hmp321 taxa but are the same
# inbred (verified against the 282-panel naming): suffix variants and the W22
# R-r:std standard stock. Applied after normalization.
ALIASES = {"IL101T": "IL101", "IL677A": "I1677A", "KUI2021": "KI2021",
           "YU796NS": "YU796", "W22RRSTD": "W22"}


def norm_name(s: str) -> str:
    s = re.sub(r"^282set_", "", s)
    s = re.sub(r"Goodman-Buckler$", "", s)
    s = re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return ALIASES.get(s, s)


def main() -> int:
    OUT.mkdir(exist_ok=True)

    # ---- genotype taxa ----
    taxa = [l.split()[0] for l in open(PSAM) if not l.startswith("#")]
    taxa_by_norm = {}
    for t in taxa:
        taxa_by_norm.setdefault(norm_name(t), t)
    print(f"genotype taxa: {len(taxa)} ({len(taxa_by_norm)} unique normalized)")

    # ---- expression matrix ----
    df = pd.read_csv(EXPR, sep=r"\s+", index_col=0)
    print(f"expression: {df.shape[0]:,} samples x {df.shape[1]:,} genes")

    # Sample names are `[prefix_]seqid_TISSUE[batch]_line[_more]_barcode` — the
    # tissue token's position varies (LMAD26 at index 1, GRoot at index 2 with a
    # KAKRNAx prefix), so locate it by name; the line is everything between it
    # and the trailing barcode.
    tissue_set = set(TISSUES) | {"L3Mid"}
    meta = []
    for name in df.index:
        tok = name.split("_")
        t_ix = next((i for i, t in enumerate(tok)
                     if re.sub(r"\d+$", "", t) in tissue_set), None)
        if t_ix is None or t_ix + 2 > len(tok) - 1:
            meta.append((None, None))
            continue
        meta.append((re.sub(r"\d+$", "", tok[t_ix]),
                     "_".join(tok[t_ix + 1:-1])))
    meta = pd.DataFrame(meta, index=df.index, columns=["tissue", "line"])
    keep = meta["tissue"].isin(TISSUES)
    print(f"samples kept (7 tissues): {keep.sum():,} "
          f"(dropped {(~keep).sum()} incl. L3Mid/unparseable)")
    df, meta = df[keep.to_numpy()], meta[keep.to_numpy()]

    meta["norm"] = meta["line"].map(norm_name)
    matched = meta["norm"].isin(taxa_by_norm)
    unmatched_lines = sorted(meta.loc[~matched, "line"].unique())
    print(f"line match to genotypes: {meta.loc[matched, 'norm'].nunique()} "
          f"matched lines; {len(unmatched_lines)} expression-only lines "
          f"dropped: {unmatched_lines[:10]}{'...' if len(unmatched_lines) > 10 else ''}")
    df, meta = df[matched.to_numpy()], meta[matched.to_numpy()]

    # ---- log2 + average replicates per (tissue, line) ----
    log2 = np.log2(1.0 + df.to_numpy(np.float32))
    lines = sorted(meta["norm"].unique())
    line_ix = {l: i for i, l in enumerate(lines)}
    tis_ix = {t: i for i, t in enumerate(TISSUES)}
    G = df.shape[1]
    sums = np.zeros((len(TISSUES), len(lines), G), np.float64)
    cnts = np.zeros((len(TISSUES), len(lines)), np.int16)
    for row, (t, l) in enumerate(zip(meta["tissue"], meta["norm"])):
        sums[tis_ix[t], line_ix[l]] += log2[row]
        cnts[tis_ix[t], line_ix[l]] += 1
    with np.errstate(invalid="ignore"):
        expr = (sums / cnts[:, :, None]).astype(np.float32)  # NaN where cnt=0
    expr = np.transpose(expr, (0, 2, 1))                     # (T, G, L)
    print("lines x tissue coverage:",
          {t: int((cnts[i] > 0).sum()) for t, i in tis_ix.items()})

    # ---- genes from GFF ----
    rows = []
    import gzip
    with gzip.open(GFF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene" or f[0] not in CHROMS:
                continue
            m = re.search(r"gene_id=([^;]+)", f[8])
            if m:
                rows.append((m.group(1), f[0], int(f[3]), int(f[4]), f[6]))
    gff = pd.DataFrame(rows, columns=["gene_id", "chrom", "start", "end",
                                      "strand"]).set_index("gene_id")
    common = [g for g in df.columns if g in gff.index]
    print(f"genes: {len(df.columns):,} in matrix, {len(gff):,} in GFF "
          f"(chr1-10), {len(common):,} intersect")
    gsub = gff.loc[common]
    gsel = [df.columns.get_loc(g) for g in common]
    expr = expr[:, gsel, :]
    tss = np.where(gsub["strand"] == "+", gsub["start"], gsub["end"])
    tts = np.where(gsub["strand"] == "+", gsub["end"], gsub["start"])

    # ---- deviation targets ----
    gene_mean = np.nanmean(expr, axis=2, keepdims=True)      # (T, G, 1)
    deviation = (expr - gene_mean).astype(np.float32)

    # ---- splits ----
    rng = np.random.default_rng(42)
    order = rng.permutation(len(lines))
    n_tr = int(round(0.70 * len(lines)))
    n_va = int(round(0.15 * len(lines)))
    line_split = np.empty(len(lines), dtype=object)
    line_split[order[:n_tr]] = "train"
    line_split[order[n_tr:n_tr + n_va]] = "val"
    line_split[order[n_tr + n_va:]] = "test"
    pos_split = np.where(gsub["chrom"] == "9", "val",
                         np.where(gsub["chrom"].isin(["8", "10"]),
                                  "test", "train")).astype(object)

    # ---- write ----
    str_dt = h5py.string_dtype()
    psam_names = [taxa_by_norm[l] for l in lines]
    with h5py.File(OUT / "kremling_dataset.h5", "w") as h5:
        gg = h5.create_group("genes")
        gg.create_dataset("gene_id", data=np.array(common, object), dtype=str_dt)
        gg.create_dataset("chrom", data=gsub["chrom"].to_numpy(object), dtype=str_dt)
        gg.create_dataset("strand", data=gsub["strand"].to_numpy(object), dtype=str_dt)
        gg.create_dataset("start", data=gsub["start"].to_numpy(np.int64))
        gg.create_dataset("end", data=gsub["end"].to_numpy(np.int64))
        gg.create_dataset("tss", data=tss.astype(np.int64))
        gg.create_dataset("tts", data=tts.astype(np.int64))
        gg.create_dataset("pos_split", data=pos_split, dtype=str_dt)
        gl = h5.create_group("lines")
        gl.create_dataset("line_id", data=np.array(lines, object), dtype=str_dt)
        gl.create_dataset("taxa", data=np.array(psam_names, object), dtype=str_dt)
        gl.create_dataset("line_split", data=line_split, dtype=str_dt)
        gl.create_dataset("n_samples", data=cnts.astype(np.int16))  # (T, L)
        h5.create_dataset("tissues", data=np.array(TISSUES, object), dtype=str_dt)
        h5.create_dataset("log2_expr", data=expr)          # (T, G, L), NaN=missing
        h5.create_dataset("deviation", data=deviation)     # (T, G, L)
        h5.attrs["source"] = ("Kremling 2018 Nature 555:520; CyVerse "
                              "Kremling_Nature3RNASeq282_March2018; DESeq2 FPM")
        h5.attrs["target"] = ("deviation = log2(1+FPM) - per-(tissue,gene) "
                              "panel mean; never predict absolute level")

    n_tr_g = (pos_split == "train").sum()
    with open(OUT / "expr_report.md", "w") as fh:
        fh.write("# Kremling expression dataset build report\n\n")
        fh.write(f"- samples used: {len(meta):,} across {len(TISSUES)} tissues\n")
        fh.write(f"- lines matched to genotypes: **{len(lines)}** "
                 f"(splits {n_tr}/{n_va}/{len(lines)-n_tr-n_va} train/val/test, "
                 f"seed 42, random — relatedness-aware split is a follow-up)\n")
        fh.write(f"- expression-only lines dropped: {len(unmatched_lines)}\n")
        fh.write(f"- genes with AGPv3.29 coordinates: **{len(common):,}** "
                 f"(gene split by chrom: {n_tr_g} train / "
                 f"{(pos_split=='val').sum()} val(chr9) / "
                 f"{(pos_split=='test').sum()} test(chr8,10))\n")
        fh.write(f"- per-tissue line coverage: "
                 f"{ {t: int((cnts[i] > 0).sum()) for t, i in tis_ix.items()} }\n")
        fh.write("\nCaveats: genotypes are MAF>0.05 KNN-imputed (rare variants "
                 "in a separate tarball, not built); AGPv3 assembly; random "
                 "line split ignores population structure.\n")
    print(f"wrote {OUT/'kremling_dataset.h5'} and expr_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
build_arabidopsis_phenotypes.py
-------------------------------
Align the AraPheno phenotype tables onto the 1001 Genomes VCF sample list,
restricting to the accessions we actually have genotypes for.

Called by datasets/arabidopsis/Makefile's `pheno` target. See
datasets/arabidopsis/SOURCES.md ("Phenotypes") for provenance: `make phenotypes`
downloads one `phenotype_<id>.csv` per public AraPheno phenotype (keyed by
1001-Genomes accession id) plus `phenotype_list.csv` (the id→name/study index).
The VCF sample IIDs are those same accession ids, so the join is an exact id
match — no key translation.

Unlike soy (one table, 11 dense traits) AraPheno is ~536 sparse traits, each
covering a different subset of accessions, so a single wide matrix would be mostly
empty. We therefore emit three complementary views (all written to --out-dir):

  arabidopsis_pheno_long.csv     tidy/long — one row per measurement, restricted
                                 to genotyped accessions. Columns:
                                 IID, phenotype_id, phenotype_name, study, value.
                                 Keeps replicates (multiple rows per accession).
  arabidopsis_pheno_matrix.csv   wide — rows keyed (IID, rep) in .psam order,
                                 columns = phenotype ids. Replicates are kept on
                                 SEPARATE rows (never averaged): within each
                                 (accession, trait) the k values fill rows 0..k-1,
                                 so an accession spans as many rows as its most-
                                 replicated trait. For a single trait, select its
                                 column and dropna → every replicate with its
                                 (repeated) genotype. Sparse; NaN elsewhere.
  arabidopsis_pheno_coverage.csv one row per phenotype: phenotype_id, name, study,
                                 n_genotyped (distinct genotyped accessions with a
                                 value), n_values (rows incl. replicates), sorted
                                 by n_genotyped desc — use it to pick well-powered
                                 traits (this is the real per-GWAS sample size).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Column names in the AraPheno values.csv files and the id column that matches the
# VCF sample IIDs.
ID_COL = "accession_id"
VALUE_COL = "phenotype_value"
NAME_COL = "phenotype_name"

_FNAME_RE = re.compile(r"phenotype_(\d+)\.csv$")


def read_vcf_sample_order(psam_path: Path) -> list[str]:
    """Return sample IIDs in .psam order (skips the '#IID ...' header line)."""
    iids: list[str] = []
    with open(psam_path) as fh:
        next(fh)  # header (#IID ...)
        for line in fh:
            line = line.strip()
            if line:
                iids.append(line.split("\t")[0])
    return iids


def read_phenotype_index(list_csv: Path) -> dict[str, dict[str, str]]:
    """phenotype_id -> {'name':…, 'study':…} from phenotype_list.csv."""
    idx: dict[str, dict[str, str]] = {}
    if not list_csv.exists():
        return idx
    df = pd.read_csv(list_csv, dtype=str)
    for _, r in df.iterrows():
        pid = str(r.get("phenotype_id", "")).strip()
        if pid:
            idx[pid] = {"name": r.get("name", ""), "study": r.get("study", "")}
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pheno-dir", required=True, type=Path,
                    help="directory of downloaded phenotype_<id>.csv files")
    ap.add_argument("--psam", required=True, type=Path,
                    help="arabidopsis_1001g_final.psam (defines VCF sample set/order)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="directory to write the aligned phenotype CSVs into")
    args = ap.parse_args()

    iids = read_vcf_sample_order(args.psam)
    geno = set(iids)
    index = read_phenotype_index(args.pheno_dir / "phenotype_list.csv")

    value_files = sorted(
        p for p in args.pheno_dir.glob("phenotype_*.csv")
        if _FNAME_RE.search(p.name)  # excludes phenotype_list.csv
    )
    if not value_files:
        raise SystemExit(f"no phenotype_<id>.csv files found in {args.pheno_dir} "
                         "— run `make phenotypes` first")

    long_rows: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for fp in value_files:
        pid = _FNAME_RE.search(fp.name).group(1)
        try:
            df = pd.read_csv(fp, dtype=str)
        except Exception as e:  # empty/corrupt file — skip, but report
            print(f"[build_arabidopsis_phenotypes] WARN: could not read {fp.name}: {e}")
            continue
        if ID_COL not in df.columns or VALUE_COL not in df.columns:
            print(f"[build_arabidopsis_phenotypes] WARN: {fp.name} lacks expected columns; skipped")
            continue

        df = df[df[ID_COL].isin(geno)].copy()
        df["value"] = pd.to_numeric(df[VALUE_COL], errors="coerce")
        df = df.dropna(subset=["value"])

        meta = index.get(pid, {})
        name = (df[NAME_COL].iloc[0] if NAME_COL in df.columns and len(df)
                else meta.get("name", ""))
        study = meta.get("study", "")

        coverage.append({
            "phenotype_id": pid,
            "phenotype_name": name,
            "study": study,
            "n_genotyped": df[ID_COL].nunique(),
            "n_values": len(df),
        })
        if len(df):
            long_rows.append(pd.DataFrame({
                "IID": df[ID_COL].values,
                "phenotype_id": pid,
                "phenotype_name": name,
                "study": study,
                "value": df["value"].values,
            }))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── long (tidy) ───────────────────────────────────────────────────────────
    long = (pd.concat(long_rows, ignore_index=True) if long_rows
            else pd.DataFrame(columns=["IID", "phenotype_id", "phenotype_name",
                                       "study", "value"]))
    long_path = args.out_dir / "arabidopsis_pheno_long.csv"
    long.to_csv(long_path, index=False)

    # ── wide matrix (VCF order × phenotype id; replicates kept separate) ───────
    # Not averaged: within each (accession, trait) the k values are laid out down k
    # rows (rep 0..k-1). Rows are keyed (IID, rep); an accession gets as many rows
    # as its most-replicated trait. Genotype joins on IID (repeated across reps).
    if len(long):
        lw = long.copy()
        lw["rep"] = lw.groupby(["IID", "phenotype_id"]).cumcount()
        # (IID, rep, phenotype_id) is unique by construction → aggfunc never merges.
        wide = lw.pivot_table(index=["IID", "rep"], columns="phenotype_id",
                              values="value", aggfunc="first")
        maxrep = lw.groupby("IID")["rep"].max()   # 0-based; missing → treated as 0
    else:
        wide = pd.DataFrame()
        maxrep = pd.Series(dtype=int)
    # Full row index: every VCF sample in .psam order, each with rows 0..maxrep
    # (unphenotyped accessions still get a single all-NaN row, keeping the panel).
    full_idx = pd.MultiIndex.from_tuples(
        [(iid, r) for iid in iids for r in range(int(maxrep.get(iid, 0)) + 1)],
        names=["IID", "rep"])
    wide = wide.reindex(full_idx)
    matrix_path = args.out_dir / "arabidopsis_pheno_matrix.csv"
    wide.to_csv(matrix_path)

    # ── coverage guide ────────────────────────────────────────────────────────
    cov = (pd.DataFrame(coverage)
           .sort_values(["n_genotyped", "n_values"], ascending=False)
           if coverage else
           pd.DataFrame(columns=["phenotype_id", "phenotype_name", "study",
                                 "n_genotyped", "n_values"]))
    cov_path = args.out_dir / "arabidopsis_pheno_coverage.csv"
    cov.to_csv(cov_path, index=False)

    n_pheno = len(cov)
    n_pheno_used = int((cov["n_genotyped"] > 0).sum()) if len(cov) else 0
    covered_acc = long["IID"].nunique() if len(long) else 0
    print(f"[build_arabidopsis_phenotypes] VCF samples:            {len(iids)}")
    print(f"[build_arabidopsis_phenotypes] phenotypes scanned:     {n_pheno}")
    print(f"[build_arabidopsis_phenotypes] phenotypes w/ genotyped data: {n_pheno_used}")
    print(f"[build_arabidopsis_phenotypes] genotyped accs w/ >=1 trait:  {covered_acc}")
    print(f"[build_arabidopsis_phenotypes] measurements (long rows):     {len(long)}")
    print(f"[build_arabidopsis_phenotypes] wrote {long_path}")
    print(f"[build_arabidopsis_phenotypes] wrote {matrix_path} ({wide.shape[0]} x {wide.shape[1]})")
    print(f"[build_arabidopsis_phenotypes] wrote {cov_path}")


if __name__ == "__main__":
    main()

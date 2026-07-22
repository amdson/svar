#!/usr/bin/env python
"""
build_soy_phenotypes.py
-----------------------
Join the SoyDNGP phenotype table (data.csv, keyed by USDA accession id) onto the
soy SoySNP50K VCF sample list, producing phenotype tables aligned to the VCF.

Called by datasets/soy/Makefile's `pheno` target. See datasets/soy/SOURCES.md
("Phenotypes") for provenance: the 11 traits come from the SoyDNGP project repo
(github.com/IndigoFloyd/SoybeanWebsite), which is the source GP-WAITER's
"soybean14460" panel was built from. The accessions with complete phenotypes that
also appear in our VCF reconstruct that 14,460-accession panel on top of our own
PI-keyed, coordinate-carrying genotypes (no de-anonymisation needed).

Outputs (written to --out-dir):
  soy_pheno_aligned.csv   one row per VCF sample (IID order); 11 trait columns
                          plus `matched`/`complete` flags. NaN where unmatched or
                          the source has no value.
  soy_pheno_complete.csv  complete-case subset (all 11 traits present) — use as Y.

Exact accession-id matching is used (this reproduces the published ~14,460 panel).
A handful of extra samples differ only by an id suffix (e.g. PI594471B vs
PI594471) and are intentionally left unmatched to stay faithful to that panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# The 11 GP-WAITER / SoyDNGP traits, as named in data.csv.
TRAITS = ["protein", "oil", "Linoleic", "Linolenic", "R1", "R8",
          "Hgt", "Ldg", "SQ", "SdWgt", "Yield"]

# Column in data.csv holding the USDA accession id (matches VCF sample IIDs).
ID_COL = "acid"


def read_vcf_sample_order(psam_path: Path) -> list[str]:
    """Return sample IIDs in .psam order (skips the '#IID ...' header line)."""
    iids: list[str] = []
    with open(psam_path) as fh:
        next(fh)  # header
        for line in fh:
            line = line.strip()
            if line:
                iids.append(line.split("\t")[0])
    return iids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-csv", required=True, type=Path,
                    help="SoyDNGP phenotype table (data.csv), keyed by accession id")
    ap.add_argument("--psam", required=True, type=Path,
                    help="soysnp50k_a2_final.psam (defines VCF sample order)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="directory to write the aligned phenotype CSVs into")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv, dtype=str)
    missing = [c for c in [ID_COL, *TRAITS] if c not in df.columns]
    if missing:
        raise SystemExit(f"data.csv is missing expected columns: {missing}")

    phe = df[[ID_COL, *TRAITS]].copy()
    for t in TRAITS:
        phe[t] = pd.to_numeric(phe[t], errors="coerce")
    phe = phe.drop_duplicates(ID_COL).set_index(ID_COL)

    iids = read_vcf_sample_order(args.psam)
    aligned = phe.reindex(iids)
    aligned.index.name = "IID"
    aligned.insert(0, "matched", aligned[TRAITS].notna().any(axis=1))
    aligned.insert(1, "complete", aligned[TRAITS].notna().all(axis=1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.out_dir / "soy_pheno_aligned.csv"
    complete_path = args.out_dir / "soy_pheno_complete.csv"
    aligned.to_csv(aligned_path)
    aligned[aligned["complete"]][TRAITS].to_csv(complete_path)

    n_total = len(aligned)
    n_match = int(aligned["matched"].sum())
    n_complete = int(aligned["complete"].sum())
    print(f"[build_soy_phenotypes] VCF samples:      {n_total}")
    print(f"[build_soy_phenotypes] matched (>=1):    {n_match}")
    print(f"[build_soy_phenotypes] complete (all 11):{n_complete}  (the soybean14460 panel)")
    print(f"[build_soy_phenotypes] wrote {aligned_path}")
    print(f"[build_soy_phenotypes] wrote {complete_path} ({n_complete} x {len(TRAITS)})")


if __name__ == "__main__":
    main()

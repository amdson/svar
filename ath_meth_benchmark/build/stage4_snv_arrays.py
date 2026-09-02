#!/usr/bin/env python3
"""Stage 4 (spec §5): per-accession sorted SNV arrays from the UNFILTERED
biallelic-SNP pgen (built by stage4_snv_arrays.sbatch from
arabidopsis_1001g_biallelic_snps.vcf.gz — NOT the MAF/geno-filtered `_final`
set, which drops rare alleles that are exactly the §3b signal).

For every sample in the pgen, collect variants with hom-ALT genotype into
(pos, alt_base) arrays in TAIR10 coordinates. Heterozygous and missing calls
are left as reference (accessions are inbred; rates are logged per accession).

Output: snv_arrays.h5
    /acc/<ecotype_id>/chrom  uint8   (1..5)
    /acc/<ecotype_id>/pos    int32   (1-based TAIR10)
    /acc/<ecotype_id>/alt    uint8   (ASCII base)
    /acc/<ecotype_id>.attrs: n_alt, n_het, n_missing
    /meta.attrs: source, v1 scope note
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pgenlib

CHUNK = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfile", required=True, help="pgen/pvar/psam prefix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pvar = pd.read_csv(a.pfile + ".pvar", sep="\t", comment="#", header=None,
                       usecols=[0, 1, 3, 4], names=["chrom", "pos", "ref", "alt"],
                       dtype={"chrom": str, "pos": np.int32, "ref": str, "alt": str})
    psam = pd.read_csv(a.pfile + ".psam", sep="\t")
    samples = psam.iloc[:, 0].astype(str).tolist()
    n_var, n_samp = len(pvar), len(samples)
    print(f"{n_var:,} variants x {n_samp} samples")

    ok = (pvar["ref"].str.len() == 1) & (pvar["alt"].str.len() == 1) & \
         pvar["alt"].isin(list("ACGT")) & pvar["chrom"].isin(list("12345"))
    keep_var = ok.to_numpy()
    print(f"single-base ACGT biallelic on chr1-5: {int(keep_var.sum()):,} "
          f"(dropped {int((~keep_var).sum()):,})")

    chrom_arr = pvar["chrom"].to_numpy().astype("S1").view(np.uint8)  # ASCII '1'..'5'
    pos_arr = pvar["pos"].to_numpy(np.int32)
    alt_arr = pvar["alt"].to_numpy().astype("S1").view(np.uint8)

    per_var = [[] for _ in range(n_samp)]
    n_het = np.zeros(n_samp, dtype=np.int64)
    n_miss = np.zeros(n_samp, dtype=np.int64)

    rdr = pgenlib.PgenReader(a.pfile.encode() + b".pgen")
    buf = np.empty((CHUNK, n_samp), dtype=np.int8)
    for start in range(0, n_var, CHUNK):
        stop = min(start + CHUNK, n_var)
        b = buf[: stop - start]
        rdr.read_range(start, stop, b)
        kv = keep_var[start:stop]
        b = b[kv]
        vids = np.flatnonzero(keep_var[start:stop]) + start

        n_het += (b == 1).sum(axis=0)
        n_miss += (b == -9).sum(axis=0)

        vi, sj = np.nonzero(b == 2)
        order = np.argsort(sj, kind="stable")
        vi, sj = vi[order], sj[order]
        bounds = np.searchsorted(sj, np.arange(n_samp + 1))
        for s in range(n_samp):
            lo, hi = bounds[s], bounds[s + 1]
            if hi > lo:
                gv = vids[vi[lo:hi]]
                per_var[s].append(gv)
        if (start // CHUNK) % 10 == 0:
            print(f"  {stop:,}/{n_var:,}", flush=True)

    with h5py.File(a.out, "w") as h5:
        g = h5.create_group("acc")
        for s, name in enumerate(samples):
            v = np.concatenate(per_var[s]) if per_var[s] else np.empty(0, np.int64)
            grp = g.create_group(name)
            grp.create_dataset("chrom", data=chrom_arr[v] - ord("0"), compression="gzip")
            grp.create_dataset("pos", data=pos_arr[v], compression="gzip")
            grp.create_dataset("alt", data=alt_arr[v], compression="gzip")
            grp.attrs["n_alt"] = len(v)
            grp.attrs["n_het_skipped"] = int(n_het[s])
            grp.attrs["n_missing_skipped"] = int(n_miss[s])
        h5.attrs["source"] = "1001genomes v3.1 snp-short-indel_only, biallelic SNVs, hom-ALT only"
        h5.attrs["scope"] = ("v1: SNVs only; no indels/SVs; het and missing left as "
                             "TAIR10 reference; coordinates are TAIR10 1-based")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

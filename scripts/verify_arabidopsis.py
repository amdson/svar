#!/usr/bin/env python
"""
verify_arabidopsis.py
---------------------
Acceptance checks for a freshly rebuilt arabidopsis dataset, run after
`datasets/arabidopsis/Makefile` finishes:

    python scripts/verify_arabidopsis.py

`make check` already proves REF concordance against the genome. This proves the
rest of the contract the *code* depends on — the things that silently produce an
empty or wrong dataset rather than an error:

  1. every path in the training registry resolves;
  2. the panel is the expected 1,135 accessions, on chromosomes 1..5;
  3. the FASTA exposes integer chromosome keys matching the VCF CHROM — the
     requirement called out in SOURCES.md ("crop_embed's loader requires integer
     chromosome names"), where a mismatch yields zero usable windows;
  4. pysam can actually fetch from the (bgzipped) VCF — needs the tabix index;
  5. crop_embed builds real windows end-to-end, on a small slice so this stays a
     minutes-not-hours check;
  6. the phenotype table is IID-unique and joins onto the genotyped samples.

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_SAMPLES = 1135          # Alonso-Blanco et al. 2016, 1001 Genomes v3.1
EXPECTED_CHROMS = {1, 2, 3, 4, 5}
SLICE_REGION = "1:1-1000000"     # small end-to-end slice for the window check
SLICE_SAMPLES = 20

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def conda_run(*args: str) -> subprocess.CompletedProcess:
    """Invoke a tool from the svar env without assuming it's already activated."""
    return subprocess.run(["conda", "run", "-n", "svar", *args],
                          capture_output=True, text=True)


def main() -> int:
    from training.common.datasets import get_dataset

    spec = get_dataset("arabidopsis")
    print(f"\narabidopsis dataset verification\n{'=' * 60}")

    # ── 1. registry paths ────────────────────────────────────────────────────
    print("\n1. registry paths")
    paths = {"pgen": spec.pgen_prefix + ".pgen", "pvar": spec.pgen_prefix + ".pvar",
             "psam": spec.psam, "fasta": spec.fasta_path,
             "vcf": spec.vcf_path, "pheno": spec.pheno_csv}
    all_present = True
    for label, p in paths.items():
        all_present &= check(f"{label} exists", Path(p).exists(), p)
    if not all_present:
        print("\nmissing inputs — run: cd datasets/arabidopsis && make")
        return 1
    # The Makefile emits a bgzipped VCF (+ .tbi), but an older build may have left
    # a plain one, and `variant_ll` prefers plain anyway — that is the only form the
    # polars reader takes. Only a bgzipped VCF needs the index.
    bgzipped = spec.vcf_path.endswith((".gz", ".bgz"))
    if bgzipped:
        check("VCF tabix index exists", Path(spec.vcf_path + ".tbi").exists(),
              "pysam .fetch() needs it on a bgzipped VCF")
    else:
        check("VCF is plain text (no tabix index needed)", True,
              "polars path reads it directly; bcftools streams with -t")

    # ── 2. panel shape ───────────────────────────────────────────────────────
    print("\n2. panel shape")
    samples = spec.samples()
    check(f"{EXPECTED_SAMPLES} accessions", len(samples) == EXPECTED_SAMPLES,
          f"got {len(samples)}")
    chroms, n_variants = set(), 0
    with open(spec.pgen_prefix + ".pvar") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_variants += 1
            chroms.add(line.split("\t", 1)[0])
    check("variants present", n_variants > 0, f"{n_variants:,} variants")
    int_chroms = {int(c) for c in chroms if c.isdigit()}
    check("chromosomes are exactly 1..5", int_chroms == EXPECTED_CHROMS,
          f"got {sorted(chroms)}")

    # ── 3. FASTA keys line up with VCF CHROM ─────────────────────────────────
    print("\n3. reference genome keys")
    from pyfaidx import Fasta
    fasta = Fasta(spec.fasta_path)
    fasta_int_keys = {int(k) for k in fasta.keys() if k.isdigit()}
    check("FASTA exposes integer keys for every VCF chromosome",
          int_chroms <= fasta_int_keys,
          f"FASTA digit keys {sorted(fasta_int_keys)}; VCF {sorted(int_chroms)}")

    # ── 4./5. pysam fetch + real windows, on a small slice ───────────────────
    print(f"\n4. end-to-end window build ({SLICE_REGION}, {SLICE_SAMPLES} accessions)")
    with tempfile.TemporaryDirectory() as td:
        slice_vcf = str(Path(td) / "slice.vcf.gz")
        sub = ",".join(samples[:SLICE_SAMPLES])
        # -r seeks via the tabix index; -t streams and filters. A plain VCF has no
        # index, so it has to take the streaming form (slower, but correct).
        region_flag = "-r" if bgzipped else "-t"
        r = conda_run("bcftools", "view", region_flag, SLICE_REGION, "-s", sub,
                      "-Oz", "-o", slice_vcf, spec.vcf_path)
        if not check(f"bcftools can subset the region ({region_flag})",
                     r.returncode == 0, (r.stderr or "").strip()[-200:]):
            return 1
        conda_run("bcftools", "index", "-t", slice_vcf)

        from crop_embed.data.vcf import load_snps_from_vcf
        from crop_embed.partitioner import SNPWindowPartitioner
        from crop_embed.dataset import UniqueWindowDataset

        snps_by_chrom, slice_samples = load_snps_from_vcf(slice_vcf)
        n_snps = sum(len(v) for v in snps_by_chrom.values())
        check("loader parsed SNPs with integer chrom keys", n_snps > 0,
              f"{n_snps:,} SNPs over chroms {sorted(snps_by_chrom)}")

        part = SNPWindowPartitioner(snps_by_chrom, half_window=250, buffer=50)
        ds = UniqueWindowDataset(slice_vcf, spec.fasta_path, part)
        check("windows built", len(ds) > 0, f"{len(ds):,} unique windows")

        item = ds[0]
        seq = item["sequence"] if isinstance(item, dict) else item
        seq = seq if isinstance(seq, str) else str(seq)
        check("window sequence is 500 nt of ACGTN", len(seq) == 500,
              f"len={len(seq)}, starts {seq[:30]!r}")
        check("sequence is not all-N (reference actually read)",
              set(seq.upper()) - {"N"} != set(), f"distinct bases {sorted(set(seq.upper()))[:6]}")

    # ── 6. phenotypes ────────────────────────────────────────────────────────
    print("\n5. phenotypes")
    import pandas as pd
    ph = pd.read_csv(spec.pheno_csv)
    check("has IID column", "IID" in ph.columns, f"cols: {list(ph.columns)[:5]}")
    check("IID is unique (features.py reindexes on it)", ph["IID"].is_unique,
          f"{len(ph)} rows, {ph['IID'].nunique()} unique")
    traits = spec.resolved_trait_cols()
    check("trait columns present", len(traits) > 0, f"{len(traits)} traits")
    overlap = len(set(ph["IID"].astype(str)) & set(map(str, samples)))
    check("phenotype IIDs join onto genotyped accessions", overlap > 0,
          f"{overlap}/{len(samples)} accessions")
    if "matched" in ph.columns:
        check("accessions with >=1 phenotype", int(ph["matched"].sum()) > 0,
              f"{int(ph['matched'].sum())} of {len(ph)}")

    # ── summary ──────────────────────────────────────────────────────────────
    failed = [n for ok, n, _ in _results if not ok]
    print(f"\n{'=' * 60}")
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

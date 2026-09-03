"""
TASSEL hapmap text (hmp321 style) -> single bgzipped VCF, streaming.

Built for datasets/maize_kremling: the RNAset genotype tarballs are per-chromosome
`*.hmp.txt` files with 11 metadata columns then one single-char IUPAC genotype
column per taxon (N = missing, R/Y/S/W/K/M = het). REF is provisionally the
hapmap `alleles` field's first allele; plink2 --ref-from-fa force fixes strand
of truth downstream, so this script only needs a consistent 0/1 encoding.

Speed: one `str.translate` pass per row over the genotype tail (tabs pass
through), mapping a1->0/0, a2->1/1, the {a1,a2} IUPAC het->0/1, everything
else->./.  — no per-sample Python loop.

    python scripts/hmp_to_vcf.py --out all.vcf.gz c1.tar.gz c2.tar.gz ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile

IUPAC_HET = {frozenset("AG"): "R", frozenset("CT"): "Y", frozenset("CG"): "S",
             frozenset("AT"): "W", frozenset("GT"): "K", frozenset("AC"): "M"}
ALL_CODES = "ACGTRYSWKMN"


def open_member(tar_path: str):
    tf = tarfile.open(tar_path, "r:gz")
    members = [m for m in tf.getmembers() if m.name.endswith(".hmp.txt")]
    if len(members) != 1:
        raise SystemExit(f"{tar_path}: expected exactly one .hmp.txt, "
                         f"got {[m.name for m in tf.getmembers()]}")
    return tf, tf.extractfile(members[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tars", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Write to a temp name, rename on success — a failed run must not leave a
    # partial output that make would treat as up to date.
    tmp_out = args.out + ".tmp"
    # bgzip if available (tabix-compatible), else plain gzip — plink2 reads both.
    bgzip = None
    try:
        bgzip = subprocess.Popen(["bgzip", "-c", "-@", "4"],
                                 stdin=subprocess.PIPE,
                                 stdout=open(tmp_out, "wb"))
        w = bgzip.stdin
    except FileNotFoundError:
        import gzip
        print("bgzip not found; falling back to Python gzip (level 1)")
        w = gzip.open(tmp_out, "wb", compresslevel=1)

    taxa = None
    n_rows = n_skip = het_calls = total_calls = 0
    for tar_path in args.tars:
        tf, fh = open_member(tar_path)
        header = fh.readline().decode().rstrip("\n").split("\t")
        row_taxa = header[11:]
        if taxa is None:
            taxa = row_taxa
            w.write(b"##fileformat=VCFv4.2\n")
            w.write(b'##INFO=<ID=PR,Number=0,Type=Flag,Description="hapmap allele1 as provisional REF">\n')
            w.write(("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                     + "\t".join(taxa) + "\n").encode())
        elif row_taxa != taxa:
            raise SystemExit(f"{tar_path}: taxa differ from first file")

        for raw in fh:
            line = raw.decode()
            head, tail = line.split("\t", 11)[:11], line.split("\t", 11)[11]
            rsid, alleles, chrom, pos = head[0], head[1], head[2], head[3]
            try:
                a1, a2 = alleles.split("/")
            except ValueError:
                n_skip += 1
                continue
            if a1 not in "ACGT" or a2 not in "ACGT" or a1 == a2:
                n_skip += 1
                continue
            het = IUPAC_HET[frozenset((a1, a2))]
            tbl = {ord(c): "./." for c in ALL_CODES}
            tbl[ord(a1)] = "0/0"
            tbl[ord(a2)] = "1/1"
            tbl[ord(het)] = "0/1"
            gts = tail.rstrip("\n").translate(tbl)
            if n_rows % 997 == 0:  # cheap sampled het/miss accounting
                het_calls += gts.count("0/1")
                total_calls += len(taxa)
            w.write((f"{chrom}\t{pos}\t{rsid}\t{a1}\t{a2}\t.\t.\tPR\tGT\t"
                     f"{gts}\n").encode())
            n_rows += 1
        tf.close()
        print(f"{tar_path}: done ({n_rows:,} rows so far, {n_skip:,} skipped)",
              flush=True)

    w.close()
    if bgzip is not None and bgzip.wait() != 0:
        raise SystemExit("bgzip failed")
    import os
    os.replace(tmp_out, args.out)
    het_rate = het_calls / max(total_calls, 1)
    print(f"wrote {args.out}: {n_rows:,} variants x {len(taxa)} taxa; "
          f"{n_skip:,} non-SNP rows skipped; sampled het rate {het_rate:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

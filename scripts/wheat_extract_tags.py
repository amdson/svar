#!/usr/bin/env python
"""
wheat_extract_tags.py — stage 1 of the wheat DArTSeq build.

Read the DArTSeq genotype CSV (inside the access-gated MARKERS zip) and write the
per-marker 69bp tag sequences as FASTA + a metadata TSV. The FASTA is what gets
aligned to the wheat reference to recover genomic coordinates.

DArT CSV layout: ~7 plate/well annotation rows, then a header row starting
`AlleleID,CloneID,AlleleSequence,SNP,SnpPosition,...` (NMETA metadata columns),
then one data row per marker. See datasets/wheat/SOURCES.md.
"""
import argparse, zipfile, io, os

NMETA = 17  # AlleleID..RepAvg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markers-zip", required=True, help="MEXICO AND IRANIAN -- MARKERS zip (.zip after gunzip)")
    ap.add_argument("--member", default="Iranian_Samples.csv", help="CSV member inside the zip")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fa_path = os.path.join(a.out_dir, "tags.fasta")
    meta_path = os.path.join(a.out_dir, "tags_meta.tsv")

    zf = zipfile.ZipFile(a.markers_zip)
    n = 0
    with zf.open(a.member) as fh, open(fa_path, "w") as fa, open(meta_path, "w") as meta:
        txt = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        meta.write("fid\tAlleleID\tCloneID\tSnpPosition\tSNP\ttag_len\n")
        header_seen = False
        for line in txt:
            f = line.rstrip("\n").split(",")
            if not header_seen:
                if f and f[0] == "AlleleID":
                    header_seen = True
                continue
            allele_id, clone_id, seq, snp, snp_pos = f[0], f[1], f[2], f[3], f[4]
            if not seq or set(seq) - set("ACGTNacgtn"):
                continue
            fid = f"m{n}"
            fa.write(f">{fid}\n{seq}\n")
            meta.write(f"{fid}\t{allele_id}\t{clone_id}\t{snp_pos}\t{snp}\t{len(seq)}\n")
            n += 1
    print(f"[wheat_extract_tags] wrote {n} tags -> {fa_path}")

if __name__ == "__main__":
    main()

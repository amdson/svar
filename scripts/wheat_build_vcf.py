#!/usr/bin/env python
"""
wheat_build_vcf.py — stage 4 of the wheat DArTSeq build.

Emit a VCF from the DArTSeq genotypes + aligned marker coordinates:
  - keep only markers in marker_coords.tsv;
  - REF = genome base (cs_allele), ALT = the other allele;
  - map DArT calls (0=hom-ref, 1=hom-ALT, 2=het, -=missing) to VCF GT, flipping
    hom genotypes where the genome base is the DArT alt (so REF is always genomic);
  - rename chromosomes 1A..7D,Un -> integers 1..22 so the output loads through
    crop_embed's integer-chrom loader exactly like soy/rice (also writes chrom_map.tsv);
  - samples keyed by CIMMYT GID (header row 1), duplicate GIDs collapsed to first.
"""
import argparse, zipfile, io

NMETA = 17
CHROM_ORDER = [f"{n}{s}" for n in range(1, 8) for s in "ABD"] + ["Un"]
CHROM_MAP = {c: str(i + 1) for i, c in enumerate(CHROM_ORDER)}

def gt_noflip(c): return {"0":"0/0","1":"1/1","2":"0/1","-":"./.","":"./."}.get(c, "./.")
def gt_flip(c):   return {"0":"1/1","1":"0/0","2":"0/1","-":"./.","":"./."}.get(c, "./.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheat-dir", required=True)
    ap.add_argument("--markers-zip", required=True)
    ap.add_argument("--member", default="Iranian_Samples.csv")
    a = ap.parse_args()

    import csv
    coords = {}
    with open(f"{a.wheat_dir}/marker_coords.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            coords[row["AlleleID"]] = row
    with open(f"{a.wheat_dir}/chrom_map.tsv", "w") as fh:
        for c in CHROM_ORDER:
            fh.write(f"{c}\t{CHROM_MAP[c]}\n")

    zf = zipfile.ZipFile(a.markers_zip)
    records = []
    with zf.open(a.member) as fh:
        txt = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        gid_row = None; header_seen = False; keep_cols = None; sample_names = None
        for i, line in enumerate(txt):
            f = line.rstrip("\n").split(",")
            if i == 0:
                gid_row = f
            if not header_seen:
                if f and f[0] == "AlleleID":
                    header_seen = True
                    seen = set(); keep_cols = []; sample_names = []
                    for j in range(NMETA, len(f)):
                        gid = gid_row[j].strip()
                        if gid and gid.isdigit() and gid not in seen:
                            seen.add(gid); keep_cols.append(j); sample_names.append(gid)
                continue
            c = coords.get(f[0])
            if c is None:
                continue
            flip = (c["cs_allele"] == c["alt_fwd"])
            conv = gt_flip if flip else gt_noflip
            ref = c["cs_allele"]; alt = c["ref_fwd"] if flip else c["alt_fwd"]
            ci = CHROM_MAP[c["chrom"]]
            gts = "\t".join(conv(f[j]) for j in keep_cols)
            records.append((int(ci), int(c["pos1"]),
                            f"{ci}\t{c['pos1']}\t{c['CloneID']}\t{ref}\t{alt}\t.\t.\t.\tGT\t{gts}"))
    records.sort(key=lambda x: (x[0], x[1]))
    vcf = f"{a.wheat_dir}/wheat_dartseq.vcf"
    with open(vcf, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        for c in CHROM_ORDER:
            out.write(f"##contig=<ID={CHROM_MAP[c]}>\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names) + "\n")
        for _, _, l in records:
            out.write(l + "\n")
    print(f"[build_vcf] {len(records)} markers x {len(sample_names)} samples -> {vcf}")

if __name__ == "__main__":
    main()

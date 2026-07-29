#!/usr/bin/env python
"""
wheat_build_pheno.py — stage 5 of the wheat DArTSeq build.

Join the Iranian phenotype spreadsheets (keyed by CIMMYT GID) onto the wheat VCF
sample order, soy-style. Writes:
  wheat_pheno_aligned.csv   one row per VCF sample (GID) + matched/complete flags
  wheat_pheno_complete.csv  complete-case subset (all traits) -> use as Y
"""
import argparse, glob, re, csv, openpyxl

# per-file glob -> {out_trait: column_index}
LAYOUT = [
    ("*DTH*DTM*", {"DTM_heat":1,"DTH_heat":2,"DTM_drought":3,"DTH_drought":4}),
    ("*PHT*",     {"PHT_drought":1}),
    ("*QUALITY*", {"tkw":1,"testw":2,"klength":3,"kwidth":4,"hardness":5,"protein":6,"SDS":7}),
]
TRAITS = ["DTH_heat","DTM_heat","DTH_drought","DTM_drought","PHT_drought",
          "tkw","testw","klength","kwidth","hardness","protein","SDS"]

def is_gid(x):
    return x is not None and re.fullmatch(r"\d{4,7}", str(x).strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheat-dir", required=True)
    ap.add_argument("--pheno-dir", required=True, help="extracted 'PHENOTYPIC DATA IRANIAN' dir")
    a = ap.parse_args()

    pheno = {}
    for pat, cols in LAYOUT:
        m = glob.glob(f"{a.pheno_dir}/{pat}.xlsx")
        if not m:
            print(f"[pheno] WARN no file for {pat}"); continue
        wb = openpyxl.load_workbook(m[0], read_only=True, data_only=True)
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                if not row or not is_gid(row[0]):
                    continue
                d = pheno.setdefault(str(row[0]).strip(), {})
                for t, ci in cols.items():
                    if ci < len(row) and row[ci] is not None and str(row[ci]).strip() != "":
                        try: d[t] = float(row[ci])
                        except ValueError: pass
        wb.close()

    with open(f"{a.wheat_dir}/wheat_dartseq.vcf") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                sample_gids = line.rstrip("\n").split("\t")[9:]; break

    n_match = n_complete = 0
    with open(f"{a.wheat_dir}/wheat_pheno_aligned.csv", "w", newline="") as fa, \
         open(f"{a.wheat_dir}/wheat_pheno_complete.csv", "w", newline="") as fc:
        wa = csv.writer(fa); wc = csv.writer(fc)
        wa.writerow(["GID","matched","complete"]+TRAITS); wc.writerow(["GID"]+TRAITS)
        for gid in sample_gids:
            d = pheno.get(gid, {}); vals = [d.get(t) for t in TRAITS]
            matched = any(v is not None for v in vals); complete = all(v is not None for v in vals)
            n_match += matched; n_complete += complete
            wa.writerow([gid, matched, complete] + ["" if v is None else v for v in vals])
            if complete: wc.writerow([gid] + vals)
    print(f"[pheno] {len(sample_gids)} samples; matched {n_match}; complete {n_complete}")

if __name__ == "__main__":
    main()

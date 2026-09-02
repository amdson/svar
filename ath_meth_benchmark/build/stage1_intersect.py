#!/usr/bin/env python3
"""Stage 1 (spec §2, first checkpoint): accession intersection between
GSE43857 sample metadata and the 1001G v3.1 VCF sample names.

Emits:
  meta/gse43857_samples.tsv   — gsm, title, ecotype_id, tissue, suppl_url
  meta/accession_intersection.tsv — ecotype ids present in both sources
Prints the intersection report.
"""
import gzip
import re
from pathlib import Path

DATA = Path("/90daydata/small_grains/andrew.dickson/datasets/arabidopsis")
META = DATA / "methylation" / "meta"
SOFT = META / "GSE43857_family.soft.gz"
PSAM = DATA / "arabidopsis_1001g_final.psam"  # same 1135 samples as the unfiltered biallelic VCF

# ---- parse SOFT ----
samples = []
cur = None
with gzip.open(SOFT, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith("^SAMPLE = "):
            cur = {"gsm": line.split(" = ")[1]}
            samples.append(cur)
        elif cur is not None:
            if line.startswith("!Sample_title = "):
                cur["title"] = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_characteristics_ch1 = ecotype id: "):
                cur["ecotype_id"] = line.split("ecotype id: ", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1 = tissue: "):
                cur["tissue"] = line.split("tissue: ", 1)[1].strip()
            elif line.startswith("!Sample_supplementary_file_1 = "):
                cur["suppl_url"] = line.split(" = ", 1)[1]

print(f"GSE43857 samples parsed: {len(samples)}")
missing_eco = [s for s in samples if "ecotype_id" not in s]
if missing_eco:
    print(f"  WARNING: {len(missing_eco)} samples lack an ecotype id:")
    for s in missing_eco[:10]:
        print(f"    {s['gsm']} {s.get('title')}")
missing_url = [s for s in samples if "suppl_url" not in s]
if missing_url:
    print(f"  WARNING: {len(missing_url)} samples lack a supplementary file URL")

# tissue breakdown
from collections import Counter
print("  tissue:", dict(Counter(s.get("tissue", "?") for s in samples)))

# ecotype ids must look numeric (VCF IIDs are numeric)
bad = [s for s in samples if not re.fullmatch(r"\d+", s.get("ecotype_id", ""))]
if bad:
    print(f"  WARNING: {len(bad)} non-numeric ecotype ids, e.g. {[s.get('ecotype_id') for s in bad[:5]]}")

# duplicate ecotype ids (e.g. leaf + bud from same accession)
eco_counts = Counter(s["ecotype_id"] for s in samples if "ecotype_id" in s)
dups = {k: v for k, v in eco_counts.items() if v > 1}
print(f"  distinct ecotype ids: {len(eco_counts)}; ids with >1 sample: {len(dups)}")

# ---- VCF sample names ----
vcf_ids = set()
with open(PSAM) as fh:
    next(fh)
    for line in fh:
        vcf_ids.add(line.split()[0].strip())
print(f"VCF samples: {len(vcf_ids)}")

geo_ids = set(eco_counts)
inter = geo_ids & vcf_ids
print(f"\nINTERSECTION: {len(inter)} accessions "
      f"(GEO-only: {len(geo_ids - vcf_ids)}, VCF-only: {len(vcf_ids - geo_ids)})")
if geo_ids - vcf_ids:
    ex = sorted(geo_ids - vcf_ids, key=int)[:15]
    print(f"  GEO-only examples: {ex}")

# ---- write outputs ----
with open(META / "gse43857_samples.tsv", "w") as out:
    out.write("gsm\ttitle\tecotype_id\ttissue\tin_vcf\tsuppl_url\n")
    for s in samples:
        eco = s.get("ecotype_id", "")
        out.write("\t".join([
            s["gsm"], s.get("title", ""), eco, s.get("tissue", ""),
            "1" if eco in vcf_ids else "0", s.get("suppl_url", ""),
        ]) + "\n")
with open(META / "accession_intersection.tsv", "w") as out:
    out.write("ecotype_id\n")
    for eco in sorted(inter, key=int):
        out.write(eco + "\n")
print(f"\nWrote {META/'gse43857_samples.tsv'} and {META/'accession_intersection.tsv'}")

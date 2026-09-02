#!/usr/bin/env python3
"""Merge GSE43857 sample metadata with 1001G accession metadata into the
benchmark accession table (spec §4 `accessions` + §6 accession-axis splits).

One row per intersected ecotype. Where an ecotype has >1 GSM sample,
prefer tissue == 'leaf'; the unused GSM is recorded in alt_gsm.
"""
import csv
from collections import defaultdict
from pathlib import Path

META = Path("/90daydata/small_grains/andrew.dickson/datasets/arabidopsis/methylation/meta")

# 1001G accession metadata (no header; column order per 1001genomes.org master list)
acc_meta = {}
with open(META / "accessions_1001g.csv") as fh:
    for r in csv.reader(fh):
        acc_meta[r[0]] = {"name": r[2], "country": r[3], "lat": r[5], "long": r[6],
                          "cs_number": r[9], "admixture_group": r[10]}

# GSE43857 samples
by_eco = defaultdict(list)
with open(META / "gse43857_samples.tsv") as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for row in rd:
        if row["in_vcf"] == "1":
            by_eco[row["ecotype_id"]].append(row)

out_rows = []
for eco, samples in by_eco.items():
    samples.sort(key=lambda s: (s["tissue"] != "leaf", s["gsm"]))  # leaf first
    chosen, alts = samples[0], samples[1:]
    am = acc_meta.get(eco, {})
    if not am:
        print(f"WARNING: ecotype {eco} not in 1001G accession metadata")
    out_rows.append({
        "ecotype_id": eco,
        "name": am.get("name", ""),
        "country": am.get("country", ""),
        "admixture_group": am.get("admixture_group", ""),
        "lat": am.get("lat", ""), "long": am.get("long", ""),
        "gsm": chosen["gsm"], "tissue": chosen["tissue"],
        "allc_file": chosen["suppl_url"].rsplit("/", 1)[-1],
        "alt_gsm": ";".join(a["gsm"] for a in alts),
        "source_series": "GSE43857",
    })

out_rows.sort(key=lambda r: int(r["ecotype_id"]))
fields = list(out_rows[0].keys())
with open(META / "benchmark_accessions.tsv", "w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerows(out_rows)

from collections import Counter
print(f"accessions: {len(out_rows)}")
print("tissue of chosen sample:", dict(Counter(r["tissue"] for r in out_rows)))
print("admixture groups:", dict(Counter(r["admixture_group"] for r in out_rows)))
print(f"wrote {META/'benchmark_accessions.tsv'}")

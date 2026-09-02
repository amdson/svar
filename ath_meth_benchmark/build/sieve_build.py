#!/usr/bin/env python3
"""SIEVE (Brachypodium induced-mutation) dataset build.

A STANDALONE benchmark, not a transfer target: models are trained/fine-tuned
within Brachypodium (or are multi-species gLMs) and evaluated on induced
point-mutation effects. Arabidopsis->Brachypodium zero-shot transfer remains
an optional secondary evaluation, not the headline. Because fine-tuning on
SIEVE's own pairs is in scope, gene-family-aware splits apply (mutations are
private to lines, so leakage is by gene/family, never by line).

Mirrors the Arabidopsis expression dataset format (see docs/BENCHMARK_MECHANICS.md):
same HDF5 layout, same deviation-from-mean target, same per-line SNV-array
scheme. Differences by design: no accession-axis machinery (isogenic, no LD),
and the evaluation unit is the per-mutation contrast, so the build emits a
(gene, line) PAIR TABLE labelling which pairs carry a cis mutation.

Selective-training rule (genotype-based, never outcome-based): pairs with
>=1 SNV within --cis-window of the gene are `focus`; a seeded random sample
of wild-type pairs (--background-ratio per focus pair) is `background` for
zero-calibration; everything else is excluded from training feeds. Selecting
on observed expression instead would inflate effect sizes (winner's curse)
and teach the model that every mutation has an effect.

Inputs from Zenodo 10.5281/zenodo.18236856; gene coordinates from the
Bd21-3 v1.1 GFF3 (Phytozome) when available — without it the pair table and
TSS fields are skipped and only the expression + SNV stages run.

Controls are detected from the VCF: lines whose induced-SNV count is an
order of magnitude below the mutant median (paper: ~12.5 vs ~884 singletons).
"""
import argparse
import gzip
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CHUNK_LINES = 50_000


def parse_vcf(vcf_path):
    """Stream the VCF once. Returns (samples, per-sample dict of
    chrom/pos/alt lists for hom-ALT calls, het/miss counts, contig set)."""
    samples, contigs = None, {}
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                if line.startswith("##contig"):
                    kv = dict(p.split("=") for p in
                              line.strip()[10:-1].split(",") if "=" in p)
                    contigs[kv.get("ID", "?")] = int(kv.get("length", 0))
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                break
    n = len(samples)
    chroms = [[] for _ in range(n)]
    poss = [[] for _ in range(n)]
    alts = [[] for _ in range(n)]
    acs = [[] for _ in range(n)]     # per-variant hom-ALT carrier count
    n_het = np.zeros(n, dtype=np.int64)
    n_miss = np.zeros(n, dtype=np.int64)
    n_rows = n_multi = 0
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            n_rows += 1
            chrom, pos, ref, alt = f[0], int(f[1]), f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or alt not in "ACGT":
                n_multi += 1
                continue
            carriers = []
            for i, g in enumerate(f[9:]):
                gt = g.split(":", 1)[0]
                if gt in ("1/1", "1|1"):
                    carriers.append(i)
                elif gt in ("0/1", "1/0", "0|1", "1|0"):
                    n_het[i] += 1
                elif gt in ("./.", "."):
                    n_miss[i] += 1
            ac = len(carriers)
            for i in carriers:
                chroms[i].append(chrom)
                poss[i].append(pos)
                alts[i].append(alt)
                acs[i].append(ac)
    print(f"VCF: {n_rows:,} rows ({n_multi:,} non-SNV skipped), "
          f"{n} samples, contigs: {list(contigs)[:8]}")
    return samples, chroms, poss, alts, acs, n_het, n_miss, contigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gff3", default=None,
                    help="Bd21-3 v1.1 gene GFF3; omit to skip pair table")
    ap.add_argument("--gene-id-attr", default="Name",
                    help="GFF attribute carrying gene ids (matched after "
                         "stripping the 'BdiBd21-3.' prefix)")
    ap.add_argument("--cis-window", type=int, default=4000,
                    help="bp around gene span defining a cis mutation")
    ap.add_argument("--max-ac", type=int, default=1,
                    help="max population carrier count for a variant to count "
                         "as an induced mutation in the pair table (1 = "
                         "singletons; shared variants are seed-stock "
                         "background, not mutagenesis)")
    ap.add_argument("--background-ratio", type=float, default=2.0)
    ap.add_argument("--no-log", action="store_true",
                    help="skip log2(1+x); the Zenodo README claims the matrix "
                         "is already log10(1+TPM) but observed values (e.g. 21.1) "
                         "are clearly linear-scale, so log is applied by default")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- expression ----
    genes = pd.read_csv(d / "peer.genes.csv", header=None)[0].astype(str).tolist()
    samp = pd.read_csv(d / "peer.samples.csv", header=None)[0].astype(str).tolist()
    X = pd.read_csv(d / "peer.expression.csv", header=None).to_numpy(np.float64)
    if X.shape == (len(samp), len(genes)):
        X = X.T
    assert X.shape == (len(genes), len(samp)), \
        f"expression {X.shape} vs {len(genes)} genes x {len(samp)} samples"
    if not a.no_log:
        assert X.min() >= 0, "negative values; matrix may already be log-scale"
        X = np.log2(1.0 + X)
    print(f"expression: {len(genes):,} genes x {len(samp)} lines "
          f"(log2(1+x) applied: {not a.no_log})")

    # ---- variants ----
    vcf = d / "snps.combined.M5.filtered.renamed.vcf.gz"
    vsamples, chroms, poss, alts, acs, n_het, n_miss, contigs = parse_vcf(vcf)
    n_alt = np.array([len(p) for p in poss])
    # induced mutations are population-private; shared variants are background
    # seed-stock heterogeneity (paper counts SINGLETONS: mutants ~884, ctrl ~12.5)
    n_single = np.array([sum(1 for a in ac if a == 1) for ac in acs])
    order = {s: i for i, s in enumerate(vsamples)}
    missing = [s for s in samp if s not in order]
    print(f"expression lines absent from VCF: {len(missing)}"
          + (f" e.g. {missing[:5]}" if missing else ""))

    # controls: singleton count an order of magnitude below the mutant median
    med = np.median(n_single[n_single > 0])
    ctrl_thresh = med / 10
    is_control = np.array([n_single[order[s]] < ctrl_thresh if s in order else False
                           for s in samp])
    print(f"median singletons/line {med:.0f} (median total {np.median(n_alt):.0f}); "
          f"control threshold {ctrl_thresh:.0f}; "
          f"controls detected: {int(is_control.sum())}")

    # ---- deviation target: subtract control-line mean per gene ----
    ref_cols = np.flatnonzero(is_control) if is_control.sum() >= 10 else \
        np.arange(len(samp))
    gene_ref = X[:, ref_cols].mean(axis=1)
    dev = X - gene_ref[:, None]

    # ---- SNV arrays (same schema as arabidopsis snv_arrays.h5) ----
    with h5py.File(out / "sieve_snv_arrays.h5", "w") as h5:
        g = h5.create_group("acc")
        for s in samp:
            if s not in order:
                continue
            i = order[s]
            o = np.argsort(np.array(poss[i]) if poss[i] else np.empty(0))
            grp = g.create_group(s)
            grp.create_dataset("chrom", data=np.array(chroms[i], dtype="S16")[o]
                               if poss[i] else np.empty(0, "S16"))
            grp.create_dataset("pos", data=np.array(poss[i], np.int32)[o]
                               if poss[i] else np.empty(0, np.int32))
            grp.create_dataset("alt", data=np.array(alts[i], "S1")[o].view(np.uint8)
                               if poss[i] else np.empty(0, np.uint8))
            grp.create_dataset("ac", data=np.array(acs[i], np.int16)[o]
                               if poss[i] else np.empty(0, np.int16))
            grp.attrs["n_alt"] = len(poss[i])
            grp.attrs["n_singletons"] = int(n_single[i])
            grp.attrs["n_het_skipped"] = int(n_het[i])
            grp.attrs["n_missing_skipped"] = int(n_miss[i])
        h5.attrs["source"] = "SIEVE M5 VCF (Zenodo 18236856), hom-ALT SNVs only"
        h5.attrs["note"] = ("chrom stored as strings; per-chrom sort within line; "
                            "coordinates are the VCF's (Bd21-3 assembly)")

    # ---- gene coordinates + pair table (needs GFF3) ----
    pair_note = "skipped (no --gff3)"
    coords = None
    if a.gff3:
        gf = pd.read_csv(a.gff3, sep="\t", comment="#", header=None,
                         names=["chrom", "src", "type", "start", "end",
                                "score", "strand", "frame", "attr"],
                         dtype={"chrom": str}, compression="infer")
        gg = gf[gf["type"] == "gene"].copy()
        gid = gg["attr"].str.extract(rf"{a.gene_id_attr}=([^;]+)")[0]
        gid = gid.str.replace("^BdiBd21-3\\.", "", regex=True) \
                 .str.replace("\\.v1\\.1$", "", regex=True)
        gg["gene_id"] = gid
        gg = gg.dropna(subset=["gene_id"]).drop_duplicates("gene_id") \
               .set_index("gene_id")
        common = [g for g in genes if g in gg.index]
        print(f"genes with GFF coordinates: {len(common):,} / {len(genes):,}")
        coords = gg.loc[common]

        rng = np.random.default_rng(a.seed)
        gi = {g: k for k, g in enumerate(genes)}
        rows = []
        for s in samp:
            if s not in order or is_control[samp.index(s)]:
                continue
            i = order[s]
            if not poss[i]:
                continue
            induced = np.array(acs[i]) <= a.max_ac
            mchrom = np.array(chroms[i])[induced]
            mpos = np.array(poss[i], np.int64)[induced]
            for c in coords["chrom"].unique():
                sel = mchrom == c
                if not sel.any():
                    continue
                cp = np.sort(mpos[sel])
                sub = coords[coords["chrom"] == c]
                lo = np.searchsorted(cp, sub["start"].to_numpy() - a.cis_window)
                hi = np.searchsorted(cp, sub["end"].to_numpy() + a.cis_window,
                                     side="right")
                hit = hi > lo
                for g, n in zip(sub.index[hit], (hi - lo)[hit]):
                    rows.append((g, s, int(n)))
        pairs = pd.DataFrame(rows, columns=["gene_id", "line", "n_cis_mut"])
        pairs["role"] = "focus"
        n_bg = int(len(pairs) * a.background_ratio)
        mut_lines = [s for s in samp if s in order and
                     not is_control[samp.index(s)]]
        bg_g = rng.choice(len(common), size=n_bg)
        bg_l = rng.choice(len(mut_lines), size=n_bg)
        focus_set = set(zip(pairs["gene_id"], pairs["line"]))
        bg = pd.DataFrame({"gene_id": [common[k] for k in bg_g],
                           "line": [mut_lines[k] for k in bg_l]})
        bg = bg[~bg.apply(lambda r: (r["gene_id"], r["line"]) in focus_set,
                          axis=1)].drop_duplicates()
        bg["n_cis_mut"] = 0
        bg["role"] = "background"
        pairs = pd.concat([pairs, bg], ignore_index=True)
        for df_, col in [(pairs, "deviation")]:
            df_[col] = [dev[gi[g], samp.index(l)] for g, l in
                        zip(df_["gene_id"], df_["line"])]
        pairs.to_parquet(out / "sieve_pairs.parquet")
        pair_note = (f"{int((pairs['role']=='focus').sum()):,} focus + "
                     f"{int((pairs['role']=='background').sum()):,} background")
        print("pair table:", pair_note)

    # ---- dataset h5 ----
    str_dt = h5py.string_dtype()
    with h5py.File(out / "sieve_dataset.h5", "w") as h5:
        gg_ = h5.create_group("genes")
        gg_.create_dataset("gene_id", data=np.array(genes, dtype=object), dtype=str_dt)
        if coords is not None:
            sub = coords.reindex(genes)
            gg_.create_dataset("chrom", data=sub["chrom"].fillna("").to_numpy(dtype=object),
                               dtype=str_dt)
            for col in ["start", "end"]:
                gg_.create_dataset(col, data=sub[col].fillna(-1).to_numpy(np.int64))
            gg_.create_dataset("strand", data=sub["strand"].fillna("").to_numpy(dtype=object),
                               dtype=str_dt)
        gl = h5.create_group("lines")
        gl.create_dataset("line_id", data=np.array(samp, dtype=object), dtype=str_dt)
        gl.create_dataset("is_control", data=is_control)
        gl.create_dataset("n_induced_snvs",
                          data=np.array([n_alt[order[s]] if s in order else -1
                                         for s in samp], np.int32))
        h5.create_dataset("expr", data=X.astype(np.float32),
                          chunks=(1024, len(samp)), compression="lzf")
        h5.create_dataset("deviation", data=dev.astype(np.float32),
                          chunks=(1024, len(samp)), compression="lzf")
        h5.attrs["expr_transform"] = ("log2(1+x) of PEER-corrected matrix"
                                      if not a.no_log else "as distributed")
        h5.attrs["source"] = "SIEVE / EMPRES Zenodo 18236856; PEER-corrected expression"
        h5.attrs["target"] = ("deviation = expr - per-gene control-line mean "
                              f"(ref cols: {'controls' if is_control.sum() >= 10 else 'all lines'})")
        h5.attrs["pairs"] = pair_note
    print("wrote", out / "sieve_dataset.h5")


if __name__ == "__main__":
    main()

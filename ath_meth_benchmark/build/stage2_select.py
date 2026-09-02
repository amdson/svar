#!/usr/bin/env python3
"""Stage 2 (spec §3): site selection from Stage 1 accumulator arrays.

Produces:
  sites_variable.parquet    — the evaluation set (§3a), with subtask labels (§3b)
  sites_invariant.parquet   — matched train-only low/high invariant sites (§3d)
  chh_bins.parquet          — binned CHH target (§3c), emitted if single-site
                              CHH retention < --chh-bin-threshold
  report_stage2.md          — GO/NO-GO checkpoint report

Annotation classes (int8 / string): te > te_boundary > genic > intergenic,
painted position-wise from the TAIR10 TE list and the Ensembl GFF3.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHROMS = ["1", "2", "3", "4", "5"]
CTX_NAME = {1: "CG", 2: "CHG", 3: "CHH"}
ANN_NAME = {0: "intergenic", 1: "genic", 2: "te_boundary", 3: "te"}
BOUNDARY_BP = 200


def paint_annotation(gff3, te_txt, offsets, lens, genome_len):
    """int8 position-class array over the strand-collapsed genome."""
    ann = np.zeros(genome_len, dtype=np.int8)

    def paint(chrom, start, end, val):  # 1-based inclusive coords
        off = offsets[chrom]
        s = max(0, start - 1) + off
        e = min(lens[chrom], end) + off
        if e > s:
            ann[s:e] = val

    gf = pd.read_csv(gff3, sep="\t", comment="#", header=None,
                     names=["chrom", "src", "type", "start", "end", "score",
                            "strand", "frame", "attr"],
                     dtype={"chrom": str}, compression="infer")
    genes = gf[gf["type"].isin(["gene", "ncRNA_gene"]) & gf["chrom"].isin(CHROMS)]
    for c, s, e in zip(genes["chrom"], genes["start"], genes["end"]):
        paint(c, s, e, 1)

    te = pd.read_csv(te_txt, sep="\t")
    te["chrom"] = te["Transposon_Name"].str.extract(r"AT(\d)TE")[0]
    te = te[te["chrom"].isin(CHROMS)]
    for c, s, e in zip(te["chrom"], te["Transposon_min_Start"], te["Transposon_max_End"]):
        paint(c, s - BOUNDARY_BP, e + BOUNDARY_BP, 2)
    for c, s, e in zip(te["chrom"], te["Transposon_min_Start"], te["Transposon_max_End"]):
        paint(c, s, e, 3)
    return ann, len(genes), len(te)


def cg_density(context, genome_len, halfwin=100):
    """Per-position count of CG sites (either strand) within +-halfwin bp."""
    cg = ((context[0::2] == 1) | (context[1::2] == 1)).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(cg, dtype=np.int64)])
    pos = np.arange(genome_len, dtype=np.int64)
    lo = np.clip(pos - halfwin, 0, genome_len)
    hi = np.clip(pos + halfwin + 1, 0, genome_len)
    return (cs[hi] - cs[lo]).astype(np.int16)


def idx_to_coords(idx, offsets, lens):
    gpos = idx // 2
    strand = np.where(idx % 2 == 0, "+", "-")
    chrom = np.empty(len(idx), dtype="U1")
    pos = np.empty(len(idx), dtype=np.int64)
    bounds = [(c, offsets[c], offsets[c] + lens[c]) for c in CHROMS]
    for c, lo, hi in bounds:
        m = (gpos >= lo) & (gpos < hi)
        chrom[m] = c
        pos[m] = gpos[m] - lo + 1
    return chrom, pos, strand


def pos_split(chrom):
    out = np.full(len(chrom), "train", dtype="U5")
    out[chrom == "4"] = "val"
    out[chrom == "5"] = "test"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", required=True)
    ap.add_argument("--gff3", required=True)
    ap.add_argument("--te-txt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kappa", type=float, default=3.0)
    ap.add_argument("--min-frac", type=float, default=0.9,
                    help="min fraction of accessions with n_obs coverage (§3a)")
    ap.add_argument("--invariant-ratio", type=float, default=2.0,
                    help="invariant sites sampled per variable site, per stratum (§3d)")
    ap.add_argument("--chh-bin-threshold", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    s1 = Path(a.stage1_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((s1 / "stage1_meta.json").read_text())
    offsets, lens = meta["chrom_offsets"], meta["chrom_lengths"]
    genome_len, n_acc = meta["genome_len"], meta["n_accessions"]
    n_min = int(np.ceil(a.min_frac * n_acc))

    n_obs = np.load(s1 / "n_obs.npy")
    context = np.load(s1 / "context.npy")
    ctx_changed = np.load(s1 / "ctx_changed.npy")

    covered = np.flatnonzero(n_obs >= n_min)
    print(f"sites with n_obs >= {n_min} ({a.min_frac} * {n_acc}): {len(covered):,}")

    no = n_obs[covered].astype(np.float64)
    sum_p = np.load(s1 / "sum_p.npy")[covered]
    sum_p2 = np.load(s1 / "sum_p2.npy")[covered]
    sum_noise = np.load(s1 / "sum_noise.npy")[covered]
    var_obs = (sum_p2 - sum_p ** 2 / no) / (no - 1)
    noise = sum_noise / no
    mean_p = sum_p / no
    ctx = context[covered]
    chg = ctx_changed[covered]

    ann_pos, n_genes, n_tes = paint_annotation(a.gff3, a.te_txt, offsets, lens, genome_len)
    cgd = cg_density(context, genome_len)
    ann = ann_pos[covered // 2]
    cgd_cov = cgd[covered // 2]

    known_ctx = ctx > 0  # drop sites whose 3bp context was N in every accession
    retain = (var_obs > a.kappa * noise) & known_ctx
    low_inv = (mean_p < 0.02) & (var_obs < noise) & known_ctx
    high_inv = (mean_p > 0.8) & (var_obs < noise) & known_ctx

    def make_table(sel, subtask=None, split_role="eval"):
        i = covered[sel]
        chrom, pos, strand = idx_to_coords(i, offsets, lens)
        df = pd.DataFrame({
            "chrom": chrom, "pos": pos, "strand": strand,
            "context": pd.Series(ctx[sel]).map(CTX_NAME),
            "mean_p": mean_p[sel].astype(np.float32),
            "var_obs": var_obs[sel].astype(np.float32),
            "noise": noise[sel].astype(np.float32),
            "n_obs": n_obs[covered[sel]],
            "annotation_class": pd.Series(ann[sel]).map(ANN_NAME),
            "cg_density": cgd_cov[sel],
            "site_idx": i,
        })
        if subtask is not None:
            df["subtask"] = subtask
        df["split_role"] = split_role
        df["pos_split"] = pos_split(df["chrom"].to_numpy())
        return df

    # ---- variable (evaluation) set with §3b subtask labels ----
    var_df = make_table(retain)
    var_df["subtask"] = np.where(chg[retain] == 1, "context_changing", "invariant_context")
    var_df.to_parquet(out / "sites_variable.parquet", index=False)

    # ---- §3d matched invariant sampling ----
    rng = np.random.default_rng(a.seed)
    qs = np.quantile(var_df["cg_density"], [0.2, 0.4, 0.6, 0.8])

    def stratum(annv, cgv):
        return annv.astype(np.int32) * 8 + np.digitize(cgv, qs)

    var_strat = stratum(ann[retain], cgd_cov[retain])
    strat_counts = pd.Series(var_strat).value_counts()

    inv_parts = []
    for name, mask in [("low_invariant", low_inv), ("high_invariant", high_inv)]:
        pool = np.flatnonzero(mask)
        pool_strat = stratum(ann[pool], cgd_cov[pool])
        take = []
        for st, cnt in strat_counts.items():
            cand = pool[pool_strat == st]
            k = min(len(cand), int(cnt * a.invariant_ratio))
            if k:
                take.append(rng.choice(cand, size=k, replace=False))
        if take:
            sel = np.zeros(len(covered), dtype=bool)
            sel[np.concatenate(take)] = True
            df = make_table(sel, subtask=name, split_role="train_only")
            inv_parts.append(df)
    inv_df = pd.concat(inv_parts, ignore_index=True)
    inv_df.to_parquet(out / "sites_invariant.parquet", index=False)

    # ---- §3c binned CHH ----
    chh_retained = int((var_df["context"] == "CHH").sum())
    bins_df = None
    if chh_retained < a.chh_bin_threshold:
        WINL = meta["win"]
        wno = np.load(s1 / "win_n_obs.npy")
        wcov = np.flatnonzero(wno >= n_min)
        wn = wno[wcov].astype(np.float64)
        wsp = np.load(s1 / "win_sum_p.npy")[wcov]
        wsp2 = np.load(s1 / "win_sum_p2.npy")[wcov]
        wsn = np.load(s1 / "win_sum_noise.npy")[wcov]
        wvar = (wsp2 - wsp ** 2 / wn) / (wn - 1)
        wnoise = wsn / wn
        wret = wcov[wvar > a.kappa * wnoise]
        gstart = wret.astype(np.int64) * WINL
        chrom = np.empty(len(wret), dtype="U1")
        start = np.empty(len(wret), dtype=np.int64)
        for c in CHROMS:
            m = (gstart >= offsets[c]) & (gstart < offsets[c] + lens[c])
            chrom[m] = c
            start[m] = gstart[m] - offsets[c] + 1
        keep = wvar > a.kappa * wnoise
        bins_df = pd.DataFrame({
            "chrom": chrom, "start": start, "end": start + WINL - 1,
            "mean_p": (wsp[keep] / wn[keep]).astype(np.float32),
            "var_obs": wvar[keep].astype(np.float32),
            "noise": wnoise[keep].astype(np.float32),
            "n_obs": wno[wret],
            "annotation_class": pd.Series(ann_pos[np.minimum(gstart + WINL // 2,
                                                             genome_len - 1)]).map(ANN_NAME),
            "win_idx": wret,
        })
        bins_df["pos_split"] = pos_split(bins_df["chrom"].to_numpy())
        bins_df.to_parquet(out / "chh_bins.parquet", index=False)

    # ---- report ----
    kappa_ceiling = np.sqrt((a.kappa - 1) / a.kappa)
    by_ca = var_df.groupby(["context", "annotation_class"]).size().unstack(fill_value=0)
    by_ctx = var_df.groupby("context").size()
    by_sub = var_df.groupby(["context", "subtask"]).size().unstack(fill_value=0)
    cg_n = int(by_ctx.get("CG", 0))
    verdict = ("GO — CG retained > 500,000" if cg_n > 500_000 else
               "NO-GO — CG retained < 50,000; not enough signal" if cg_n < 50_000 else
               "MARGINAL — CG retained between 50k and 500k; user decision")

    lines = [
        "# Stage 2 report — site selection (GO/NO-GO checkpoint)\n",
        f"- accessions: {n_acc}; coverage requirement: n_obs >= {n_min} "
        f"({a.min_frac:.0%} of accessions at MIN_COV={meta['min_cov']})",
        f"- KAPPA = {a.kappa} -> measurement ceiling r = {kappa_ceiling:.2f}",
        f"- sites passing coverage: {len(covered):,} "
        f"(of {2 * genome_len:,} possible strand-sites)",
        f"- annotation: {n_genes:,} genes, {n_tes:,} TEs "
        f"(te > te_boundary({BOUNDARY_BP}bp) > genic > intergenic)\n",
        f"## VERDICT: {verdict}\n",
        "## Variable (evaluation) set — retained by context",
        by_ctx.to_markdown(), "",
        "## Retained by context x annotation",
        by_ca.to_markdown(), "",
        "## Subtask split (§3b)",
        by_sub.to_markdown(), "",
        "## Invariant train-only sets (§3d, matched sampling at "
        f"ratio {a.invariant_ratio})",
        inv_df.groupby(["subtask", "annotation_class"]).size().unstack(fill_value=0).to_markdown(), "",
    ]
    if bins_df is not None:
        lines += [
            f"## Binned CHH (§3c) — triggered (single-site CHH = {chh_retained:,} "
            f"< {a.chh_bin_threshold:,})",
            f"- retained 100bp windows: {len(bins_df):,}",
            bins_df.groupby("annotation_class").size().to_markdown(), "",
        ]
    else:
        lines += [f"## Binned CHH — not triggered (single-site CHH = {chh_retained:,})\n"]
    (out / "report_stage2.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""
Dumb baseline for SIEVE within-gene prediction: pooled elasticnet on shared
mutation/gene features.

Per-gene elasticnet over cis-SNPs (the PrediXcan-style baseline) is structurally
degenerate on SIEVE — mutations are private, so a held-out line's variant column
is all-zero in training and the model collapses to predict-zero. The dumb
baseline that IS available pools all single-mutation focus pairs and predicts
the z-scored deviation from features shared across genes and lines:

  * mutation position: strand-relative signed distance to TSS/TTS, region
    (promoter / 5' / body / 3' / distal), log distance
  * mutation identity: ref>alt one-hot (12), transition flag, +/-25 bp GC
  * gene context: control mean expression, control noise sd

Targets: signed z (the real task) and |z| (magnitude only). Splits: held-out
gene families (genes/family_split) and held-out lines (hash 80/20) — both valid
because features are shared. Expectation worth falsifying: dumb features get
some magnitude signal (TSS-proximal mutations hit harder) and ~none of the
sign; a sequence model earns its keep only by beating the signed row.

    python -m model_dev.sieve_dumb_baseline
"""
from __future__ import annotations

import sys

import numpy as np

BASE = "/90daydata/small_grains/andrew.dickson/datasets/brachypodium_sieve/"
DATA = BASE + "dataset/"
FASTA = BASE + "reference/Bd21_3.fa"
HW = 5000  # pairs builder window: gene body +/- 5 kb (calibrated, 99% match)


def main() -> int:
    import h5py
    import pandas as pd
    import pysam
    from sklearn.linear_model import ElasticNetCV
    from scipy.stats import pearsonr, spearmanr

    with h5py.File(DATA + "sieve_dataset.h5", "r") as f:
        dev = f["deviation"][:]
        gid = f["genes/gene_id"][:].astype(str)
        chrom = f["genes/chrom"][:].astype(str)
        start, end = f["genes/start"][:], f["genes/end"][:]
        strand = f["genes/strand"][:].astype(str)
        fam_split = f["genes/family_split"][:].astype(str)
        is_ctrl = f["lines/is_control"][:]
        line_id = f["lines/line_id"][:].astype(str)
        mean_log2 = f["expr"][:][:, is_ctrl].mean(axis=1)  # control-line mean
    tss = np.where(strand == "+", start, end)
    tts = np.where(strand == "+", end, start)
    gix = {g: i for i, g in enumerate(gid)}
    lix = {l: i for i, l in enumerate(line_id)}

    sd = dev[:, is_ctrl].std(axis=1, ddof=1)
    ok_gene = sd > 1e-3

    snv = {}
    with h5py.File(DATA + "sieve_snv_arrays.h5", "r") as f:
        for line in f["acc"]:
            g = f["acc"][line]
            snv[line] = (g["chrom"][:].astype(str), g["pos"][:],
                         g["alt"][:])

    pairs = pd.read_parquet(DATA + "sieve_pairs.parquet")

    # per-line offset from background pairs (z units), as in sieve_signal_gate
    bg = pairs[pairs.role == "background"]
    bgi = bg["gene_id"].map(gix).to_numpy()
    bli = bg["line"].map(lix).to_numpy()
    m = ok_gene[bgi]
    zbg = dev[bgi[m], bli[m]] / sd[bgi[m]]
    off = np.zeros(len(line_id))
    cnt = np.zeros(len(line_id))
    np.add.at(off, bli[m], zbg)
    np.add.at(cnt, bli[m], 1)
    off = np.where(cnt > 0, off / np.maximum(cnt, 1), 0.0)

    foc = pairs[(pairs.role == "focus") & (pairs.n_cis_mut == 1)]
    fa = pysam.FastaFile(FASTA)

    rows, y = [], []
    n_skip = 0
    for gene, line in zip(foc["gene_id"], foc["line"]):
        i, j = gix.get(gene), lix.get(line)
        if i is None or j is None or not ok_gene[i] or line not in snv:
            n_skip += 1
            continue
        c, p, alt = snv[line]
        m = (c == chrom[i]) & (p >= start[i] - HW) & (p < end[i] + HW)
        if m.sum() != 1:
            n_skip += 1
            continue
        pos, altb = int(p[m][0]), chr(alt[m][0])
        sgn = 1 if strand[i] == "+" else -1
        d_tss = sgn * (pos - tss[i])
        d_tts = sgn * (pos - tts[i])
        try:
            ctx = fa.fetch(chrom[i], max(pos - 26, 0), pos + 25).upper()
            refb = fa.fetch(chrom[i], pos - 1, pos).upper()
        except (KeyError, ValueError):
            n_skip += 1
            continue
        gc = (ctx.count("G") + ctx.count("C")) / max(len(ctx), 1)
        transition = (refb + altb) in ("AG", "GA", "CT", "TC")
        region = ("promoter" if -1000 <= d_tss < 0 else
                  "five_prime" if 0 <= d_tss < 1000 else
                  "body" if min(start[i], end[i]) <= pos < max(start[i], end[i])
                  else "three_prime" if 0 <= d_tts < 1000 else "distal")
        rows.append(dict(gene_i=i, line=line, d_tss=d_tss,
                         log_d_tss=np.log1p(abs(d_tss)),
                         log_d_tts=np.log1p(abs(d_tts)), gc=gc,
                         transition=float(transition), region=region,
                         mut=refb + ">" + altb, mean_log2=mean_log2[i],
                         noise_sd=sd[i], fam=fam_split[i]))
        y.append(dev[i, j] / sd[i] - off[j])
    df = pd.DataFrame(rows)
    df["z"] = np.array(y)
    print(f"featurized {len(df):,} single-mutation focus pairs "
          f"(skipped {n_skip:,}: multi/unlocatable/unscoreable)")

    X = pd.get_dummies(df[["d_tss", "log_d_tss", "log_d_tts", "gc",
                           "transition", "mean_log2", "noise_sd",
                           "region", "mut"]],
                       columns=["region", "mut"]).astype(float)
    X = (X - X.mean()) / X.std().replace(0, 1)

    rng = np.random.default_rng(7)
    lines = np.unique(df["line"])
    test_lines = set(rng.choice(lines, size=len(lines) // 5, replace=False))
    splits = {
        "held-out gene families": (df["fam"] == "train").to_numpy(),
        "held-out lines": (~df["line"].isin(test_lines)).to_numpy(),
    }
    for target, yv in [("signed z", df["z"].to_numpy()),
                       ("|z| (magnitude)", np.abs(df["z"].to_numpy()))]:
        print(f"\ntarget: {target}")
        for name, tr in splits.items():
            te = ~tr
            enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=20, cv=5,
                                max_iter=5000, n_jobs=-1, random_state=0)
            enet.fit(X[tr], yv[tr])
            pred = enet.predict(X[te])
            ss = 1 - ((yv[te] - pred) ** 2).sum() / ((yv[te] - yv[te].mean()) ** 2).sum()
            r_p = pearsonr(yv[te], pred)
            r_s = spearmanr(yv[te], pred)
            nz = (np.abs(enet.coef_) > 1e-8).sum()
            print(f"  {name:>22}: R2={ss:+.4f}  pearson={r_p.statistic:+.4f} "
                  f"(p={r_p.pvalue:.1e})  spearman={r_s.statistic:+.4f}  "
                  f"[{nz}/{X.shape[1]} features kept]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

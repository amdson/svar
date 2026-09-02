#!/usr/bin/env python3
"""T1 baseline sandwich (design doc §4): for every gene, on the accession
split (train=531 / val / test by admixture group):

  1. population mean        — floor; exactly r=0 on deviations by construction
  2. kinship BLUP           — global GRM kernel ridge; per-gene lambda by
                              closed-form LOO on the train eigendecomposition
  3. elastic net on cis SNPs— MAF>=0.01 within panel, +-CIS_WINDOW of TSS;
                              the model to beat
  4. cis-h2 ceiling         — two-component Haseman-Elston regression
                              (K_cis + K_global), fast and unbiased; can be
                              swapped for REML/GEMMA later

Outputs per-gene parquet (spearman r for 2&3 on val+test, cis-h2, n_cis, ...)
and t1_report.md with the headroom summary that decides whether T1
discriminates anything (see benchmark-viability discussion).

Genotypes come from the UNFILTERED biallelic pgen (rare alleles kept).
"""
import argparse
import json
from multiprocessing import Pool, shared_memory
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pgenlib
from sklearn.linear_model import ElasticNetCV

G = {}


def rank(a, axis=-1):
    order = np.argsort(a, axis=axis, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(a.shape[axis]), axis=axis)
    return r.astype(np.float64)


def spearman_rows(A, b):
    """Spearman r of each row of A vs b (ties broken by order; fine for
    continuous predictions)."""
    ra, rb = rank(A), rank(b[None, :])[0]
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum())
    return np.where(denom > 0, (ra * rb).sum(1) / np.maximum(denom, 1e-12), 0.0)


def load_genotypes(pfile, sample_ids):
    psam = pd.read_csv(pfile + ".psam", sep="\t", dtype=str)
    iid = psam.iloc[:, 0].tolist()
    idx = np.array([iid.index(s) for s in sample_ids], dtype=np.uint32)
    order = np.argsort(idx)
    rdr = pgenlib.PgenReader((pfile + ".pgen").encode(),
                             sample_subset=np.sort(idx))
    n_var = rdr.get_variant_ct()
    geno = np.empty((n_var, len(idx)), dtype=np.int8)
    step = 200_000
    for s in range(0, n_var, step):
        e = min(s + step, n_var)
        rdr.read_range(s, e, geno[s:e])
    # columns come back in sorted-idx order; restore requested order
    inv = np.empty(len(idx), dtype=np.int64)
    inv[order] = np.arange(len(idx))
    geno = geno[:, inv]
    pvar = pd.read_csv(pfile + ".pvar", sep="\t", comment="#", header=None,
                       usecols=[0, 1], names=["chrom", "pos"],
                       dtype={"chrom": str, "pos": np.int64})
    return geno, pvar


def snp_stats(geno):
    """Per-SNP alt-allele frequency and MAF over non-missing calls (chunked)."""
    n_var = geno.shape[0]
    af = np.empty(n_var, dtype=np.float32)
    for s in range(0, n_var, 500_000):
        g = geno[s:s + 500_000].astype(np.float32)
        miss = g < 0
        g[miss] = np.nan
        af[s:s + 500_000] = np.nanmean(g, axis=1) / 2.0
    maf = np.minimum(af, 1 - af)
    return af, maf


def standardized_window(geno, rows, cols, af):
    X = geno[np.ix_(rows, cols)].astype(np.float32).T  # (n_samples, p)
    for j, r in enumerate(rows):
        m = X[:, j] < 0
        if m.any():
            X[m, j] = 2.0 * af[r]
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def grm_from_rows(geno, rows, af):
    n = geno.shape[1]
    K = np.zeros((n, n), dtype=np.float64)
    for s in range(0, len(rows), 20_000):
        Z = standardized_window(geno, rows[s:s + 20_000], np.arange(n), af)
        K += Z @ Z.T
    return K / len(rows)


def worker(gi):
    geno = G["geno"]
    af, maf = G["af"], G["maf"]
    tr, te, va = G["tr"], G["te"], G["va"]
    y = G["Y"][gi]
    key = int(G["chrom"][gi]) * 10 ** 9 + G["tss"][gi]
    lo, hi = np.searchsorted(G["snp_key"],
                             [key - G["cis_w"], key + G["cis_w"] + 1])
    rows = np.arange(lo, hi)[maf[lo:hi] >= G["maf_min"]]
    out = {"gene_i": gi, "n_cis": len(rows)}
    if len(rows) == 0:
        return out

    Ztr = standardized_window(geno, rows, tr, af)
    yt = y[tr]

    # --- elastic net ---
    try:
        en = ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=5, max_iter=3000,
                          random_state=0)
        en.fit(Ztr, yt)
        Zte = standardized_window(geno, rows, te, af)
        Zva = standardized_window(geno, rows, va, af)
        out["en_r_test"] = float(spearman_rows(en.predict(Zte)[None, :], y[te])[0])
        out["en_r_val"] = float(spearman_rows(en.predict(Zva)[None, :], y[va])[0])
        out["en_nnz"] = int((en.coef_ != 0).sum())
    except Exception as e:
        out["en_error"] = str(e)

    # --- cis-h2 via two-component Haseman-Elston on train accessions ---
    Kc = (Ztr @ Ztr.T) / max(len(rows), 1)
    Kg = G["K_train"]
    yc = yt - yt.mean()
    vy = yc.var()
    if vy > 0:
        iu = np.triu_indices(len(tr), k=1)
        z = (np.outer(yc, yc) / vy)[iu]
        A = np.column_stack([Kc[iu], Kg[iu], np.ones(len(z))])
        try:
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            out["cis_h2"] = float(np.clip(coef[0], 0, 1))
            out["glob_h2"] = float(np.clip(coef[1], 0, 1))
        except np.linalg.LinAlgError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--pfile", required=True)
    ap.add_argument("--prune-in", required=True,
                    help="plink2 .prune.in id list (@:#:$r:$a ids) for the GRM")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cis-window", type=int, default=100_000)
    ap.add_argument("--maf-min", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.h5, "r") as h5:
        acc = [x.decode() for x in h5["accessions/ecotype_id"][:]]
        acc_split = np.array([x.decode() for x in h5["accessions/acc_split"][:]])
        Y = h5["deviation"][:]
        chrom = np.array([x.decode() for x in h5["genes/chrom"][:]])
        tss = h5["genes/tss"][:]
        gene_id = [x.decode() for x in h5["genes/gene_id"][:]]
    tr = np.flatnonzero(acc_split == "train")
    va = np.flatnonzero(acc_split == "val")
    te = np.flatnonzero(acc_split == "test")
    print(f"{len(gene_id):,} genes; accessions train/val/test = "
          f"{len(tr)}/{len(va)}/{len(te)}")

    print("loading genotypes...")
    geno, pvar = load_genotypes(a.pfile, acc)
    af, maf = snp_stats(geno)
    snp_key = pvar["chrom"].astype(np.int64).to_numpy() * 10 ** 9 + \
        pvar["pos"].to_numpy(np.int64)
    assert (np.diff(snp_key) >= 0).all(), "pvar must be coordinate-sorted"

    # ---- GRM from pruned SNPs ----
    ids = set(open(a.prune_in).read().split())
    pid = (pvar["chrom"] + ":" + pvar["pos"].astype(str)).to_numpy()
    # prune.in ids are chrom:pos:ref:alt; match on chrom:pos prefix
    prefix = np.array([i.rsplit(":", 2)[0] for i in ids])
    grm_rows = np.flatnonzero(np.isin(pid, prefix))
    print(f"GRM SNPs: {len(grm_rows):,}")
    K = grm_from_rows(geno, grm_rows, af)
    np.save(out / "grm.npy", K)
    K_train = K[np.ix_(tr, tr)]

    # ---- kinship BLUP, all genes at once ----
    S, U = np.linalg.eigh(K_train)
    S = np.maximum(S, 0)
    Yt = Y[:, tr]
    Yr = Yt @ U                                     # genes x n_tr rotated
    lambdas = np.logspace(-2, 3, 30)
    best_loo = np.full(len(gene_id), np.inf)
    best_lam = np.ones(len(gene_id))
    for lam in lambdas:
        d = S / (S + lam)                           # hat matrix eigenvalues
        # LOO residual: (y - Hy)_i / (1 - h_ii)
        H_diag = (U ** 2 * d).sum(1)
        resid = Yt - (Yr * d) @ U.T
        loo = ((resid / (1 - H_diag)) ** 2).mean(1)
        better = loo < best_loo
        best_loo[better] = loo[better]
        best_lam[better] = lam
    blup_r_test = np.zeros(len(gene_id))
    blup_r_val = np.zeros(len(gene_id))
    for lam in np.unique(best_lam):
        gsel = best_lam == lam
        alpha = (Yr[gsel] / (S + lam)) @ U.T        # (K_tt+lam I)^-1 y
        pred_te = alpha @ K[np.ix_(tr, te)]
        pred_va = alpha @ K[np.ix_(tr, va)]
        for gi_local, gi in enumerate(np.flatnonzero(gsel)):
            blup_r_test[gi] = spearman_rows(pred_te[gi_local][None, :], Y[gi, te])[0]
            blup_r_val[gi] = spearman_rows(pred_va[gi_local][None, :], Y[gi, va])[0]
    print("kinship BLUP done")

    # ---- per-gene EN + cis-h2 (parallel; geno shared via fork) ----
    G.update(geno=geno, af=af, maf=maf, tr=tr, te=te, va=va, Y=Y,
             chrom=chrom, tss=tss, snp_key=snp_key, cis_w=a.cis_window,
             maf_min=a.maf_min, K_train=K_train)
    with Pool(a.workers) as pool:
        rows = pool.map(worker, range(len(gene_id)), chunksize=64)
    df = pd.DataFrame(rows).set_index("gene_i").sort_index()
    df["gene_id"] = gene_id
    df["blup_r_test"] = blup_r_test
    df["blup_r_val"] = blup_r_val
    df.to_parquet(out / "t1_sandwich.parquet")

    # ---- report ----
    en = df["en_r_test"]
    h2 = df["cis_h2"]
    primary = df[h2 >= 0.1]
    lines = [
        "# T1 baseline sandwich report\n",
        f"- genes: {len(df):,}; cis window ±{a.cis_window/1000:.0f} kb, "
        f"MAF ≥ {a.maf_min}",
        f"- median cis-h2 (HE, all genes): {h2.median():.3f}; "
        f"genes with cis-h2 ≥ 0.05 / 0.1: "
        f"{int((h2 >= .05).sum()):,} / {int((h2 >= .1).sum()):,}",
        f"- kinship BLUP median Spearman r (test accessions): "
        f"{df['blup_r_test'].median():.3f}",
        f"- elastic net median Spearman r (test): {en.median():.3f}",
        f"\n## Primary gene set (cis-h2 ≥ 0.1, n={len(primary):,})",
        f"- elastic net median r: {primary['en_r_test'].median():.3f}",
        f"- BLUP median r: {primary['blup_r_test'].median():.3f}",
        f"- median headroom (cis_h2 − en_r_test²): "
        f"{(primary['cis_h2'] - primary['en_r_test'].clip(lower=0)**2).median():.3f}",
        "\nHeadroom > ~0.05 on a few thousand genes = T1 discriminates; "
        "≈0 = the action is in T2/T3 (see benchmark-viability note).",
    ]
    (out / "t1_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

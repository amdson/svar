"""CPU fitting on cached site features (from extract_features.py).

Per site, on train accessions, four models are fit and scored on held-out
accession groups (default val = asia; test = italy_balkan_caucasus):
  emb    RidgeCV on backbone features (delta from the TAIR10 window by default)
  geno   RidgeCV on cis SNV dosage inside the window   <- the linear bar
  null   emb with targets permuted across accessions   <- the floor
  stack  RidgeCV on features, fit to the residual of geno (is anything additive?)
Metric: per-site Pearson r across held-out accessions, median over sites, by
context x subtask (never pooled over site x accession pairs; measurement
ceiling r ~ 0.82 at KAPPA=3).

--transfer additionally fits ONE shared ridge across train-position sites
(features = delta, target = p minus the site's train mean) and scores it per
site on positions never seen (pos_split val/test in the npz): the unseen-site
axis where no per-site model exists.

  python -m training_meth.fit_sites --features $SVAR_SCRATCH/runs/meth_feat_carbon.npz --transfer
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV

ALPHAS = np.logspace(-2, 6, 17)
MIN_EVAL = 20


def _r(y, yhat):
    # nan when the prediction is (numerically) constant on the held-out accessions,
    # e.g. every held-out accession shares one window: no information, not a score.
    if len(y) < MIN_EVAL or np.std(y) < 1e-6 or np.std(yhat) < 1e-6:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(y.astype(np.float64), yhat.astype(np.float64))[0, 1]
    return float(r) if np.isfinite(r) else np.nan


def _ridge(X, y, w, Xe):
    m = RidgeCV(alphas=ALPHAS).fit(X, y, sample_weight=w)
    return m.predict(Xe), m.alpha_


def fit_site(s, Xf, y, total, geno, tr, evals, weight_cap, seed):
    """Returns one dict of results for site s (rows = accessions)."""
    cov = np.isfinite(y)
    trm = tr & cov
    if trm.sum() < 30:
        return None
    w = np.minimum(total[trm], weight_cap).astype(np.float64) / weight_cap
    out = {}
    rng = np.random.default_rng(seed + s)
    perm = y.copy()
    perm[cov] = rng.permutation(y[cov])
    for name, X, target in [("emb", Xf, y), ("null", Xf, perm)]:
        pred, alpha = _ridge(X[trm], target[trm], w, X)
        out[f"{name}_alpha"] = alpha
        for tag, em in evals.items():
            e = em & cov
            out[f"{name}_{tag}"] = _r(target[e], pred[e])
    if geno.shape[0] > 0:
        G = geno.T.astype(np.float32)
        pred_g, alpha = _ridge(G[trm], y[trm], w, G)
        out["geno_alpha"] = alpha
        pred_s, _ = _ridge(Xf[trm], (y - pred_g)[trm], w, Xf)
        for tag, em in evals.items():
            e = em & cov
            out[f"geno_{tag}"] = _r(y[e], pred_g[e])
            out[f"stack_{tag}"] = _r(y[e], (pred_g + pred_s)[e])
    else:
        for tag in evals:
            out[f"geno_{tag}"] = np.nan
            out[f"stack_{tag}"] = np.nan
    out["n_snv"] = int(geno.shape[0])
    out["n_train"] = int(trm.sum())
    return out


def site_features(d, s, raw=False):
    X = d["feats"][d["inverse"][s]].astype(np.float32)
    if not raw:
        X = X - d["feats"][d["ref_idx"][s]].astype(np.float32)[None]
    return X


def summarize(df, models, tags, by=("site_context", "site_subtask")):
    rows = []
    for key, g in df.groupby(list(by)):
        for m in models:
            for t in tags:
                col = f"{m}_{t}"
                v = g[col].to_numpy(dtype=float)
                ok = np.isfinite(v)
                rows.append({**dict(zip(by, key)), "model": m, "eval": t, "n_sites": int(ok.sum()),
                             "median_r": np.nanmedian(v) if ok.any() else np.nan,
                             "median_r_zerofill": np.median(np.where(ok, v, 0.0)),
                             "mean_r": np.nanmean(v) if ok.any() else np.nan,
                             "frac_r>0.3": float((v[ok] > 0.3).mean()) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def transfer(d, tags, evals, tr, raw, row_cap, seed):
    """Shared ridge across train-position sites, scored per site on held-out positions."""
    pos = d["site_pos_split"].astype(str)
    p = d["p"]
    fit_sites = np.flatnonzero(pos == "train")
    held = np.flatnonzero(pos != "train")
    if len(held) == 0:
        print("--transfer: no held-out-position sites in this npz (extract with --pos-splits train val)")
        return None
    rng = np.random.default_rng(seed)
    Xs, ys, ws = [], [], []
    for s in fit_sites:
        y = p[s]
        m = tr & np.isfinite(y)
        if m.sum() < 30:
            continue
        X = site_features(d, s, raw)[m]
        Xs.append(X); ys.append(y[m] - y[m].mean()); ws.append(np.minimum(d["total"][s][m], 30) / 30.0)
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    if len(y) > row_cap:
        keep = rng.choice(len(y), row_cap, replace=False)
        X, y, w = X[keep], y[keep], w[keep]
    print(f"--transfer: shared ridge on {len(fit_sites)} train-position sites, {len(y):,} rows, dim {X.shape[1]}")
    model = RidgeCV(alphas=ALPHAS).fit(X, y, sample_weight=w)
    print(f"  alpha = {model.alpha_:g}")
    rows = []
    for group, sites in [("seen_sites", fit_sites), ("unseen_sites", held)]:
        for s in sites:
            y = p[s]
            pred = model.predict(site_features(d, s, raw))
            row = {"group": group, "site": int(s), "context": d["site_context"][s], "subtask": d["site_subtask"][s]}
            for tag, em in evals.items():
                e = em & np.isfinite(y)
                row[tag] = _r(y[e], pred[e])
            rows.append(row)
    df = pd.DataFrame(rows)
    for tag in tags:
        print(f"  eval={tag}")
        print(df.groupby(["group", "context", "subtask"])[tag]
                .agg(n="count", median="median", mean="mean").round(4).to_string())
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval-splits", nargs="+", default=["val"])
    ap.add_argument("--raw", action="store_true", help="raw embeddings instead of delta from reference")
    ap.add_argument("--weight-cap", type=int, default=30, help="coverage weight = min(total, cap)/cap")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--row-cap", type=int, default=300_000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="per-site csv (default: next to --features)")
    a = ap.parse_args()

    t0 = time.time()
    d = dict(np.load(a.features, allow_pickle=False))
    meta = json.loads(str(d["meta"]))
    S = d["p"].shape[0]
    acc_split = d["acc_split"].astype(str)
    tr = acc_split == "train"
    evals = {t: acc_split == t for t in a.eval_splits}
    print(f"{a.features}: {S} sites x {len(acc_split)} accessions, backend {meta['backend']} "
          f"({meta['model_path']}), dim {meta['dim']}; train accs {tr.sum()}, "
          + ", ".join(f"{t} {m.sum()}" for t, m in evals.items()))

    def one(s):
        r = fit_site(s, site_features(d, s, a.raw), d["p"][s], d["total"][s],
                     d["geno"][d["geno_ptr"][s]:d["geno_ptr"][s + 1]], tr, evals, a.weight_cap, a.seed)
        return None if r is None else {"site": s, **r}

    res = [r for r in Parallel(n_jobs=a.n_jobs)(delayed(one)(s) for s in range(S)) if r]
    df = pd.DataFrame(res)
    for c in ["site_context", "site_subtask", "site_annotation_class", "site_pos_split", "site_chrom",
              "site_pos", "site_strand", "site_mean_p", "site_var_obs", "site_noise"]:
        if c in d:
            df[c] = d[c][df["site"].to_numpy()]
    models = ["emb", "geno", "stack", "null"]
    summ = summarize(df, models, a.eval_splits)
    pd.set_option("display.width", 200)
    print(f"\nper-site models ({len(df)} sites fit, {time.time()-t0:.0f}s); r = per-site Pearson across held-out accessions")
    print(summ.round(4).to_string(index=False))
    print("\nby annotation class (emb vs geno, first eval split):")
    t = a.eval_splits[0]
    print(df.groupby(["site_context", "site_annotation_class"])[[f"emb_{t}", f"geno_{t}", f"null_{t}"]]
            .median().round(4).to_string())

    out = a.out or a.features.replace(".npz", "_fit.csv")
    df.to_csv(out, index=False)
    summ.to_csv(out.replace("_fit.csv", "_summary.csv"), index=False)
    print(f"wrote {out}")

    if a.transfer:
        tdf = transfer(d, a.eval_splits, evals, tr, a.raw, a.row_cap, a.seed)
        if tdf is not None:
            tdf.to_csv(out.replace("_fit.csv", "_transfer.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

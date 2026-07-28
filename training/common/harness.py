"""
training/common/harness.py
--------------------------
The shared run loop. A per-directory ``run.py`` parses its args and hands a model
factory to ``run_sklearn`` (or, later, a torch runner); the harness does
everything shared: resolve dataset + split, build the feature matrix for the
chosen modality, fit one trait at a time on train (with the model's own internal
tuning), evaluate on val AND test, and write the run record + artifacts.
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from training.common import artifacts, features as feat, run_record
from training.common.datasets import get_dataset
from training.common.metrics import evaluate
from training.common.splits import default_split_path, get_or_build_split


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, help="registry name, e.g. soy")
    p.add_argument("--model", required=True, help="estimator name (see the dir's estimators)")
    p.add_argument("--traits", default="all", help="'all' or comma-separated trait names")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true",
                   help="content-hash split/cache into the run record (slower)")
    p.add_argument("--out-root", default=None, help="override runs/ root")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    # SNP-modality knobs
    p.add_argument("--impute", default="ref", choices=["ref", "mean"])
    # embedding-modality knobs (ignored by snp)
    p.add_argument("--backbone", default=None)
    p.add_argument("--half-window", type=int, default=None)
    p.add_argument("--cache", default=None, help="explicit cache path override")
    p.add_argument("--recipe", default="center_ln_mean",
                   choices=["center_ln_mean", "sum_std"])
    p.add_argument("--snp-only", action="store_true")


def resolve_traits(spec, arg: str) -> list[str]:
    allc = spec.resolved_trait_cols()
    if arg == "all":
        return allc
    want = [t.strip() for t in arg.split(",") if t.strip()]
    bad = [t for t in want if t not in allc]
    if bad:
        raise SystemExit(f"[{spec.name}] unknown traits {bad}; available: {allc}")
    return want


def _build_features(spec, modality: str, samples: list[str], args):
    """Return (X aligned to samples, cache_path_or_None)."""
    if modality == "snp":
        X, _ = feat.snp_matrix(spec, samples, impute=args.impute)
        return X, None
    if modality == "emb":
        if not args.backbone or args.half_window is None:
            raise SystemExit("emb modality needs --backbone and --half-window")
        X = feat.pooled_embeddings(spec, args.backbone, args.half_window, samples,
                                   recipe=args.recipe, snp_only=args.snp_only,
                                   cache_path=args.cache)
        cache = args.cache or spec.cache_path(args.backbone, args.half_window,
                                              snp_only=args.snp_only)
        return X, cache
    raise ValueError(f"unknown modality {modality!r}")


def run_sklearn(modality: str, make_estimator: Callable[[], object], args) -> dict:
    """Fit `args.model` (one trait at a time) over `modality` features; save results."""
    spec = get_dataset(args.dataset)
    split = get_or_build_split(args.dataset, seed=args.seed)
    traits = resolve_traits(spec, args.traits)
    samples = split.sample_ids

    X, cache_path = _build_features(spec, modality, samples, args)
    tcol = {t: i for i, t in enumerate(split.trait_cols)}
    Y = split.targets[:, [tcol[t] for t in traits]]           # (n, T) aligned to samples

    tr = split.indices("train", samples)
    va = split.indices("val", samples)
    te = split.indices("test", samples)
    Xtr, Xva, Xte = X[tr], X[va], X[te]

    yhat_val = np.full((len(va), len(traits)), np.nan)
    yhat_test = np.full((len(te), len(traits)), np.nan)
    fitted, selected = {}, {}
    for i, trait in enumerate(traits):
        ytr = Y[tr, i]
        m = ~np.isnan(ytr)
        if m.sum() < 2:
            print(f"  [skip] {trait}: <2 training labels")
            continue
        est = make_estimator()
        est.fit(Xtr[m], ytr[m])
        fitted[trait] = est
        selected[trait] = getattr(est, "best_params_", {})
        yhat_val[:, i] = est.predict(Xva)
        yhat_test[:, i] = est.predict(Xte)
        pv = _safe_pearson(Y[va, i], yhat_val[:, i])
        pt = _safe_pearson(Y[te, i], yhat_test[:, i])
        print(f"  {trait:10s} best={selected[trait]}  val_r={pv:.3f}  test_r={pt:.3f}")

    metrics = {"val": evaluate(Y[va], yhat_val, traits),
               "test": evaluate(Y[te], yhat_test, traits)}

    rec = run_record.build(
        dataset=args.dataset, features=modality, model=args.model, seed=args.seed,
        traits=traits, hyperparams=selected, metrics=metrics,
        backbone=getattr(args, "backbone", None),
        half_window=getattr(args, "half_window", None),
        split_path=str(default_split_path(args.dataset, args.seed)),
        cache_path=cache_path, strict=args.strict,
    )
    rd = artifacts.run_dir(args.dataset, modality, args.model,
                           backbone=getattr(args, "backbone", None),
                           run_id=rec.run_id, root=args.out_root)
    artifacts.save_pickle(rd / "model.pkl", {"estimators": fitted, "traits": traits})
    run_record.write(rec, rd)

    vm = metrics["val"].get("mean", {}).get("pearson", float("nan"))
    tm = metrics["test"].get("mean", {}).get("pearson", float("nan"))
    print(f"\n[{args.dataset}/{modality}/{args.model}] run_id={rec.run_id}"
          f"  mean val_r={vm:.3f}  test_r={tm:.3f}\n  → {rd}")
    return {"run_id": rec.run_id, "run_dir": str(rd), "metrics": metrics}


def _safe_pearson(yt, yp) -> float:
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    m = ~np.isnan(yt) & ~np.isnan(yp)
    if m.sum() < 2 or yt[m].std() == 0 or yp[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(yt[m], yp[m])[0, 1])

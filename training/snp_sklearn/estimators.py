"""
training/snp_sklearn/estimators.py
----------------------------------
The sklearn estimator suite, shared by snp_sklearn and emb_sklearn (same models,
different feature matrix). Each ``make_estimator`` returns a fresh, tuned
estimator: a ``Pipeline`` (scale → optional dim-reduction → model) wrapped in
``GridSearchCV`` scored by Pearson correlation over a seeded KFold on the train
partition. The harness records ``best_params_`` per trait.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ridge = RR-BLUP-style linear; svr/krr = RBF-kernel heads; rf/gbm = trees; pls = latent-factor.
MODELS = ("ridge", "svr", "krr", "rf", "gbm", "pls")


def add_sklearn_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--svd", type=int, default=0,
                   help="TruncatedSVD components before the model (0 = off)")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)


def _pearson(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if y.std() == 0 or yhat.std() == 0:
        return 0.0
    return float(np.corrcoef(y, yhat)[0, 1])


PEARSON = "pearson", _pearson


def _cv(args):
    return KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)


def _prefix(args) -> list:
    """Shared leading pipeline steps.

    Dense features: standardize, then optional SVD.
    Sparse features (--sparse): TruncatedSVD FIRST (it consumes the CSR directly and
    densifies it), then standardize the reduced factors. Requires --svd, since SVD is
    what makes the sparse matrix usable by the downstream (dense) model.
    """
    if getattr(args, "sparse", False):
        if not getattr(args, "svd", 0) or args.svd <= 0:
            raise SystemExit("--sparse requires --svd N (TruncatedSVD reduces the sparse SNP matrix).")
        return [("svd", TruncatedSVD(n_components=args.svd, random_state=args.seed)),
                ("scale", StandardScaler())]
    steps = [("scale", StandardScaler())]
    if getattr(args, "svd", 0) and args.svd > 0:
        steps.append(("svd", TruncatedSVD(n_components=args.svd, random_state=args.seed)))
    return steps


def _kernel_prefix(args) -> list:
    """Prefix for RBF kernel models (svr/krr). RBF is scale-sensitive, so the final
    features MUST be unit-variance -- SVD factors have decreasing variance, so append a
    StandardScaler after SVD (kernel distances otherwise blow up -> near-constant kernel)."""
    pre = _prefix(args)
    if not pre or pre[-1][0] != "scale":
        pre = pre + [("scale2", StandardScaler())]
    return pre


def make_estimator(name: str, args):
    from sklearn.metrics import make_scorer
    scorer = make_scorer(_pearson, greater_is_better=True)
    cv = _cv(args)
    nj = args.n_jobs

    if name == "ridge":
        pipe = Pipeline(_prefix(args) + [("model", Ridge())])
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    elif name == "svr":
        pipe = Pipeline(_kernel_prefix(args) + [("model", SVR(kernel="rbf"))])
        grid = {"model__C": [1.0, 10.0, 100.0], "model__gamma": ["scale", "auto"]}
    elif name == "krr":
        # RBF kernel-ridge head (kernel RR-BLUP): closed-form, good for small/medium n.
        pipe = Pipeline(_kernel_prefix(args) + [("model", KernelRidge(kernel="rbf"))])
        grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0],
                "model__gamma": [None, 1e-3, 1e-2, 1e-1]}
    elif name == "rf":
        pipe = Pipeline(_prefix(args) + [("model", RandomForestRegressor(
            random_state=args.seed, n_jobs=nj))])
        grid = {"model__n_estimators": [300, 600], "model__max_depth": [None, 10, 20]}
    elif name == "gbm":
        pipe = Pipeline(_prefix(args) + [("model", GradientBoostingRegressor(
            random_state=args.seed))])
        grid = {"model__n_estimators": [200, 400], "model__max_depth": [2, 3],
                "model__learning_rate": [0.05, 0.1]}
    elif name == "pls":
        # PLS does its own centering/scaling, so no prefix on dense input. On sparse
        # input it needs a dense matrix, so reduce with SVD first (requires --svd).
        if getattr(args, "sparse", False):
            if not getattr(args, "svd", 0) or args.svd <= 0:
                raise SystemExit("--sparse requires --svd N (PLS needs a dense matrix).")
            pre = [("svd", TruncatedSVD(n_components=args.svd, random_state=args.seed))]
        else:
            pre = []
        pipe = Pipeline(pre + [("model", PLSRegression())])
        grid = {"model__n_components": [2, 5, 10, 20]}
    else:
        raise SystemExit(f"unknown model {name!r}; choices: {MODELS}")

    return GridSearchCV(pipe, grid, scoring=scorer, cv=cv, n_jobs=nj, refit=True)

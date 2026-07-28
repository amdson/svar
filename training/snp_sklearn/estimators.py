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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

MODELS = ("ridge", "svr", "rf", "gbm", "pls")


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
    """Shared leading pipeline steps: standardize, then optional SVD."""
    steps = [("scale", StandardScaler())]
    if args.svd and args.svd > 0:
        steps.append(("svd", TruncatedSVD(n_components=args.svd, random_state=args.seed)))
    return steps


def make_estimator(name: str, args):
    from sklearn.metrics import make_scorer
    scorer = make_scorer(_pearson, greater_is_better=True)
    cv = _cv(args)
    nj = args.n_jobs

    if name == "ridge":
        pipe = Pipeline(_prefix(args) + [("model", Ridge())])
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    elif name == "svr":
        pipe = Pipeline(_prefix(args) + [("model", SVR(kernel="rbf"))])
        grid = {"model__C": [1.0, 10.0, 100.0], "model__gamma": ["scale", "auto"]}
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
        # PLS does its own centering/scaling; no SVD prefix.
        pipe = Pipeline([("model", PLSRegression())])
        grid = {"model__n_components": [2, 5, 10, 20]}
    else:
        raise SystemExit(f"unknown model {name!r}; choices: {MODELS}")

    return GridSearchCV(pipe, grid, scoring=scorer, cv=cv, n_jobs=nj, refit=True)

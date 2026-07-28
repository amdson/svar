"""
training/common/metrics.py
--------------------------
NaN-masked per-trait regression metrics, shared by every runner so results are
directly comparable. Missing targets (NaN) are dropped per trait before scoring.
"""
from __future__ import annotations

import numpy as np


def _pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[m], y_pred[m]


def trait_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """{pearson, r2, mse, mae, n} for one trait, NaN-masked."""
    yt, yp = _pair(y_true, y_pred)
    n = int(yt.size)
    if n < 2:
        return {"pearson": float("nan"), "r2": float("nan"),
                "mse": float("nan"), "mae": float("nan"), "n": n}
    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float("nan") if sst == 0 else 1.0 - float(np.sum((yt - yp) ** 2)) / sst
    if yt.std() == 0 or yp.std() == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    return {"pearson": pearson, "r2": r2, "mse": mse, "mae": mae, "n": n}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, trait_cols: list[str]) -> dict:
    """Per-trait metrics for a (n, T) block (or single-trait 1-D arrays).

    Returns {trait: {...}} plus a 'mean' entry averaging pearson/r2/mse/mae over
    traits that have >=2 scored samples."""
    yt = np.atleast_2d(np.asarray(y_true, dtype=float))
    yp = np.atleast_2d(np.asarray(y_pred, dtype=float))
    if yt.shape[0] == 1 and len(trait_cols) > 1:      # was (T,) not (1,T)
        yt, yp = yt.reshape(-1, len(trait_cols)), yp.reshape(-1, len(trait_cols))
    if yt.shape[1] != len(trait_cols):
        yt, yp = yt.T, yp.T
    out = {t: trait_metrics(yt[:, i], yp[:, i]) for i, t in enumerate(trait_cols)}
    scored = [m for m in out.values() if m["n"] >= 2 and not np.isnan(m["pearson"])]
    if scored:
        out["mean"] = {k: float(np.mean([m[k] for m in scored]))
                       for k in ("pearson", "r2", "mse", "mae")}
    return out

"""
training/sweeps/soy_carbon3b_baselines.py
-----------------------------------------
The classical sklearn suite on the soy Carbon-3B *pooled embeddings* — the 3B
analog of soy_carbon500m_baselines.py. Motivated by the 500M result: RBF kernels
(krr 0.683, svr 0.679) on the pooled embeddings beat the trained NN heads, so the
fair "best 3B embedding model" is the RBF kernel on the 3B cache, not the 3B head
(0.673). This closes that gap.

The pooled 3B vector is 3072-dim and dense, so — like the 500M baselines — no
sparse/SVD machinery: every model runs straight on it. Cheap set ridge/pls/svr/krr
by default; rf/gbm broken out below (append `trees` to SWEEP to include).

    python -m training.sweep --config training/sweeps/soy_carbon3b_baselines.py --dry-run
    python -m training.sweep --config training/sweeps/soy_carbon3b_baselines.py
"""
import os
from pathlib import Path

_SCRATCH = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
CACHE = f"{_SCRATCH}/caches/soy/carbon3b_hw500.ckpt.pt"

_COMMON = {"dataset": "soy", "traits": "all", "seed": 42,
           "backbone": "carbon3b", "half_window": 500,
           "cache": CACHE, "recipe": "center_ln_mean"}

# Cheap baselines on the dense 3072-dim pooled embedding (linear/latent + RBF).
cheap = {
    "runner": "emb_sklearn",
    "name": "emb3072",
    "fixed": _COMMON,
    "grid": {"model": ["ridge", "pls", "svr", "krr"]},
}

# Expensive tree ensembles — same features, slow to tune. Off by default.
trees = {
    "runner": "emb_sklearn",
    "name": "emb3072_trees",
    "fixed": _COMMON,
    "grid": {"model": ["rf", "gbm"]},
}

SWEEP = [cheap]

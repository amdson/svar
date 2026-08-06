"""
training/sweeps/soy_carbon500m_baselines.py
-------------------------------------------
The same sklearn baseline suite as soy_baselines.py (SNPs), but on the soy
Carbon-500M *pooled embeddings* — a head-to-head of classical regressors on
DNA-LM features vs. the raw SNP matrix, and vs. the trained NN heads.

Key difference from the SNP sweep: the pooled embedding is only 1024-dim and
dense, so NONE of the sparse/TruncatedSVD machinery is needed — every model
runs straight on the (n_samples, 1024) matrix. At 1024 features the RBF kernels
(svr/krr) are tractable too, so the "cheap" set is ridge/pls/svr/krr; only the
tree ensembles (rf/gbm) are genuinely expensive, so they're broken out below
and left out of the default SWEEP (add `trees` to SWEEP or `--only rf,gbm`).

Pooling recipe: center_ln_mean (center each window embedding, LayerNorm, then
mean-pool) — the analog of the pool=mean the winning NN heads used. sum_std is
the other available recipe if we want to compare pooling later.

    python -m training.sweep --config training/sweeps/soy_carbon500m_baselines.py --dry-run
    python -m training.sweep --config training/sweeps/soy_carbon500m_baselines.py
"""
import os
from pathlib import Path

_SCRATCH = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
CACHE = f"{_SCRATCH}/caches/soy/carbon500m_hw500.ckpt.pt"

_COMMON = {"dataset": "soy", "traits": "all", "seed": 42,
           "backbone": "carbon500m", "half_window": 500,
           "cache": CACHE, "recipe": "center_ln_mean"}

# Cheap baselines on the dense 1024-dim pooled embedding (linear/latent + RBF).
cheap = {
    "runner": "emb_sklearn",
    "name": "emb1024",
    "fixed": _COMMON,
    "grid": {"model": ["ridge", "pls", "svr", "krr"]},
}

# Expensive tree ensembles — same features, but slow to tune. Off by default;
# append to SWEEP or run `--only rf,gbm` to include them.
trees = {
    "runner": "emb_sklearn",
    "name": "emb1024_trees",
    "fixed": _COMMON,
    "grid": {"model": ["rf", "gbm"]},
}

SWEEP = [cheap]

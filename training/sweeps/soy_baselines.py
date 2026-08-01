"""
training/sweeps/soy_baselines.py
--------------------------------
Baseline genomic-prediction sweep on soy: every sklearn baseline model over all
11 traits, on the additive SNP matrix. Prep is chosen per model so the heavy ones
stay tractable on ~10k train samples × ~39.6k SNPs:

  * ridge (RR-BLUP) and pls  -> dense full SNP matrix (linear/latent, cheap).
  * krr/svr (RBF), rf, gbm    -> sparse -> TruncatedSVD-500 -> model
        (kernels/trees are intractable on 39.6k raw features; SVD-500 also gives
         the RBF heads a standardized, low-dim input).

Each point loops all 11 traits internally with per-trait GridSearchCV; the winning
params + val/test metrics land in the run manifest (compare via load_runs()).

    python -m training.sweep --config training/sweeps/soy_baselines.py --dry-run
    python -m training.sweep --config training/sweeps/soy_baselines.py
"""

_COMMON = {"dataset": "soy", "traits": "all", "seed": 42}

# Linear / latent baselines on the full dense SNP matrix.
dense = {
    "runner": "snp_sklearn",
    "name": "dense",
    "fixed": _COMMON,
    "grid": {"model": ["ridge", "pls"]},
}

# Kernel + tree baselines on the raw sparse matrix reduced by TruncatedSVD.
reduced = {
    "runner": "snp_sklearn",
    "name": "svd500",
    "fixed": {**_COMMON, "sparse": True, "svd": 500},
    "grid": {"model": ["krr", "svr", "rf", "gbm"]},
}

SWEEP = [dense, reduced]

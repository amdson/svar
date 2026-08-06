"""
training/sweeps/soy_carbon3b_bottleneck.py
------------------------------------------
Fight the 8-layer head's overfit (train PCC 0.94 / val 0.67 on the raw 3072-dim
3B embeddings) with dimensionality reduction, three ways, all with val
early-stopping (the last epoch is past the val peak):

  svd     — TruncatedSVD to N dims, fit on TRAIN (fixed/unsupervised), then the
            standardizer + 8-layer MLP. The exact SNP-pipeline recipe.  (--svd N)
  bneck   — a learnable Linear(3072->N) bottleneck (no activation) as the MLP's
            first layer (trainable analog of SVD); blocks run at N.  (--bottleneck-dim N)
  wd      — no reduction; sweep weight decay on the raw 3072-dim base, to compare
            a scalar L2 regularizer against dimensional bottlenecking.

Everything else at the 3B head optimum: mlp, pool=mean, warm-start, lr=3e-4,
n_layers=8, dropout=0, epochs=60 (early-stop picks the best-val epoch).
Reference bars: best head so far 0.686 (hidden 512), best overall 3B-krr 0.701.

    python -m training.sweep --config training/sweeps/soy_carbon3b_bottleneck.py --gpus 0 --dry-run
    python -m training.sweep --config training/sweeps/soy_carbon3b_bottleneck.py --gpus 0
"""
import os
from pathlib import Path

_SCRATCH = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
CACHE = f"{_SCRATCH}/caches/soy/carbon3b_hw500.ckpt.pt"

_BASE = {"dataset": "soy", "cache": CACHE, "half_window": 500,
         "head": "mlp", "pool": "mean", "center_windows": False,
         "warm_start_standardizer": True, "lr": "3e-4", "epochs": 60,
         "dropout": 0.0, "n_layers": 8, "early_stopping": True}

_OUT = "trained_heads/soy_carbon3b_bottleneck/{label}/model.pt"

# SVD reduction (fixed, unsupervised) → 8-layer MLP.
svd = {
    "runner": "emb_nn", "name": "svd", "gpu": True,
    "fixed": {**_BASE, "weight_decay": "1e-4"},
    "grid": {"svd": [50, 100, 200, 500]},
    "output_template": _OUT,
}

# Learnable linear bottleneck (trainable) → 8-layer MLP running at the bottleneck dim.
bneck = {
    "runner": "emb_nn", "name": "bneck", "gpu": True,
    "fixed": {**_BASE, "weight_decay": "1e-4"},
    "grid": {"bottleneck_dim": [50, 100, 200, 500]},
    "output_template": _OUT,
}

# Weight-decay sweep on the raw 3072-dim base (no reduction) — regularizer control.
wd = {
    "runner": "emb_nn", "name": "wd", "gpu": True,
    "fixed": _BASE,
    "grid": {"weight_decay": ["1e-4", "1e-3", "1e-2", "1e-1"]},
    "output_template": _OUT,
}

SWEEP = [svd, bneck, wd]   # 4 + 4 + 4 = 12

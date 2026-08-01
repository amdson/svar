"""
training/sweeps/soy_carbon_heads_search.py
------------------------------------------
Expanded head-model search on the soy Carbon-500M cache (the minimal 4-point
soy_carbon_heads.py was just head×pool). ~40 combinations, all head=mlp, pool=mean
(the best baseline pooling), GPU-pinned and resumable. With the VCF-skip fast path
each run is <90s, so the whole search is ~50 min on one GPU.

Split into two blocks to cover the axes without a wasteful full cartesian:
  Block A (32) — training hyperparameters on the default mlp architecture:
      center-windows (variant) × warm-start-standardizer × lr × weight_decay × epochs
  Block B (8)  — mlp architecture at a fixed good-guess training config:
      n_layers × hidden_dim × dropout

Booleans map to emb_nn flags: center_windows=True -> --center-windows (else the
plain "absolute" variant); warm_start_standardizer=True -> --warm-start-standardizer.

    python -m training.sweep --config training/sweeps/soy_carbon_heads_search.py --gpus 0 --dry-run
    python -m training.sweep --config training/sweeps/soy_carbon_heads_search.py --gpus 0
"""
import os
from pathlib import Path

_SCRATCH = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
CACHE = f"{_SCRATCH}/caches/soy/carbon500m_hw500.ckpt.pt"

_BASE = {"dataset": "soy", "cache": CACHE, "half_window": 500,
         "head": "mlp", "pool": "mean"}

# Block A — training-hyperparameter search (2×2×2×2×2 = 32), default mlp arch.
train_search = {
    "runner": "emb_nn",
    "name": "train",
    "gpu": True,
    "fixed": _BASE,
    "grid": {
        "center_windows": [False, True],           # absolute vs centered variant
        "warm_start_standardizer": [False, True],
        "lr": ["1e-3", "3e-4"],
        "weight_decay": ["1e-4", "1e-3"],
        "epochs": [30, 60],
    },
    "output_template": "trained_heads/soy_carbon500m_search/{label}/model.pt",
}

# Block B — mlp architecture search (2×2×2 = 8) at a fixed good-guess training config.
arch_search = {
    "runner": "emb_nn",
    "name": "arch",
    "gpu": True,
    "fixed": {**_BASE, "center_windows": True, "warm_start_standardizer": True,
              "lr": "1e-3", "weight_decay": "1e-4", "epochs": 30},
    "grid": {
        "n_layers": [2, 4],
        "hidden_dim": [512, 1024],
        "dropout": [0.0, 0.2],
    },
    "output_template": "trained_heads/soy_carbon500m_search/{label}/model.pt",
}

SWEEP = [train_search, arch_search]   # 32 + 8 = 40

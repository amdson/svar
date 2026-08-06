"""
training/sweeps/soy_carbon3b_heads.py
-------------------------------------
Head-model search on the soy Carbon-3B cache (carbon3b_hw500.ckpt.pt, 3072-dim).

Deliberately LEAN (10 runs, not the 40 of soy_carbon_heads_search.py). The 500M
search showed the head is capacity-saturated: once warm_start_standardizer=True,
NOTHING else moved the mean Pearson off ~0.66 — depth, width, epochs, lr, wd,
dropout, centering were all a wash, and warm_start=False was catastrophic
(head never trains). So the interesting lever here is the 3B *embeddings*, not
the head; we only spend points to (a) confirm 3B saturates the same way and
(b) probe the couple of axes that had any signal at all.

  Block tune (4) — training region at the known-good arch (mlp, dropout 0):
      lr {1e-3, 3e-4} × epochs {30, 60}
  Block arch (4) — capacity sanity (was flat at 500M; verify at 3B's 3072-dim):
      n_layers {2, 4} × hidden_dim {512, 1024}   (lr 1e-3, epochs 60)
  Block ctrl (2) — a linear-head baseline, and the warm_start=False control
      that must reproduce the 500M failure mode on 3B.

All warm_start_standardizer=True except the explicit control; pool=mean,
center_windows=False, weight_decay=1e-4, dropout=0.0 (the 500M optimum).
emb_dim is read from the cache metadata (3072 for 3B), so it is NOT set here.

    python -m training.sweep --config training/sweeps/soy_carbon3b_heads.py --gpus 0 --dry-run
    python -m training.sweep --config training/sweeps/soy_carbon3b_heads.py --gpus 0
"""
import os
from pathlib import Path

_SCRATCH = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
CACHE = f"{_SCRATCH}/caches/soy/carbon3b_hw500.ckpt.pt"

# The saturated-good region distilled from the 500M search.
_GOOD = {"dataset": "soy", "cache": CACHE, "half_window": 500,
         "head": "mlp", "pool": "mean", "center_windows": False,
         "warm_start_standardizer": True, "weight_decay": "1e-4", "dropout": 0.0}

_OUT = "trained_heads/soy_carbon3b/{label}/model.pt"

# Block tune — training region on the default mlp (dropout 0). (2×2 = 4)
tune = {
    "runner": "emb_nn", "name": "tune", "gpu": True,
    "fixed": _GOOD,
    "grid": {"lr": ["1e-3", "3e-4"], "epochs": [30, 60]},
    "output_template": _OUT,
}

# Block arch — capacity sanity at a fixed good training config. (2×2 = 4)
arch = {
    "runner": "emb_nn", "name": "arch", "gpu": True,
    "fixed": {**_GOOD, "lr": "1e-3", "epochs": 60},
    "grid": {"n_layers": [2, 4], "hidden_dim": [512, 1024]},
    "output_template": _OUT,
}

# Block ctrl — linear-head baseline (1) + the warm_start=False failure control (1).
linear = {
    "runner": "emb_nn", "name": "linear", "gpu": True,
    "fixed": {**_GOOD, "head": "linear", "lr": "1e-3", "epochs": 60},
    "grid": {"pool": ["mean"]},                      # 1-point block
    "output_template": _OUT,
}
nowarm = {
    "runner": "emb_nn", "name": "nowarm", "gpu": True,
    "fixed": {**_GOOD, "lr": "1e-3", "epochs": 60},
    "grid": {"warm_start_standardizer": [False]},    # must reproduce the 500M failure
    "output_template": _OUT,
}

SWEEP = [tune, arch, linear, nowarm]                 # 4 + 4 + 1 + 1 = 10

"""
training/sweeps/soy_carbon3b_mlp_depth.py
-----------------------------------------
Depth × dropout sweep of the 3B mlp head — probe how far n_layers and dropout can
be pushed. The earlier lean search only tried n_layers {2,4} and dropout {0,0.2};
this spans the full requested ranges (1–15 layers, 0.0–0.5 dropout) as a 3×3 grid
(9 models), holding everything else at the 3B head optimum found so far:
mlp, pool=mean, warm_start_standardizer=True, lr=3e-4, epochs=60, weight_decay=1e-4.

  n_layers {1, 8, 15} × dropout {0.0, 0.25, 0.5}   = 9 runs

Deep MLPs (15 layers) + high dropout (0.5) are a stress test — at 500M and in the
lean 3B search, capacity/regularization were flat-to-slightly-harmful, so the
expectation is this stays ≤ the shallow optimum; the point is to confirm depth
isn't a missed lever on the richer 3072-dim 3B features.

    python -m training.sweep --config training/sweeps/soy_carbon3b_mlp_depth.py --gpus 0 --dry-run
    python -m training.sweep --config training/sweeps/soy_carbon3b_mlp_depth.py --gpus 0
"""
import os
from pathlib import Path

_SCRATCH = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
CACHE = f"{_SCRATCH}/caches/soy/carbon3b_hw500.ckpt.pt"

depth = {
    "runner": "emb_nn",
    "name": "depth",
    "gpu": True,
    "fixed": {"dataset": "soy", "cache": CACHE, "half_window": 500,
              "head": "mlp", "pool": "mean", "center_windows": False,
              "warm_start_standardizer": True, "lr": "3e-4", "epochs": 60,
              "weight_decay": "1e-4"},
    "grid": {"n_layers": [1, 8, 15], "dropout": [0.0, 0.25, 0.5]},
    "output_template": "trained_heads/soy_carbon3b_depth/{label}/model.pt",
}

SWEEP = [depth]   # 3 × 3 = 9

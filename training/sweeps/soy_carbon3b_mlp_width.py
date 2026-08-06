"""
training/sweeps/soy_carbon3b_mlp_width.py
-----------------------------------------
Width sweep at the depth+dropout cell (n_layers=8, dropout=0.25) — the point that
scored 0.638 in the depth sweep, where depth was rescuing dropout. Question: does
more hidden width help there, i.e. is that cell capacity-starved rather than just
regularized? Everything else at the 3B head optimum (mlp, pool=mean, warm-start,
lr=3e-4, epochs=60, weight_decay=1e-4).

  hidden_dim {512, 1024, 2048, 4096}   = 4 runs

(default hidden_dim is emb_dim=3072; this brackets it below and above.)

    python -m training.sweep --config training/sweeps/soy_carbon3b_mlp_width.py --gpus 0 --dry-run
    python -m training.sweep --config training/sweeps/soy_carbon3b_mlp_width.py --gpus 0
"""
import os
from pathlib import Path

_SCRATCH = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
CACHE = f"{_SCRATCH}/caches/soy/carbon3b_hw500.ckpt.pt"

width = {
    "runner": "emb_nn",
    "name": "width",
    "gpu": True,
    "fixed": {"dataset": "soy", "cache": CACHE, "half_window": 500,
              "head": "mlp", "pool": "mean", "center_windows": False,
              "warm_start_standardizer": True, "lr": "3e-4", "epochs": 60,
              "weight_decay": "1e-4", "n_layers": 8, "dropout": 0.25},
    "grid": {"hidden_dim": [512, 1024, 2048, 4096]},
    "output_template": "trained_heads/soy_carbon3b_width/{label}/model.pt",
}

SWEEP = [width]   # 4 points

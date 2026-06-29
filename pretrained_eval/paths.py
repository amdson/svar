"""
pretrained_eval/paths.py
------------------------
Where pretrained-eval outputs land. Generated artifacts go to cluster scratch
(see svar/env.sh), never the home quota or git — the eval writer and the viewing
notebook both import these so they agree on one location.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def scratch_base() -> Path:
    return Path(os.environ.get("SVAR_SCRATCH", "/90daydata/small_grains/andrew.dickson"))


def output_dir() -> Path:
    """Scratch dir for per-window loss tables; created on first use."""
    d = scratch_base() / "pretrained_eval_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(model_path: str) -> str:
    """Filesystem-safe tag from a HF repo id or local path (basename only)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", Path(model_path).name).strip("-")


def per_window_csv(model_path: str, half_window: int) -> Path:
    """Canonical per-window CSV path for a (checkpoint, window) eval run."""
    return output_dir() / f"per_window_{_slug(model_path)}_hw{half_window}.csv"

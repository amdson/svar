"""
training/common/artifacts.py
----------------------------
Where a run's outputs land, and how they're saved. Run dirs live under scratch
(``$SVAR_SCRATCH/runs/<dataset>/<features>[/<backbone>]/<model>/<run_id>/``),
gitignored like trained_heads/. Splits, by contrast, are committed in-repo.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path


def runs_root() -> Path:
    scratch = os.environ.get("SVAR_SCRATCH") or str(Path.home() / "svar_scratch")
    return Path(scratch) / "runs"


def run_dir(
    dataset: str,
    features: str,
    model: str,
    *,
    backbone: str | None = None,
    run_id: str,
    root: str | Path | None = None,
) -> Path:
    base = Path(root) if root else runs_root()
    parts = [dataset, features]
    if backbone:
        parts.append(backbone)
    parts += [model, run_id]
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(path: str | Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))

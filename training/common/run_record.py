"""
training/common/run_record.py
-----------------------------
Lightweight, decoupled record of what a run actually used — so months of tuning
stay comparable. Deliberately small: a dataclass + build/write/load_runs. No
wandb, no search framework, no orchestration. It records the *final* config
(e.g. a tuned model's selected hyperparameters), never the search space.

Defaults keep light experiments frictionless: run_id, created_at and git_sha are
auto-filled; content hashing of large artifacts is OFF unless strict=True (paths
+ size/mtime are always cheap to record). Every run appends one flat row to
``$SVAR_SCRATCH/runs/index.jsonl`` for one-read downstream comparison.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _git_sha() -> tuple[str | None, bool]:
    """(short sha, dirty?) — best-effort; (None, False) if git is unavailable."""
    try:
        repo = Path(__file__).resolve().parents[2]
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True).strip())
        return sha, dirty
    except Exception:
        return None, False


def _artifact_tag(path: str | None, *, strict: bool) -> dict | None:
    """Cheap provenance for a file: path (+ size/mtime), and a content sha only if strict."""
    if not path:
        return None
    p = Path(path)
    tag: dict = {"path": str(path)}
    if p.exists():
        st = p.stat()
        tag.update(size=st.st_size, mtime=int(st.st_mtime))
        if strict:
            h = hashlib.sha1()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            tag["sha1"] = h.hexdigest()[:16]
    return tag


@dataclass
class RunRecord:
    dataset: str
    features: str                 # "snp" | "emb" | "window"
    model: str
    seed: int
    traits: list[str]
    hyperparams: dict             # final params actually used (per-trait dict allowed)
    metrics: dict                 # {"val": {...}, "test": {...}}
    backbone: str | None = None
    half_window: int | None = None
    split: dict | None = None     # artifact tag for the split file
    cache: dict | None = None     # artifact tag for the embedding cache (emb only)
    run_id: str = ""
    created_at: str = ""
    git_sha: str | None = None
    git_dirty: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, default=str)


def build(
    *,
    dataset: str,
    features: str,
    model: str,
    seed: int,
    traits: list[str],
    hyperparams: dict,
    metrics: dict,
    backbone: str | None = None,
    half_window: int | None = None,
    split_path: str | None = None,
    cache_path: str | None = None,
    strict: bool = False,
    extra: dict | None = None,
) -> RunRecord:
    """Assemble a RunRecord, auto-filling run_id (config hash), timestamp, git sha."""
    rec = RunRecord(
        dataset=dataset, features=features, model=model, seed=seed,
        traits=list(traits), hyperparams=hyperparams, metrics=metrics,
        backbone=backbone, half_window=half_window,
        split=_artifact_tag(split_path, strict=strict),
        cache=_artifact_tag(cache_path, strict=strict),
        extra=extra or {},
    )
    # run_id is a hash of the identifying config (NOT metrics/timestamp) so re-runs
    # of the same config collide → dedupable; traits/hyperparams included.
    ident = {"dataset": dataset, "features": features, "model": model, "seed": seed,
             "backbone": backbone, "half_window": half_window,
             "traits": sorted(traits), "hyperparams": hyperparams}
    rec.run_id = hashlib.sha1(_canonical(ident).encode()).hexdigest()[:10]
    rec.created_at = _dt.datetime.now().isoformat(timespec="seconds")
    rec.git_sha, rec.git_dirty = _git_sha()
    return rec


def _runs_root() -> Path:
    scratch = os.environ.get("SVAR_SCRATCH") or str(Path.home() / "svar_scratch")
    return Path(scratch) / "runs"


def manifest_path() -> Path:
    return _runs_root() / "index.jsonl"


def _flatten(rec: RunRecord) -> dict:
    """One flat row for the manifest: scalars + mean val/test metrics per trait."""
    row = {"run_id": rec.run_id, "created_at": rec.created_at, "git_sha": rec.git_sha,
           "dataset": rec.dataset, "features": rec.features, "model": rec.model,
           "seed": rec.seed, "backbone": rec.backbone, "half_window": rec.half_window,
           "n_traits": len(rec.traits)}
    for phase in ("val", "test"):
        block = rec.metrics.get(phase, {})
        mean = block.get("mean", {})
        for k in ("pearson", "r2", "mse", "mae"):
            if k in mean:
                row[f"{phase}.{k}"] = mean[k]
    return row


def write(rec: RunRecord, run_dir: str | Path) -> Path:
    """Write run.json into run_dir and append a flat row to the central manifest."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(rec.to_dict(), indent=2))
    mpath = manifest_path()
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "a") as f:
        f.write(json.dumps(_flatten(rec)) + "\n")
    return run_dir / "run.json"


def load_runs(manifest: str | Path | None = None):
    """Read the manifest into a DataFrame for downstream comparison."""
    import pandas as pd
    path = Path(manifest) if manifest else manifest_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)

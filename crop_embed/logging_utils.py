"""
logging_utils.py
----------------
Shared training-metric logging so every script (train_pipeline/*, scripts/*)
tracks runs identically.

`MetricLogger` writes a local JSONL sidecar — one JSON object per logged row,
appended and flushed so a notebook can tail it live (see
notebooks/track_training.ipynb) — and optionally mirrors each row to wandb. The
local sidecar is the primary tracking path; wandb is opt-in on top, so runs are
fully observable even when the cloud connection is flaky.

    from crop_embed.logging_utils import MetricLogger, metrics_path_for

    logger = MetricLogger(
        metrics_path_for(args.output),
        wandb_project=args.wandb_project, wandb_name=args.wandb_name,
        config=vars(args),
    )
    logger.log({"step": step, **metrics})
    ...
    logger.close()            # or use `with MetricLogger(...) as logger:`
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def metrics_path_for(output_path: str | Path) -> Path:
    """JSONL sidecar path next to a run's --output (e.g. model.pt → model.metrics.jsonl)."""
    p = Path(output_path)
    return p.with_name(p.stem + ".metrics.jsonl")


class MetricLogger:
    """
    Local-first metric logger.

    Parameters
    ----------
    metrics_path  : where to write the JSONL sidecar. Truncated on open (one log
                    per run). Parent dirs are created.
    wandb_project : if given, also init wandb and mirror every logged row to it.
    wandb_name    : optional wandb run name.
    config        : run config (e.g. vars(args)) recorded in wandb.
    """

    def __init__(
        self,
        metrics_path: str | Path,
        *,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.metrics_path, "w")

        self.use_wandb = wandb_project is not None
        if self.use_wandb:
            if not _HAS_WANDB:
                raise RuntimeError(
                    "wandb_project set but wandb is not installed; pip install wandb"
                )
            wandb.init(project=wandb_project, name=wandb_name, config=config)

    def log(self, payload: dict) -> None:
        """Append one row to the sidecar (flushed) and mirror to wandb if enabled."""
        self._file.write(json.dumps(payload) + "\n")
        self._file.flush()
        if self.use_wandb:
            wandb.log(payload)

    def close(self) -> None:
        self._file.close()
        if self.use_wandb:
            wandb.finish()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

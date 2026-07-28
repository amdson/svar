"""
training/common/splits.py
-------------------------
Precomputed **70/15/15 nested** train/val/test split, built once per dataset and
committed to ``splits/<crop>_seed<seed>.pt``. Held out by sample ID so partitions
survive reordering and are identical across every feature modality (SNP matrix,
pooled embeddings, window cache) — no cross-model leakage.

Nesting: hold out ``test`` (15%) first, then split the remaining 85% into
train/val so that ``val`` is 15% of the total (train 70%). Doing test first means
re-drawing the inner train/val split never disturbs the test set.

The split is over the **labeled** samples (those with at least one phenotype), so
train/val/test contain only trainable rows.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from training.common.datasets import DatasetSpec, get_dataset
from training.common.features import labeled_samples, load_targets


@dataclass
class Split:
    sample_ids: list[str]                 # labeled samples that were partitioned
    train_sample_ids: list[str]
    val_sample_ids: list[str]
    test_sample_ids: list[str]
    trait_cols: list[str]
    targets: np.ndarray | None = None     # (n, T), aligned to sample_ids, NaN=missing
    metadata: dict = field(default_factory=dict)

    def indices(self, which: str, order: list[str] | None = None) -> np.ndarray:
        """Row indices of partition ``which`` ('train'|'val'|'test') within ``order``
        (default: this split's ``sample_ids``)."""
        ids = {"train": self.train_sample_ids, "val": self.val_sample_ids,
               "test": self.test_sample_ids}[which]
        order = order or self.sample_ids
        pos = {s: i for i, s in enumerate(order)}
        missing = [s for s in ids if s not in pos]
        if missing:
            raise KeyError(f"{len(missing)} '{which}' ids absent from order (e.g. {missing[:3]})")
        return np.fromiter((pos[s] for s in ids), dtype=np.intp, count=len(ids))


def build_split(
    spec: DatasetSpec,
    *,
    seed: int = 42,
    test: float = 0.15,
    val: float = 0.15,
    traits: list[str] | None = None,
) -> Split:
    """Build the nested split over ``spec``'s labeled samples."""
    trait_cols = traits or spec.resolved_trait_cols()
    samples = sorted(labeled_samples(spec, trait_cols))     # sort → deterministic given seed
    n = len(samples)
    if n < 10:
        raise ValueError(f"[{spec.name}] only {n} labeled samples — too few to split")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    ordered = [samples[i] for i in perm]

    n_test = round(n * test)
    n_val = round(n * val)
    test_ids = sorted(ordered[:n_test])
    val_ids = sorted(ordered[n_test:n_test + n_val])
    train_ids = sorted(ordered[n_test + n_val:])

    Y, _ = load_targets(spec, samples, trait_cols)
    return Split(
        sample_ids=samples,
        train_sample_ids=train_ids,
        val_sample_ids=val_ids,
        test_sample_ids=test_ids,
        trait_cols=list(trait_cols),
        targets=Y,
        metadata={
            "dataset": spec.name, "seed": seed, "test": test, "val": val,
            "n_labeled": n, "n_train": len(train_ids), "n_val": len(val_ids),
            "n_test": len(test_ids),
            "generated_by": "training.common.splits.build_split",
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
    )


def save_split(path: str | Path, split: Split) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_ids": split.sample_ids,
        "train_sample_ids": split.train_sample_ids,
        "val_sample_ids": split.val_sample_ids,
        "test_sample_ids": split.test_sample_ids,
        "trait_cols": split.trait_cols,
        "targets": None if split.targets is None else torch.from_numpy(split.targets),
        "metadata": split.metadata,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_split(path: str | Path) -> Split:
    d = torch.load(path, map_location="cpu", weights_only=False)
    tgt = d.get("targets")
    return Split(
        sample_ids=d["sample_ids"],
        train_sample_ids=d["train_sample_ids"],
        val_sample_ids=d["val_sample_ids"],
        test_sample_ids=d["test_sample_ids"],
        trait_cols=d["trait_cols"],
        targets=None if tgt is None else (tgt.numpy() if hasattr(tgt, "numpy") else np.asarray(tgt)),
        metadata=d.get("metadata", {}),
    )


def default_split_path(dataset: str, seed: int = 42) -> Path:
    """Committed location: <repo>/splits/<crop>_seed<seed>.pt."""
    repo = Path(__file__).resolve().parents[2]
    return repo / "splits" / f"{dataset}_seed{seed}.pt"


def get_or_build_split(dataset: str, *, seed: int = 42, rebuild: bool = False) -> Split:
    """Load the committed split, building (and saving) it on first use."""
    path = default_split_path(dataset, seed)
    if path.exists() and not rebuild:
        return load_split(path)
    split = build_split(get_dataset(dataset), seed=seed)
    save_split(path, split)
    return split

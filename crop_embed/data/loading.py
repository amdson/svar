"""
loading.py
----------
Convenience wrappers around the VCF → dataset → phenotype pipeline that the
training scripts (demo_train.py, train_pipeline/*) all duplicate, plus a cached
train/val split so every run trains and evaluates on the *same* sample points
regardless of which embedding cache or model it uses.

    build_dataset()           VCF + FASTA + windowing      -> UniqueWindowDataset
    load_targets()            phenotypes                   -> (Y, trait_cols)
    make_split()              deterministic id-keyed split -> (train_idx, val_idx)
    save_split() / load_split() cache the split by sample ID and reload it
    prepare_data()            one call -> dataset, Y, trait_cols, train/val sets

The split is cached as *sample IDs*, not raw row indices, so it stays valid even
if the dataset's sample ordering changes — load_split() re-resolves IDs against
whatever dataset you hand it.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from crop_embed.dataset import SampleDataset, UniqueWindowDataset
from crop_embed.partitioner import SNPWindowPartitioner
from crop_embed.data.coords import FASTA_PATH
from crop_embed.data.preprocessing import (
    DEFAULT_PHENO_PATH,
    align_targets_to_dataset,
    load_phenotypes,
    scale_phenotypes,
)
from crop_embed.data.vcf import load_snps_from_vcf

DEFAULT_VCF_PATH = "/home/andrew/rice_data/sativas413_msu7_final.vcf"


def build_dataset(
    vcf_path: str = DEFAULT_VCF_PATH,
    fasta_path: str = str(FASTA_PATH),
    half_window: int = 500,
    buffer: int = 0,
    *,
    verbose: bool = True,
) -> UniqueWindowDataset:
    """
    Run the VCF → partitioner → UniqueWindowDataset pipeline.

    This is the dataset-building block that demo_train.py and generate_cache.py
    both inline; keeping it here means the windowing is defined in exactly one
    place, so a cache and a training run can't silently disagree.
    """
    if verbose:
        print("Loading SNPs from VCF …")
    snps_by_chrom, samples = load_snps_from_vcf(vcf_path)
    if verbose:
        n_snps = sum(len(v) for v in snps_by_chrom.values())
        print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {len(samples)} samples")
        print("Building SNPWindowPartitioner …")

    partitioner = SNPWindowPartitioner(
        snps_by_chrom, half_window=half_window, buffer=buffer
    )
    if verbose:
        stats = partitioner.snps_per_window_stats()
        print(f"  {stats['n_windows']:,} windows (mean {stats['mean']:.1f} SNPs/window)")
        print("Building UniqueWindowDataset …")

    dataset = UniqueWindowDataset(vcf_path, fasta_path, partitioner)
    if verbose:
        print(f"  {len(dataset):,} unique windows; {len(dataset.samples)} samples")
    return dataset


def load_targets(
    dataset: UniqueWindowDataset,
    pheno_path: str | Path = DEFAULT_PHENO_PATH,
    *,
    verbose: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """
    Load phenotypes, align them to `dataset.samples`, z-score per trait (NaN-safe),
    and return (Y, trait_cols). Y is FloatTensor[n_samples, n_traits]; NaN = missing.
    """
    if verbose:
        print(f"Loading phenotypes from {pheno_path} …")
    pheno_df, trait_cols = load_phenotypes(pheno_path)
    Y_np = align_targets_to_dataset(dataset, pheno_df, trait_cols)
    Y_np = scale_phenotypes(Y_np)                       # per-trait z-score, NaN-safe
    Y = torch.tensor(Y_np, dtype=torch.float32)
    if verbose:
        n_with_pheno = (~torch.isnan(Y).all(dim=1)).sum().item()
        print(f"  {Y.shape[1]} traits; {n_with_pheno}/{Y.shape[0]} samples have phenotype data")
    return Y, list(trait_cols)


def make_split(
    sample_ids: Sequence[str],
    test_split: float = 0.30,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic train/val split over `sample_ids`, returning LongTensors of
    *row indices into sample_ids*.

    The permutation is keyed on (seed, sorted sample IDs) rather than the given
    order, so the same set of IDs always lands in the same partition even if the
    dataset hands them to us in a different order.
    """
    order = sorted(range(len(sample_ids)), key=lambda i: sample_ids[i])
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(order), generator=rng)
    shuffled = torch.tensor(order, dtype=torch.long)[perm]

    n_val = round(len(sample_ids) * test_split)
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return train_idx, val_idx

def save_split(
    path: str | Path,
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    trait_cols: Sequence[str] | None = None,
    targets: torch.Tensor | None = None,
    sample_ids: Sequence[str] | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Cache a train/val split (and optionally the aligned targets) to disk.

    Stored by sample ID so it can be reloaded against any dataset built from the
    same VCF. If `targets` is given, `sample_ids` must be its row order so it can
    be re-aligned on load.
    """
    if targets is not None and sample_ids is None:
        raise ValueError("save_split: `sample_ids` is required when `targets` is given")

    payload: dict = {
        "train_sample_ids": list(train_ids),
        "val_sample_ids": list(val_ids),
        "trait_cols": list(trait_cols) if trait_cols is not None else None,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "crop_embed.data.loading.save_split",
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            **(metadata or {}),
        },
    }
    if targets is not None:
        payload["targets"] = targets
        payload["sample_ids"] = list(sample_ids)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_split(
    path: str | Path,
    dataset: UniqueWindowDataset,
    *,
    require_all: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reload a cached split and resolve it against `dataset.samples`, returning
    (train_idx, val_idx) as LongTensors of row indices into dataset.samples.

    With `require_all=True` (default) every cached ID must be present in the
    dataset — a missing ID means the split and dataset don't match and we'd
    silently train on a different population, so we raise instead.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    id_to_idx = {sid: i for i, sid in enumerate(dataset.samples)}

    def resolve(ids: Sequence[str], name: str) -> torch.Tensor:
        missing = [s for s in ids if s not in id_to_idx]
        if missing and require_all:
            raise ValueError(
                f"load_split: {len(missing)} {name} sample(s) from {path} are "
                f"not in the dataset (e.g. {missing[:3]}). Split/dataset mismatch."
            )
        return torch.tensor([id_to_idx[s] for s in ids if s in id_to_idx], dtype=torch.long)

    return resolve(payload["train_sample_ids"], "train"), resolve(payload["val_sample_ids"], "val")


def prepare_data(
    *,
    vcf_path: str = DEFAULT_VCF_PATH,
    fasta_path: str = str(FASTA_PATH),
    pheno_path: str | Path = DEFAULT_PHENO_PATH,
    half_window: int = 500,
    buffer: int = 0,
    split_path: str | Path | None = None,
    test_split: float = 0.30,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    One-call setup: build the dataset, load targets, and produce train/val
    SampleDatasets using either a cached split (`split_path`) or a fresh
    deterministic one.

    Returns a dict with keys: dataset, Y, trait_cols, train_idx, val_idx,
    train_dataset, val_dataset.
    """
    dataset = build_dataset(vcf_path, fasta_path, half_window, buffer, verbose=verbose)
    Y, trait_cols = load_targets(dataset, pheno_path, verbose=verbose)

    if split_path is not None:
        if not Path(split_path).exists():
            raise FileNotFoundError(
                f"prepare_data: split file {split_path} not found. Generate it first "
                f"with scripts/cache_split.py so every run shares the same train/val "
                f"samples."
            )
        train_idx, val_idx = load_split(split_path, dataset)
        if verbose:
            print(f"Loaded cached split from {split_path}: "
                  f"{len(train_idx)} train / {len(val_idx)} val")
    else:
        train_idx, val_idx = make_split(dataset.samples, test_split, seed)
        if verbose:
            print(f"Built fresh split: {len(train_idx)} train / {len(val_idx)} val "
                  f"(seed={seed}, test_split={test_split})")

    return {
        "dataset": dataset,
        "Y": Y,
        "trait_cols": trait_cols,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_dataset": SampleDataset(dataset, Y, train_idx),
        "val_dataset": SampleDataset(dataset, Y, val_idx),
    }

    
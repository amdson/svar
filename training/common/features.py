"""
training/common/features.py
---------------------------
The feature-modality boundary. Every runner gets its data through here, so
"SNP matrix vs pooled embeddings vs window cache" is the only thing that differs
between model families — everything downstream sees an ``(n_samples, features)``
matrix (or a window bundle) plus targets, all aligned to a requested sample order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training.common.datasets import DatasetSpec


# ── targets ──────────────────────────────────────────────────────────────────
def load_targets(
    spec: DatasetSpec,
    samples: list[str],
    traits: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """(Y (n, T) float32, trait_cols) — pheno CSV reindexed onto ``samples`` by IID.
    Samples with no phenotype row, or a blank trait value, become NaN (masked in loss/metrics)."""
    if not Path(spec.pheno_csv).exists():
        raise FileNotFoundError(f"[{spec.name}] phenotype CSV not found: {spec.pheno_csv}")
    traits = traits or spec.resolved_trait_cols()
    df = pd.read_csv(spec.pheno_csv).set_index("IID")
    missing_cols = [t for t in traits if t not in df.columns]
    if missing_cols:
        raise KeyError(f"[{spec.name}] traits not in {spec.pheno_csv}: {missing_cols}")
    Y = df.reindex(samples)[traits].apply(pd.to_numeric, errors="coerce")
    return Y.to_numpy(dtype=np.float32), list(traits)


def scaled_targets(
    spec: DatasetSpec,
    samples: list[str],
    traits: list[str] | None = None,
):
    """Like load_targets but per-trait z-scored (NaN-safe) — the target form the NN
    head path uses (matches crop_embed.data.loading.load_targets)."""
    from crop_embed.data.preprocessing import scale_phenotypes
    Y, cols = load_targets(spec, samples, traits)
    return scale_phenotypes(Y), cols


def build_window_dataset(spec: DatasetSpec, half_window: int = 500, buffer: int = 0,
                         *, verbose: bool = True):
    """VCF → partitioner → UniqueWindowDataset for this dataset (reuses the exact
    crop_embed builder, so dataset.samples order matches the legacy path)."""
    from crop_embed.data.loading import build_dataset
    if not spec.vcf_path or not Path(spec.vcf_path).exists():
        raise FileNotFoundError(f"[{spec.name}] VCF not found: {spec.vcf_path}")
    return build_dataset(spec.vcf_path, spec.fasta_path, half_window, buffer, verbose=verbose)


def labeled_samples(spec: DatasetSpec, traits: list[str] | None = None) -> list[str]:
    """Canonical samples that have at least one non-NaN target — the supervised set."""
    samples = spec.samples()
    Y, _ = load_targets(spec, samples, traits)
    keep = ~np.all(np.isnan(Y), axis=1)
    return [s for s, k in zip(samples, keep) if k]


# ── modality: snp ────────────────────────────────────────────────────────────
def snp_matrix(
    spec: DatasetSpec,
    samples: list[str],
    *,
    impute: str = "ref",
) -> tuple[np.ndarray, list[str]]:
    """(X (n, V) additive 0/1/2 dosage, variant_ids) from the pgen fileset."""
    from crop_embed.data.genotype_matrix import load_dosage_matrix
    X, _, variant_ids = load_dosage_matrix(spec.pgen_prefix, samples=samples, impute=impute)
    return X, variant_ids


def snp_matrix_sparse(
    spec: DatasetSpec,
    samples: list[str],
) -> tuple["object", list[str]]:
    """(X (n, V) additive-dosage CSR, variant_ids) — the raw sparse genotype matrix.

    Missing calls are treated as reference (0), so the reference-homozygous majority
    stays structurally zero and the matrix is genuinely sparse. Feed straight into
    TruncatedSVD (which operates on sparse input) for the classical
    sparse-SNP -> SVD -> model pipeline; see training/snp_sklearn/estimators.py.
    """
    from scipy import sparse
    from crop_embed.data.genotype_matrix import load_dosage_matrix
    # ref-fill keeps zeros structural; int8 dense first, then compress.
    X, _, variant_ids = load_dosage_matrix(
        spec.pgen_prefix, samples=samples, impute="ref", dtype=np.int8)
    return sparse.csr_matrix(X, dtype=np.float32), variant_ids


# ── modality: emb (pooled per-sample) ────────────────────────────────────────
def pooled_embeddings(
    spec: DatasetSpec,
    backbone: str,
    half_window: int,
    samples: list[str],
    *,
    recipe: str = "center_ln_mean",
    snp_only: bool = False,
    cache_path: str | None = None,
) -> np.ndarray:
    """X (n, D) — window-embedding cache pooled to one vector per sample, aligned to ``samples``."""
    from presentation.pool_cache import per_sample_matrix
    path = cache_path or spec.cache_path(backbone, half_window, snp_only=snp_only)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[{spec.name}] embedding cache not found: {path}\n"
            f"  generate it first (train_pipeline/embed_windows.py) or pass --cache."
        )
    X_all, cache_ids = per_sample_matrix(path, recipe)
    pos = {s: i for i, s in enumerate(cache_ids)}
    missing = [s for s in samples if s not in pos]
    if missing:
        raise KeyError(
            f"[{spec.name}] {len(missing)} requested samples absent from cache {path} "
            f"(e.g. {missing[:3]}); cache was built on a different sample set."
        )
    return X_all[[pos[s] for s in samples]]


# ── modality: window (per-window cache, for NN heads / e2e) ──────────────────
def window_cache(
    spec: DatasetSpec,
    backbone: str,
    half_window: int,
    *,
    snp_only: bool = False,
    cache_path: str | None = None,
):
    """Return (FixedWindowEmbedder, sample_ids) for heads that pool internally."""
    from crop_embed import FixedWindowEmbedder
    import torch
    path = cache_path or spec.cache_path(backbone, half_window, snp_only=snp_only)
    if not Path(path).exists():
        raise FileNotFoundError(f"[{spec.name}] embedding cache not found: {path}")
    emb = FixedWindowEmbedder.from_file(path)
    sample_ids = torch.load(path, map_location="cpu", weights_only=False).get("sample_ids")
    return emb, sample_ids

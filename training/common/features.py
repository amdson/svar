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
    # IID always as string: some datasets (wheat GIDs) have purely-numeric IDs that
    # pandas would otherwise read as int64 and fail to align to the string sample IDs.
    df = pd.read_csv(spec.pheno_csv, dtype={"IID": str}).set_index("IID")
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


class CachedWindows:
    """The subset of ``UniqueWindowDataset`` the head trainer actually reads, loaded
    straight from an embedding cache (.ckpt.pt) — no VCF/FASTA. Mirrors the attributes
    the trainer touches (``samples``, ``sample_fp_index``, ``unique_fingerprints``) plus
    the embedding ``cache`` and ``metadata``, so downstream code is unchanged."""

    def __init__(self, samples, sample_fp_index, unique_fingerprints, cache, metadata):
        self.samples = samples
        self.sample_fp_index = sample_fp_index
        self.unique_fingerprints = unique_fingerprints
        self.cache = cache
        self.metadata = metadata


def load_cached_windows(cache_path: str) -> "CachedWindows | None":
    """Load everything the head trainer needs directly from a window-embedding cache,
    skipping the (slow) VCF windowing. Returns None for legacy caches that predate the
    ``sample_ids`` key, so the caller can fall back to ``build_window_dataset``."""
    import torch
    obj = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict)
            and {"cache", "sample_fp_index", "sample_ids"} <= set(obj)):
        return None
    return CachedWindows(
        samples=list(obj["sample_ids"]),
        sample_fp_index=obj["sample_fp_index"],
        unique_fingerprints=obj.get("unique_fingerprints"),
        cache=obj["cache"].float(),
        metadata=obj.get("metadata", {}),
    )


def labeled_samples(spec: DatasetSpec, traits: list[str] | None = None) -> list[str]:
    """Canonical samples that have at least one non-NaN target — the supervised set."""
    samples = spec.samples()
    Y, _ = load_targets(spec, samples, traits)
    keep = ~np.all(np.isnan(Y), axis=1)
    return [s for s, k in zip(samples, keep) if k]


# ── SNP importance prior (per-SNP feature rescaling) ─────────────────────────
def load_snp_prior(prior) -> dict[str, float]:
    """Normalize a prior spec into a {variant_id: weight} dict.

    Accepts: a path to a torch-saved file ``{"variant_ids": [...], "weights":
    [...]}``, that same dict directly, a plain ``{variant_id: weight}`` mapping,
    or a ``(variant_ids, weights)`` pair. Weights are raw importances (any
    non-negative scale); normalization happens in ``apply_snp_prior``."""
    if isinstance(prior, (str, Path)):
        import torch
        prior = torch.load(prior, map_location="cpu", weights_only=False)
    if isinstance(prior, tuple) and len(prior) == 2:
        ids, w = prior
        return {str(i): float(x) for i, x in zip(ids, w)}
    if isinstance(prior, dict) and "variant_ids" in prior and "weights" in prior:
        w = prior["weights"]
        w = w.tolist() if hasattr(w, "tolist") else list(w)
        return {str(i): float(x) for i, x in zip(prior["variant_ids"], w)}
    if isinstance(prior, dict):
        return {str(k): float(v) for k, v in prior.items()}
    raise TypeError(f"unrecognized snp prior spec: {type(prior)!r}")


def resolve_prior_weights(variant_ids: list[str], prior) -> np.ndarray:
    """Per-SNP prior weight aligned to ``variant_ids`` (float64, mean-normalized to 1).

    Present weights are normalized to mean 1; variants absent from the prior default
    to 1.0 (neutral). This is the ``w_j`` a downstream reweighting applies as a
    per-SNP prior variance — see ``apply_snp_prior`` (raw-column form) and the
    post-standardization pipeline step the harness actually uses."""
    id2w = load_snp_prior(prior)
    w = np.array([id2w.get(v, np.nan) for v in variant_ids], dtype=np.float64)
    present = ~np.isnan(w)
    n_missing = int((~present).sum())
    if present.any():
        w[present] /= w[present].mean()          # mean-normalize the covered SNPs to 1
    w[~present] = 1.0                             # uncovered SNPs: neutral (no reweight)
    if n_missing:
        print(f"  [snp-prior] {n_missing}/{len(variant_ids)} variants absent from prior "
              f"→ weight 1.0")
    return w


def apply_snp_prior(X, variant_ids: list[str], prior):
    """Rescale each SNP column by ``sqrt(weight)`` — the raw-column form of the prior.

    NOTE: a no-op in front of a per-column StandardScaler (z-scoring divides the
    scale right back out). The harness therefore applies the prior *after*
    standardization inside the estimator pipeline instead; this helper is for
    standalone/manual use on already-standardized (or tree/SVD) features. Handles
    dense arrays and scipy CSR."""
    from scipy import sparse
    scale = np.sqrt(resolve_prior_weights(variant_ids, prior)).astype(np.float32)
    if sparse.issparse(X):
        return (X @ sparse.diags(scale)).tocsr()
    return X * scale[None, :]


def variant_window_index(
    spec: DatasetSpec, half_window: int, *, buffer: int = 0
) -> dict[str, int]:
    """Map each variant_id → its partitioner window index (the cache's window axis).

    Rebuilds the SNPWindowPartitioner for this dataset's VCF at (half_window,
    buffer); its window order is deterministic (chrom then greedy left-to-right), so
    it matches a cache built with the same windowing column-for-column. Used to turn
    a per-window score (e.g. a Carbon window-conservation score, in window order)
    into a per-variant prior via ``snp_prior_from_window_scores``."""
    from crop_embed.data.vcf import load_snps_from_vcf
    from crop_embed.partitioner import SNPWindowPartitioner
    if not spec.vcf_path or not Path(spec.vcf_path).exists():
        raise FileNotFoundError(f"[{spec.name}] VCF not found: {spec.vcf_path}")
    snps_by_chrom, _ = load_snps_from_vcf(spec.vcf_path, None)
    part = SNPWindowPartitioner(snps_by_chrom, half_window=half_window, buffer=buffer)
    # (chrom, pos, variant_id) from the .pvar; assign via the partitioner's map.
    vid_to_win: dict[str, int] = {}
    pvar = spec.pgen_prefix + ".pvar"
    with open(pvar) as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom_s, pos_s, vid = line.split("\t", 3)[:3]
            if not chrom_s.isdigit():
                continue                          # skip scaffolds / non-integer chroms
            # .pvar POS is 1-based (plink); the partitioner keys on SNPRecord.pos,
            # which is pysam's 0-based rec.start. Convert before the lookup.
            win = part.snp_to_window_idx.get((int(chrom_s), int(pos_s) - 1))
            if win is not None:
                vid_to_win[vid] = win
    return vid_to_win


def snp_prior_from_window_scores(
    spec: DatasetSpec, half_window: int, window_scores, *, buffer: int = 0
) -> dict:
    """Build a per-variant prior from a per-window score array (window order).

    ``window_scores[j]`` is the importance of window j; every SNP assigned to that
    window inherits it. Returns ``{"variant_ids": [...], "weights": [...]}`` ready to
    torch.save and hand to ``--snp-prior``."""
    window_scores = np.asarray(window_scores, dtype=np.float64)
    vid_to_win = variant_window_index(spec, half_window, buffer=buffer)
    ids, weights = [], []
    for vid, win in vid_to_win.items():
        if 0 <= win < len(window_scores):
            ids.append(vid)
            weights.append(float(window_scores[win]))
    return {"variant_ids": ids, "weights": np.asarray(weights, dtype=np.float32)}


# ── modality: snp ────────────────────────────────────────────────────────────
def snp_matrix(
    spec: DatasetSpec,
    samples: list[str],
    *,
    impute: str = "ref",
    prior=None,
) -> tuple[np.ndarray, list[str]]:
    """(X (n, V) additive 0/1/2 dosage, variant_ids) from the pgen fileset.

    ``prior`` (path or {variant_id: weight}) rescales each SNP column by sqrt(weight)
    — a Carbon-derived per-SNP prior variance; see ``apply_snp_prior``."""
    from crop_embed.data.genotype_matrix import load_dosage_matrix
    X, _, variant_ids = load_dosage_matrix(spec.pgen_prefix, samples=samples, impute=impute)
    if prior is not None:
        X = apply_snp_prior(X, variant_ids, prior)
    return X, variant_ids


def snp_matrix_sparse(
    spec: DatasetSpec,
    samples: list[str],
    *,
    prior=None,
) -> tuple["object", list[str]]:
    """(X (n, V) additive-dosage CSR, variant_ids) — the raw sparse genotype matrix.

    Missing calls are treated as reference (0), so the reference-homozygous majority
    stays structurally zero and the matrix is genuinely sparse. Feed straight into
    TruncatedSVD (which operates on sparse input) for the classical
    sparse-SNP -> SVD -> model pipeline; see training/snp_sklearn/estimators.py.

    ``prior`` rescales columns by sqrt(weight) (a diagonal scale keeps it sparse).
    """
    from scipy import sparse
    from crop_embed.data.genotype_matrix import load_dosage_matrix
    # ref-fill keeps zeros structural; int8 dense first, then compress.
    X, _, variant_ids = load_dosage_matrix(
        spec.pgen_prefix, samples=samples, impute="ref", dtype=np.int8)
    X = sparse.csr_matrix(X, dtype=np.float32)
    if prior is not None:
        X = apply_snp_prior(X, variant_ids, prior)
    return X, variant_ids


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

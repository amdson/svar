"""
presentation/pool_cache.py
--------------------------
Goal 3 (helper). Turn a window-embedding cache (.ckpt.pt from embed_windows.py /
embed_windows_vc.py) into a per-sample matrix (n_samples, D) under one of two
aggregation recipes, plus inferred-subpopulation labels for coloring.

Recipes (mirroring train_pipeline/train_head.py):
  "sum_std"        : sum-pool windows, then z-score columns ("summed and
                     standardized").
  "center_ln_mean" : per-window center + parameter-free LayerNorm, then mean-pool
                     ("centered + normed -> average"; the --center-ln-pool path).

The cache file already stores `cache`, `sample_fp_index`, and `sample_ids`, so we
read it directly with FixedWindowEmbedder.from_file (no VCF/dataset rebuild). For
an unsupervised visualization we standardize/center over ALL samples (there's no
label leakage to worry about in a UMAP), so no train/val split is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed import FixedWindowEmbedder


def per_sample_matrix(cache_path: str, recipe: str,
                      chunk: int = 16) -> tuple[np.ndarray, list[str]]:
    """Return (X (n_samples, D) float32 ndarray, sample_ids)."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    sample_ids = payload.get("sample_ids")
    emb = FixedWindowEmbedder.from_file(cache_path)
    cache = emb.cache.float()             # (n_fps, D)
    sfi = emb.sample_fp_index             # (n_samples, n_windows)
    if sample_ids is None:
        sample_ids = [str(i) for i in range(sfi.shape[0])]

    if recipe == "sum_std":
        X = F.embedding_bag(sfi, cache, mode="sum")            # (n_samples, D)
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
        X = (X - mu) / sd
        return X.numpy(), sample_ids

    if recipe == "center_ln_mean":
        # per-window across-sample mean (all samples), then param-free LN, mean-pool.
        # Done in sample-chunks so (n_samples, n_windows, D) never materializes whole.
        n_samples, n_windows = sfi.shape
        # window mean over samples: average the gathered window vectors across rows.
        wsum = torch.zeros(n_windows, cache.shape[1])
        for i in range(0, n_samples, chunk):
            wsum += cache[sfi[i:i + chunk]].sum(0)
        wmean = wsum / n_samples                               # (n_windows, D)
        parts = []
        for i in range(0, n_samples, chunk):
            g = cache[sfi[i:i + chunk]] - wmean                # (b, n_windows, D)
            g = F.layer_norm(g, (g.shape[-1],))
            parts.append(g.mean(dim=1))                        # (b, D)
        X = torch.cat(parts, dim=0)
        return X.numpy(), sample_ids

    raise ValueError(f"unknown recipe: {recipe!r} (use 'sum_std' or 'center_ln_mean')")


def subpop_labels(sample_ids: list[str], n_pops: int = 4) -> np.ndarray:
    """Inferred subpopulation per sample: quartile bins on SNP-matrix PC1.

    Follows notebooks/snp_emb_subpop_analysis.ipynb. Returns an int array aligned
    to `sample_ids` (value -1 for any sample missing from the VCF matrix).
    """
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from crop_embed.data.preprocessing import load_vcf_sparse

    X_snp, vcf_samples, _, _ = load_vcf_sparse()
    pc1 = TruncatedSVD(n_components=10, random_state=42).fit_transform(X_snp)[:, 0]
    pops = pd.qcut(pc1, n_pops, labels=False)
    samp_to_pop = dict(zip(vcf_samples, pops))
    return np.array([int(samp_to_pop.get(s, -1)) for s in sample_ids])

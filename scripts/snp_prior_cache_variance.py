"""
scripts/snp_prior_cache_variance.py
-----------------------------------
Carbon-derived per-SNP importance prior, source #3: **population embedding
variance** per window.

For each window j, take that window's embedding across every sample —
``cache[sample_fp_index[:, j]]`` — and measure how much the population's observed
genotypic variation moves Carbon's representation there (total variance = mean
squared deviation from the window's mean embedding). A window whose genotypes
strongly perturb the embedding is one where the assayed variation carries real
functional signal, so its SNPs get up-weighted.

Allele-observed (unlike the static window-LM score) and confounded by allele
frequency, but needs NO new GPU embedding — just an existing cache. Writes both
the per-window scores (.npy, reusable for transforms) and the per-variant prior
(.pt) consumed by ``--snp-prior``.

    python -m scripts.snp_prior_cache_variance \
        --cache $SVAR_SCRATCH/caches/soy/carbon500m_hw500.ckpt.pt \
        --dataset soy --out $SVAR_SCRATCH/caches/soy/snp_prior_var_hw500_500m.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.common import features as feat
from training.common.datasets import get_dataset


def per_window_variance(cache: torch.Tensor, sample_fp_index: torch.Tensor,
                        device: str, chunk: int = 16) -> torch.Tensor:
    """Total embedding variance per window across all samples, streamed in sample
    chunks so the (chunk, n_windows, D) gather never fully materializes.

    var_j = (1/N) Σ_s ||x_{s,j}||^2  −  ||(1/N) Σ_s x_{s,j}||^2   (sum of per-dim var)."""
    cache = cache.to(device)
    n_samples, n_windows = sample_fp_index.shape
    D = cache.shape[1]
    sum_vec = torch.zeros(n_windows, D, device=device, dtype=torch.float64)
    sumsq = torch.zeros(n_windows, device=device, dtype=torch.float64)      # Σ_s ||x||^2
    for i in range(0, n_samples, chunk):
        idx = sample_fp_index[i:i + chunk].to(device)                      # (b, n_windows)
        g = cache[idx].double()                                            # (b, n_windows, D)
        sum_vec += g.sum(0)
        sumsq += (g * g).sum(dim=(0, 2))
        if (i // chunk) % 100 == 0:
            print(f"  variance: {i}/{n_samples} samples", flush=True)
    mean = sum_vec / n_samples                                             # (n_windows, D)
    var = sumsq / n_samples - (mean * mean).sum(dim=1)                     # (n_windows,)
    return var.clamp_min(0).cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="embedding cache (.ckpt.pt).")
    p.add_argument("--dataset", default="soy", help="registry dataset for the VCF/pvar (SNP→window map).")
    p.add_argument("--buffer", type=int, default=0, help="windowing buffer (must match the cache).")
    p.add_argument("--out", required=True, help="output prior file (.pt).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk", type=int, default=16, help="samples per streaming chunk.")
    args = p.parse_args()

    print(f"Loading cache {args.cache} …", flush=True)
    cw = feat.load_cached_windows(args.cache)
    if cw is None:
        raise SystemExit("cache lacks sample_ids/sample_fp_index (legacy); cannot score.")
    half_window = cw.metadata.get("half_window")
    print(f"  {cw.cache.shape[0]:,} fps × {cw.cache.shape[1]} dims; "
          f"{cw.sample_fp_index.shape[0]} samples × {cw.sample_fp_index.shape[1]} windows; "
          f"half_window={half_window} buffer={args.buffer}", flush=True)

    scores = per_window_variance(cw.cache, cw.sample_fp_index, args.device, args.chunk).numpy()
    print(f"  per-window variance: min={scores.min():.4g} median={np.median(scores):.4g} "
          f"max={scores.max():.4g}", flush=True)

    spec = get_dataset(args.dataset)
    print("Mapping SNPs → windows (rebuild partitioner from VCF) …", flush=True)
    prior = feat.snp_prior_from_window_scores(spec, half_window, scores, buffer=args.buffer)
    n_cov = len(prior["variant_ids"])
    print(f"  prior covers {n_cov} variants", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**prior,
                "source": "cache_variance", "cache": args.cache,
                "half_window": half_window, "buffer": args.buffer}, out)
    np.save(out.with_suffix(".windowscores.npy"), scores)
    print(f"Saved prior → {out}\nSaved window scores → {out.with_suffix('.windowscores.npy')}",
          flush=True)


if __name__ == "__main__":
    main()

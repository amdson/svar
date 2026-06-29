"""
presentation/umap_preprocessing.py
----------------------------------
Goal 3 (process + plot). UMAP (or t-SNE) of per-sample embeddings under the
different preprocessing options, colored by inferred subpopulation:

  * all vs snp-only windows           -> different cache files
  * Carbon vs DNABERT-2               -> different cache files
  * "centered + normed -> average" vs "summed and standardized"  -> recipe

Panels are auto-skipped if their cache file is missing, so this runs against
whatever caches you have built (4 Carbon caches exist by default; add DNABERT-2
caches with embed_windows.py --backend dnabert2 to populate those panels).

    python presentation/umap_preprocessing.py                 # -> figs/03_umap_grid.png
    python presentation/umap_preprocessing.py --reducer tsne
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from presentation.pool_cache import per_sample_matrix, subpop_labels

CK = "checkpoints/manual"

# (title, cache_path, recipe)
PANELS = [
    ("Carbon all / sum+std",        f"{CK}/sativas413_carbon500m_hw500.ckpt.pt",          "sum_std"),
    ("Carbon all / center+ln",      f"{CK}/sativas413_carbon500m_hw500.ckpt.pt",          "center_ln_mean"),
    ("Carbon snp-only / sum+std",   f"{CK}/sativas413_carbon500m_hw500_snponly.ckpt.pt",  "sum_std"),
    ("Carbon snp-only / center+ln", f"{CK}/sativas413_carbon500m_hw500_snponly.ckpt.pt",  "center_ln_mean"),
    ("Carbon-VC all / sum+std",     f"{CK}/sativas413_carbon500m_vc_hw500.ckpt.pt",       "sum_std"),
    ("Carbon-VC snp-only / sum+std",f"{CK}/sativas413_carbon500m_vc_hw500_snponly.ckpt.pt","sum_std"),
    # DNABERT-2 panels (built only if you run embed_windows.py --backend dnabert2):
    ("DNABERT2 all / sum+std",      f"{CK}/sativas413_dnabert2_hw500.ckpt.pt",            "sum_std"),
    ("DNABERT2 snp-only / sum+std", f"{CK}/sativas413_dnabert2_hw500_snponly.ckpt.pt",    "sum_std"),
]


def reduce_2d(X: np.ndarray, kind: str, seed: int) -> np.ndarray:
    if kind == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed,
                         n_neighbors=15, min_dist=0.05).fit_transform(X)
    if kind == "tsne":
        from sklearn.manifold import TSNE
        perp = min(30, max(5, (X.shape[0] - 1) // 3))
        return TSNE(n_components=2, perplexity=perp,
                    random_state=seed, init="pca").fit_transform(X)
    raise ValueError(kind)


def main() -> int:
    p = argparse.ArgumentParser(description="UMAP/t-SNE grid over preprocessing options.")
    p.add_argument("--reducer", choices=["umap", "tsne"], default="umap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    available = [(t, c, r) for (t, c, r) in PANELS if Path(c).exists()]
    missing = [t for (t, c, r) in PANELS if not Path(c).exists()]
    if missing:
        print("Skipping panels with no cache on disk:")
        for t in missing:
            print(f"  - {t}")
    if not available:
        print("No caches found under checkpoints/manual/. Build them first.")
        return 1

    ncol = min(4, len(available))
    nrow = (len(available) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow),
                             squeeze=False)
    axes = axes.ravel()

    sc = None
    for ax, (title, cache, recipe) in zip(axes, available):
        print(f"[{args.reducer}] {title}  ({cache}, {recipe})")
        X, sample_ids = per_sample_matrix(cache, recipe)
        coords = reduce_2d(X, args.reducer, args.seed)
        pops = subpop_labels(sample_ids)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=pops, cmap="tab10",
                        vmin=0, vmax=int(pops.max()), s=16, alpha=.85, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(available):]:
        ax.axis("off")

    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), label="inferred subpop (SNP PC1 quartile)",
                     fraction=0.025, pad=0.01)
    fig.suptitle(f"{args.reducer.upper()} of per-sample embeddings — preprocessing options",
                 fontsize=13)

    out = args.out or str(HERE / "figs" / f"03_{args.reducer}_grid.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

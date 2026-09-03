"""
Cis-window characterization for the Kremling maize panel — the maize
counterpart of `probe_expression_windows.py` (arabidopsis), same columns so the
two tables read side by side. The numbers every training-budget decision
depends on, at hmp321 MAF>0.05 density with NO LD pruning:

  * SNPs per TSS-centred window and cache size cs (unique 6-mer token
    positions; nearby SNPs share a token slot)
  * cs/T, which bounds the cache's per-haplotype compute advantage
  * unique cis-genotype patterns among the train-split lines (the dedup
    factor, and the N that actually hits the GPU per gene) — expect ~1 per
    line at maize density, unlike rice
  * residual missing/het accounting: the genotypes are KNN-imputed, so any
    -9 here means the imputation claim is wrong — this doubles as a check

Reads gene/line metadata from kremling_dataset.h5 and genotypes from the
plink2 .pgen. CPU only, a few minutes.

    python -m model_dev.probe_kremling_windows --n-genes 1000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from model_dev.kremling_elasticnet_baseline import DATA, H5, PFILE, load_pvar_fast

TOKEN = 6  # Carbon 6-mer tokenization; window tokens T = 2*half_window // 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=1000)
    ap.add_argument("--half-windows", default="4000,12000,24000")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    half_windows = [int(x) for x in args.half_windows.split(",")]

    import h5py
    import pgenlib

    with h5py.File(H5, "r") as f:
        chrom = f["genes/chrom"][:].astype(str)
        tss = f["genes/tss"][:]
        taxa = f["lines/taxa"][:].astype(str)
        line_split = f["lines/line_split"][:].astype(str)

    psam_ids = [l.split()[0] for l in open(PFILE + ".psam")
                if not l.startswith("#")]
    col = {s: i for i, s in enumerate(psam_ids)}
    panel_cols = np.array([col[t] for t in taxa])
    train_cols = np.array([col[t] for t in taxa[line_split == "train"]])
    print(f"panel {len(panel_cols)}/{len(psam_ids)} pgen samples; "
          f"{len(train_cols)} train lines")

    v_chrom, v_pos = load_pvar_fast(PFILE + ".pvar")
    by_chrom = {c: np.flatnonzero(v_chrom == c) for c in np.unique(v_chrom)}
    print(f"{len(v_pos):,} variants across chroms "
          f"{ {c: len(ix) for c, ix in sorted(by_chrom.items(), key=lambda kv: int(kv[0]))} }")

    reader = pgenlib.PgenReader(str(PFILE + ".pgen").encode())
    n_samples = reader.get_raw_sample_ct()
    assert n_samples == len(psam_ids)

    rng = np.random.default_rng(args.seed)
    gene_ix = rng.choice(len(tss), size=min(args.n_genes, len(tss)),
                         replace=False)

    stats = {hw: [] for hw in half_windows}
    hw_max = max(half_windows)
    for gi in gene_ix:
        ix = by_chrom.get(chrom[gi])
        if ix is None:
            continue
        pos_c = v_pos[ix]
        a, b = np.searchsorted(pos_c, [tss[gi] - hw_max, tss[gi] + hw_max])
        if b <= a:
            for hw in half_windows:
                stats[hw].append((0, 0, 0.0, 1, 0.0, 0.0))
            continue
        # one pgen read at the widest window serves all half-widths
        geno = np.empty((b - a, n_samples), dtype=np.int8)
        reader.read_range(int(ix[a]), int(ix[b - 1]) + 1, geno)
        pos_w = pos_c[a:b]

        for hw in half_windows:
            m = (pos_w >= tss[gi] - hw) & (pos_w < tss[gi] + hw)
            g = geno[m][:, panel_cols]           # (S, lines) 0/1/2, -9 missing
            S = g.shape[0]
            if S == 0:
                stats[hw].append((0, 0, 0.0, 1, 0.0, 0.0))
                continue
            tok = np.unique((pos_w[m] - (tss[gi] - hw)) // TOKEN)
            T = 2 * hw // TOKEN
            miss = float((g == -9).mean())       # KNN-imputed -> should be 0
            het = float((g == 1).mean())
            gt = geno[m][:, train_cols]
            gt = np.where(gt == -9, 0, gt)
            n_uniq = np.unique(gt.T, axis=0).shape[0]
            stats[hw].append((S, len(tok), len(tok) / T, n_uniq, miss, het))

    n_train = len(train_cols)
    print(f"\nsampled {len(gene_ix)} genes (seed {args.seed}); "
          f"dedup is over the {n_train} train lines")
    print(f"{'hw':>6} {'T':>5} | {'SNPs p50':>8} {'p90':>6} | {'cs p50':>6} "
          f"{'p90':>6} | {'cs/T p50':>8} {'p90':>6} | {'uniq-line p50':>13} "
          f"{'p90':>6} {'max':>5} | {'miss%':>6} {'het%':>6} | {'0-SNP%':>6}")
    for hw in half_windows:
        arr = np.array(stats[hw])
        S, cs, frac, uniq, miss, het = arr.T
        q = lambda v, p: np.percentile(v, p)
        print(f"{hw:>6} {2*hw//TOKEN:>5} | {q(S,50):>8.0f} {q(S,90):>6.0f} | "
              f"{q(cs,50):>6.0f} {q(cs,90):>6.0f} | {q(frac,50):>8.3f} "
              f"{q(frac,90):>6.3f} | {q(uniq,50):>13.0f} {q(uniq,90):>6.0f} "
              f"{uniq.max():>5.0f} | {100*miss.mean():>6.2f} "
              f"{100*het.mean():>6.2f} | {100*(S==0).mean():>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

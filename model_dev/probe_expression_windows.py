"""
Go/no-go characterization: what does a cis expression window actually look like
for the variant cache, at 1001G density with NO LD pruning?

For a random sample of genes (TSS-centred windows at several half-widths, using
the exact accession panel and train split of the expression dataset), measures
the numbers every design constant depends on:

  * SNPs per window and cache size cs (unique 6-mer token positions — the
    full SNP set is cached; multi-SNP tokens share one cache slot)
  * cs/T, which bounds the cache's compute advantage per haplotype
  * unique cis-haplotypes among the 531 train accessions (the dedup factor,
    and the N that actually hits the GPU per gene)
  * missing-call rate inside the window (masking policy input)

Reads genotypes from the plink2 .pgen (pgenlib) — the whole panel's genotype
matrix for a window in one call — and gene/accession metadata from the
expression HDF5. CPU only, a few minutes.

    python -m model_dev.probe_expression_windows --n-genes 1000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

DATA = "/90daydata/small_grains/andrew.dickson/datasets/arabidopsis"
H5 = f"{DATA}/expression/expression_dataset.h5"
PFILE = f"{DATA}/arabidopsis_1001g_final"

TOKEN = 6  # Carbon 6-mer tokenization; window tokens T = 2*half_window // 6


def load_pvar(path):
    chroms, poss = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 2)
            chroms.append(f[0])
            poss.append(int(f[1]))
    return np.array(chroms), np.array(poss, dtype=np.int64)


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
        eco = f["accessions/ecotype_id"][:].astype(str)
        split = f["accessions/acc_split"][:].astype(str)
        chrom = f["genes/chrom"][:].astype(str)
        tss = f["genes/tss"][:]

    psam_ids = [l.split()[0] for l in open(PFILE + ".psam") if not l.startswith("#")]
    col = {s: i for i, s in enumerate(psam_ids)}
    train_cols = np.array([col[e] for e in eco[split == "train"]])
    panel_cols = np.array([col[e] for e in eco])
    print(f"panel {len(panel_cols)}/{len(psam_ids)} VCF samples; "
          f"{len(train_cols)} train accessions")

    v_chrom, v_pos = load_pvar(PFILE + ".pvar")
    # variants are sorted within chromosome; build per-chrom position arrays
    by_chrom = {c: np.flatnonzero(v_chrom == c) for c in np.unique(v_chrom)}
    print(f"{len(v_pos):,} variants across chroms "
          f"{ {c: len(ix) for c, ix in sorted(by_chrom.items())} }")

    rng = np.random.default_rng(args.seed)
    gene_ix = rng.choice(len(tss), size=min(args.n_genes, len(tss)),
                         replace=False)

    reader = pgenlib.PgenReader(str(PFILE + ".pgen").encode())
    n_samples = reader.get_raw_sample_ct()
    assert n_samples == len(psam_ids)

    stats = {hw: [] for hw in half_windows}
    hw_max = max(half_windows)
    for gi in gene_ix:
        c = chrom[gi]
        ix = by_chrom.get(c)
        if ix is None:
            continue
        pos_c = v_pos[ix]
        w_lo, w_hi = tss[gi] - hw_max, tss[gi] + hw_max
        a, b = np.searchsorted(pos_c, [w_lo, w_hi])
        if b <= a:
            for hw in half_windows:
                stats[hw].append((0, 0, 0.0, 1, 0.0))
            continue
        # one pgen read at the widest window serves all half-widths
        lo_v, hi_v = int(ix[a]), int(ix[b - 1]) + 1
        geno = np.empty((hi_v - lo_v, n_samples), dtype=np.int8)
        reader.read_range(lo_v, hi_v, geno)
        pos_w = pos_c[a:b]

        for hw in half_windows:
            m = (pos_w >= tss[gi] - hw) & (pos_w < tss[gi] + hw)
            g = geno[m][:, panel_cols]           # (S, 665) 0/1/2, -9 missing
            S = g.shape[0]
            if S == 0:
                stats[hw].append((0, 0, 0.0, 1, 0.0))
                continue
            tok = np.unique((pos_w[m] - (tss[gi] - hw)) // TOKEN)
            T = 2 * hw // TOKEN
            miss = float((g == -9).mean())
            # dedup among train accessions: unique genotype columns
            gt = geno[m][:, train_cols]
            gt = np.where(gt == -9, 0, gt)       # missing → ref, as vcf.py does
            n_uniq = np.unique(gt.T, axis=0).shape[0]
            stats[hw].append((S, len(tok), len(tok) / T, n_uniq, miss))

    print(f"\nsampled {len(gene_ix)} genes (seed {args.seed})")
    print(f"{'hw':>6} {'T':>5} | {'SNPs p50':>8} {'p90':>6} | {'cs p50':>6} "
          f"{'p90':>6} | {'cs/T p50':>8} {'p90':>6} | {'uniq-hap p50':>12} "
          f"{'p90':>6} {'max':>5} | {'miss%':>6} | {'0-SNP%':>6}")
    for hw in half_windows:
        arr = np.array(stats[hw])
        S, cs, frac, uniq, miss = arr.T
        q = lambda v, p: np.percentile(v, p)
        print(f"{hw:>6} {2*hw//TOKEN:>5} | {q(S,50):>8.0f} {q(S,90):>6.0f} | "
              f"{q(cs,50):>6.0f} {q(cs,90):>6.0f} | {q(frac,50):>8.3f} "
              f"{q(frac,90):>6.3f} | {q(uniq,50):>12.0f} {q(uniq,90):>6.0f} "
              f"{uniq.max():>5.0f} | {100*miss.mean():>6.2f} | "
              f"{100*(S==0).mean():>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

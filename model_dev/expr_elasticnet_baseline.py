"""
Go/no-go baseline: the elasticnet bar for cis expression prediction, on exactly
the inputs the variant-cache model would see.

Per gene: X = cis-SNP genotypes (0/1 alt-carrier, full SNP set, TSS ± hw —
matched to the cache's window, NOT PrediXcan's ±1 Mb), y = expression deviation
(log2 minus panel mean, the h5's `deviation`). Fit ElasticNetCV on train
accessions, score R² on val accessions. Also reports:

  * the dedup ceiling: accessions with byte-identical cis-genotypes must get
    identical predictions from ANY cis-genotype model (elasticnet or the
    cache), so 1 − (within-duplicate-group variance / total variance) on val
    is the structural ceiling on R² — the cis analogue of h². Only informative
    where duplicate groups exist.
  * how many genes have any cis signal at all (R² > 0 on val): if elasticnet
    finds nothing, there is nothing for a fine-tuned LM to beat, and the
    interesting question becomes whether pretraining finds signal elasticnet
    misses — a different experiment.

Gate (BENCHMARK.md §7 spirit): the LM comparison is only worth running on a
panel where this script shows a real distribution of positive per-gene R².

    python -m model_dev.expr_elasticnet_baseline --n-genes 300 --half-window 24000
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

DATA = "/90daydata/small_grains/andrew.dickson/datasets/arabidopsis"
H5 = f"{DATA}/expression/expression_dataset.h5"
PFILE = f"{DATA}/arabidopsis_1001g_final"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=300)
    ap.add_argument("--half-window", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maf-min", type=float, default=0.0,
                    help="drop cis-SNPs below this MAF in the panel (0 = keep all)")
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    import h5py
    import pgenlib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from model_dev.probe_expression_windows import load_pvar

    with h5py.File(H5, "r") as f:
        eco = f["accessions/ecotype_id"][:].astype(str)
        split = f["accessions/acc_split"][:].astype(str)
        chrom = f["genes/chrom"][:].astype(str)
        tss = f["genes/tss"][:]
        gene_id = f["genes/gene_id"][:].astype(str)
        dev = f["deviation"][:]  # (genes, 665)

    psam_ids = [l.split()[0] for l in open(PFILE + ".psam") if not l.startswith("#")]
    col = {s: i for i, s in enumerate(psam_ids)}
    tr, va = split == "train", split == "val"
    tr_cols = np.array([col[e] for e in eco[tr]])
    va_cols = np.array([col[e] for e in eco[va]])
    print(f"{tr.sum()} train / {va.sum()} val accessions; hw={args.half_window}")

    v_chrom, v_pos = load_pvar(PFILE + ".pvar")
    by_chrom = {c: np.flatnonzero(v_chrom == c) for c in np.unique(v_chrom)}

    reader = pgenlib.PgenReader(str(PFILE + ".pgen").encode())
    n_samples = reader.get_raw_sample_ct()

    rng = np.random.default_rng(args.seed)
    gene_ix = rng.choice(len(tss), size=min(args.n_genes, len(tss)),
                         replace=False)

    rows = []
    for k, gi in enumerate(gene_ix):
        ix = by_chrom[chrom[gi]]
        pos_c = v_pos[ix]
        a, b = np.searchsorted(pos_c, [tss[gi] - args.half_window,
                                       tss[gi] + args.half_window])
        y_tr, y_va = dev[gi, tr], dev[gi, va]
        if b <= a:
            rows.append((gene_id[gi], 0, np.nan, np.nan))
            continue
        geno = np.empty((b - a, n_samples), dtype=np.int8)
        reader.read_range(int(ix[a]), int(ix[b - 1]) + 1, geno)
        alt = (geno > 0)                            # missing (-9) → ref, like vcf.py
        X_tr = alt[:, tr_cols].T.astype(np.float32)  # (n_train, S)
        X_va = alt[:, va_cols].T.astype(np.float32)
        if args.maf_min > 0:
            p = X_tr.mean(0)
            keep = np.minimum(p, 1 - p) >= args.maf_min
            X_tr, X_va = X_tr[:, keep], X_va[:, keep]
        if X_tr.shape[1] == 0:
            rows.append((gene_id[gi], 0, np.nan, np.nan))
            continue

        enet = ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=5, max_iter=2000,
                            n_jobs=-1, random_state=0)
        enet.fit(X_tr, y_tr)
        pred = enet.predict(X_va)
        ss_res = ((y_va - pred) ** 2).sum()
        ss_tot = ((y_va - y_va.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # dedup ceiling on val: identical cis-genotype ⇒ identical prediction
        _, grp = np.unique(alt[:, va_cols].T, axis=0, return_inverse=True)
        gmean = np.zeros(grp.max() + 1)
        for g in range(grp.max() + 1):
            gmean[g] = y_va[grp == g].mean()
        ceil = 1 - ((y_va - gmean[grp]) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

        rows.append((gene_id[gi], X_tr.shape[1], r2, ceil))
        if (k + 1) % 25 == 0:
            done = [r for r in rows if np.isfinite(r[2])]
            print(f"  {k+1}/{len(gene_ix)} genes; median val R² so far "
                  f"{np.median([r[2] for r in done]):.4f}")

    r2s = np.array([r[2] for r in rows], dtype=float)
    ok = np.isfinite(r2s)
    print(f"\n{ok.sum()}/{len(rows)} genes scored "
          f"(rest had no cis-SNPs in window)")
    for name, v in [("median", np.median(r2s[ok])),
                    ("mean", r2s[ok].mean()),
                    ("p90", np.percentile(r2s[ok], 90)),
                    ("max", r2s[ok].max()),
                    ("% genes R² > 0.00", 100 * (r2s[ok] > 0).mean()),
                    ("% genes R² > 0.05", 100 * (r2s[ok] > 0.05).mean()),
                    ("% genes R² > 0.20", 100 * (r2s[ok] > 0.20).mean())]:
        print(f"  val R² {name:>18}: {v:.4f}")

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["gene_id", "n_cis_snps", "val_r2", "dedup_ceiling_r2"])
            wr.writerows(rows)
        print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

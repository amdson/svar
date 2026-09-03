"""
Elasticnet bar for the Kremling maize panel — the sanity check the variant-cache
model has to beat, on exactly the inputs it would see.

Per gene (one tissue at a time): X = cis-SNP alt-carrier genotypes in TSS +- hw
(full SNP set, no LD pruning), y = expression deviation (log2 FPM minus the
per-(tissue, gene) panel mean). Fit ElasticNetCV on train lines, score R^2 on
val lines. Also reports the dedup ceiling (lines with identical cis genotypes
must get identical predictions from ANY cis model).

Default hw=4000: Li et al. 2024 (PhytoExpr) report ~70% of maize TFBSs and
~50% of lead cis-eQTL SNPs within 4 kb of the TSS/TTS, and small windows keep
this fast (~300 variants/window at hmp321 MAF>0.05 density).

    python -m model_dev.kremling_elasticnet_baseline --n-genes 300 --tissue GShoot
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

DATA = "/90daydata/small_grains/andrew.dickson/datasets/maize_kremling"
H5 = f"{DATA}/expression/kremling_dataset.h5"
PFILE = f"{DATA}/kremling_agpv3"


def load_pvar_fast(path):
    """chrom (str) and pos arrays from a .pvar, fast via pandas."""
    import pandas as pd
    n_header = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                n_header += 1
            else:
                break
    df = pd.read_csv(path, sep="\t", skiprows=n_header, header=None,
                     usecols=[0, 1], names=["chrom", "pos"],
                     dtype={"chrom": str, "pos": np.int64})
    return df["chrom"].to_numpy(), df["pos"].to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=300)
    ap.add_argument("--half-window", type=int, default=4000)
    ap.add_argument("--tissue", default="GShoot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gene-split", default="train",
                    help="pos_split stratum to sample genes from (accession "
                    "axis is the held-out axis; keep test-chrom genes unseen)")
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    import h5py
    import pgenlib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    with h5py.File(H5, "r") as f:
        tissues = list(f["tissues"][:].astype(str))
        t = tissues.index(args.tissue)
        dev = f["deviation"][t]                    # (genes, lines)
        chrom = f["genes/chrom"][:].astype(str)
        tss = f["genes/tss"][:]
        gene_id = f["genes/gene_id"][:].astype(str)
        pos_split = f["genes/pos_split"][:].astype(str)
        taxa = f["lines/taxa"][:].astype(str)
        line_split = f["lines/line_split"][:].astype(str)

    psam_ids = [l.split()[0] for l in open(PFILE + ".psam")
                if not l.startswith("#")]
    col = {s: i for i, s in enumerate(psam_ids)}
    cols = np.array([col[x] for x in taxa])        # h5 line order -> pgen column

    has = ~np.isnan(dev)
    tr = (line_split == "train")
    va = (line_split == "val")
    print(f"tissue {args.tissue}: {has.any(0).sum()} lines with data "
          f"({(tr & has.any(0)).sum()} train, {(va & has.any(0)).sum()} val); "
          f"hw={args.half_window}")

    v_chrom, v_pos = load_pvar_fast(PFILE + ".pvar")
    by_chrom = {c: np.flatnonzero(v_chrom == c) for c in np.unique(v_chrom)}
    reader = pgenlib.PgenReader(str(PFILE + ".pgen").encode())
    n_samples = reader.get_raw_sample_ct()

    rng = np.random.default_rng(args.seed)
    pool = np.flatnonzero((pos_split == args.gene_split) & (has.sum(1) > 100))
    gene_ix = rng.choice(pool, size=min(args.n_genes, len(pool)), replace=False)
    print(f"gene pool {len(pool):,} ({args.gene_split} chroms, >100 lines); "
          f"sampled {len(gene_ix)}")

    rows = []
    for k, gi in enumerate(gene_ix):
        ix = by_chrom[chrom[gi]]
        pos_c = v_pos[ix]
        a, b = np.searchsorted(pos_c, [tss[gi] - args.half_window,
                                       tss[gi] + args.half_window])
        y = dev[gi]
        m_tr = tr & ~np.isnan(y)
        m_va = va & ~np.isnan(y)
        # Skip degenerate targets: a gene with ~constant deviation in either
        # split (e.g. unexpressed panel-wide) makes R^2 unbounded/meaningless.
        if (b <= a or m_tr.sum() < 50 or m_va.sum() < 10
                or np.nanvar(y[m_tr]) < 1e-4 or np.nanvar(y[m_va]) < 1e-4):
            rows.append((gene_id[gi], max(b - a, 0), np.nan, np.nan))
            continue
        geno = np.empty((b - a, n_samples), dtype=np.int8)
        reader.read_range(int(ix[a]), int(ix[b - 1]) + 1, geno)
        alt = (geno[:, cols] > 0)                  # (S, lines) alt carrier
        X_tr = alt[:, m_tr].T.astype(np.float32)
        X_va = alt[:, m_va].T.astype(np.float32)
        y_tr, y_va = y[m_tr], y[m_va]

        enet = ElasticNetCV(l1_ratio=0.5, alphas=20, cv=5, max_iter=2000,
                            n_jobs=-1, random_state=0)
        enet.fit(X_tr, y_tr)
        pred = enet.predict(X_va)
        ss_tot = ((y_va - y_va.mean()) ** 2).sum()
        r2 = 1 - ((y_va - pred) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

        _, grp = np.unique(alt[:, m_va].T, axis=0, return_inverse=True)
        gmean = np.array([y_va[grp == g].mean() for g in range(grp.max() + 1)])
        ceil = 1 - ((y_va - gmean[grp]) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

        rows.append((gene_id[gi], b - a, r2, ceil))
        if (k + 1) % 25 == 0:
            done = [r[2] for r in rows if np.isfinite(r[2])]
            print(f"  {k+1}/{len(gene_ix)}; median val R2 so far "
                  f"{np.median(done):.4f}  (median cis SNPs "
                  f"{int(np.median([r[1] for r in rows]))})")

    r2s = np.array([r[2] for r in rows], dtype=float)
    snps = np.array([r[1] for r in rows], dtype=float)
    ok = np.isfinite(r2s)
    print(f"\n{ok.sum()}/{len(rows)} genes scored; cis SNPs/window "
          f"p50={np.median(snps):.0f} p90={np.percentile(snps, 90):.0f}")
    for name, v in [("median", np.median(r2s[ok])),
                    ("mean", r2s[ok].mean()),
                    ("p90", np.percentile(r2s[ok], 90)),
                    ("max", r2s[ok].max()),
                    ("% genes R2 > 0.00", 100 * (r2s[ok] > 0).mean()),
                    ("% genes R2 > 0.05", 100 * (r2s[ok] > 0.05).mean()),
                    ("% genes R2 > 0.20", 100 * (r2s[ok] > 0.20).mean())]:
        print(f"  val R2 {name:>18}: {v:.4f}")

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

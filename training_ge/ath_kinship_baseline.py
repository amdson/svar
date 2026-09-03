"""
Arabidopsis expression sanity control: how much of the deviation target is
predictable from *relatedness alone* (GBLUP on the genome-wide kinship
matrix), with no sequence at all?

This is the confound the SIEVE data cannot have (random mutations) and 1001G
data is soaked in: relatives share expression for trans/background reasons, so
a cis-window model evaluated on random accession splits can look good by
implicitly matching haplotypes. Any cache fine-tune on arabidopsis must be
read against this bar — beat it, or show complementary signal (mixture).

Per gene g: standardize deviation by TRAIN accession mean/sd, then predict
held-out accessions via GBLUP:  yhat_v = K_vt (K_tt + lambda I)^-1 y_t.
K_tt is shared across genes, so one factorization per lambda serves all
22,611 genes at once. Reports pooled pearson/R^2 on the val accessions per
lambda (test untouched).

    python -m training_ge.ath_kinship_baseline
"""
from __future__ import annotations

import sys

import numpy as np

D = "/90daydata/small_grains/andrew.dickson/datasets/arabidopsis/expression/"


def main() -> int:
    import h5py

    K = np.load(D + "baselines/grm.npy")
    with h5py.File(D + "expression_dataset.h5", "r") as f:
        dev = f["deviation"][:]                    # (genes, 665)
        split = f["accessions/acc_split"][:].astype(str)
    tr, va = split == "train", split == "val"
    print(f"{tr.sum()} train / {va.sum()} val accessions; {dev.shape[0]:,} genes")

    mu = dev[:, tr].mean(axis=1, keepdims=True)
    sd = dev[:, tr].std(axis=1, ddof=1, keepdims=True)
    ok = (sd[:, 0] > 1e-3)
    Y = (dev - mu) / np.where(sd > 1e-3, sd, 1.0)
    Y_tr, Y_va = Y[ok][:, tr], Y[ok][:, va]
    print(f"{ok.sum():,} scoreable genes (train-sd > 1e-3)")

    K_tt = K[np.ix_(tr, tr)]
    K_vt = K[np.ix_(va, tr)]
    n_t = K_tt.shape[0]

    print(f"{'lambda':>8} {'pooled pearson':>15} {'pooled R2':>10}")
    for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
        A = K_vt @ np.linalg.solve(K_tt + lam * np.eye(n_t), np.eye(n_t))
        P = Y_tr @ A.T                            # (genes, n_val)
        y, p = Y_va.ravel(), P.ravel()
        r = np.corrcoef(p, y)[0, 1]
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"{lam:>8.1f} {r:>15.4f} {r2:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

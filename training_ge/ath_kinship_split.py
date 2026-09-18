"""
Kinship-level accession split for arabidopsis expression.

The committed `accessions/acc_split` is random over accessions, so every val
accession has close relatives in train (median max-kinship to train ~0.17)
and a cis model can score by implicit haplotype matching. We currently
neutralise that by residualising out GBLUP, which is conservative (it also
removes the population-stratified share of real cis effects).

This split does the opposite: keep the raw target, hold out whole admixture
groups that are genetically distant from everything else, and drop from
train any accession still related to them. Relatedness then cannot help;
a positive result is interpretable on the raw z, and the held-out
accessions carry alleles / haplotype combinations rare or absent in train —
the regime where a per-gene linear model is structurally weakest.

Choice (from the between-group GRM means):
  val  = italy_balkan_caucasus (82; median max-K to outside 0.05, p90 0.10)
  test = south_sweden          (41; 0.06 / 0.09)
  excluded = anyone else with max-K > 0.1 to a val/test accession (~13)
  train = the rest (~530)

Writes `accessions/kin_split` into the h5 (train/val/test/excluded) and
reports the GBLUP calibration on it: if GBLUP still scores well above 0
the split isn't distant enough.

    python -m training_ge.ath_kinship_split [--write]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

D = "/90daydata/small_grains/andrew.dickson/datasets/arabidopsis/expression/"
KEY = "accessions/kin_split"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-group", default="italy_balkan_caucasus")
    ap.add_argument("--test-group", default="south_sweden")
    ap.add_argument("--max-k", type=float, default=0.1,
                    help="train accessions with kinship above this to any "
                         "val/test accession are excluded")
    ap.add_argument("--write", action="store_true", help="write split to h5")
    args = ap.parse_args()

    import h5py

    K = np.load(D + "baselines/grm.npy")
    with h5py.File(D + "expression_dataset.h5", "r") as f:
        grp = f["accessions/admixture_group"][:].astype(str)
        old = f["accessions/acc_split"][:].astype(str)
        dev = f["deviation"][:]

    va = grp == args.val_group
    te = grp == args.test_group
    held = va | te
    related = (~held) & (K[:, held].max(1) > args.max_k)
    tr = ~held & ~related
    split = np.full(len(grp), "train", dtype=object)
    split[va], split[te], split[related] = "val", "test", "excluded"
    print({s: int((split == s).sum()) for s in ("train", "val", "test", "excluded")})
    print("excluded by group:",
          dict(zip(*np.unique(grp[related], return_counts=True))))
    for name, m in (("val", va), ("test", te)):
        mx = K[np.ix_(m, tr)].max(1)
        print(f"{name}: max-K to train median {np.median(mx):.3f} "
              f"p90 {np.percentile(mx, 90):.3f} max {mx.max():.3f}")
    otr, ova = old == "train", old == "val"
    print(f"(random acc_split for reference: val max-K to train median "
          f"{np.median(K[np.ix_(ova, otr)].max(1)):.3f})")

    # GBLUP calibration on the new split, same recipe as ath_kinship_baseline
    mu = dev[:, tr].mean(1, keepdims=True)
    sd = dev[:, tr].std(1, ddof=1, keepdims=True)
    ok = sd[:, 0] > 1e-3
    Y = (dev - mu) / np.where(sd > 1e-3, sd, 1.0)
    Y_tr, Y_va = Y[ok][:, tr], Y[ok][:, va]
    K_tt = K[np.ix_(tr, tr)]
    K_vt = K[np.ix_(va, tr)]
    print(f"\nGBLUP on kin_split ({ok.sum():,} genes, {va.sum()} val accs)")
    print(f"{'lambda':>8} {'pooled pearson':>15}")
    for lam in (0.3, 1.0, 3.0, 10.0):
        A = K_vt @ np.linalg.solve(K_tt + lam * np.eye(tr.sum()), np.eye(tr.sum()))
        P = Y_tr @ A.T
        print(f"{lam:>8.1f} {np.corrcoef(P.ravel(), Y_va.ravel())[0, 1]:>15.4f}")

    if args.write:
        with h5py.File(D + "expression_dataset.h5", "a") as f:
            if KEY in f:
                del f[KEY]
            f.create_dataset(KEY, data=split.astype("S"))
            f[KEY].attrs["val_group"] = args.val_group
            f[KEY].attrs["test_group"] = args.test_group
            f[KEY].attrs["max_k"] = args.max_k
        print(f"wrote {KEY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build a small dataset.h5 with the stage-3 schema but synthetic counts, for
end-to-end smoke tests before the real extraction finishes.

Targets are genotype-driven: p = sigmoid(site_base + sum_j beta_j * dosage_j + noise)
so a correct pipeline shows positive per-site r for both emb and geno models.

  python training_meth/tests/make_mock_dataset.py --out $SCRATCH/mock_dataset.h5 --n-train 60 --n-val 20
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ath_meth_benchmark" / "build"))
from window_loader import WindowLoader  # noqa: E402

DATA = "/90daydata/small_grains/andrew.dickson/datasets/arabidopsis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    var = pd.read_parquet(f"{DATA}/methylation/stage2/sites_variable.parquet")
    var = var[var.context == "CG"]
    pick = pd.concat([var[var.pos_split == "train"].sample(a.n_train, random_state=a.seed),
                      var[var.pos_split == "val"].sample(a.n_val, random_state=a.seed)])
    pick = pick.sort_values("site_idx").reset_index(drop=True)

    acc = pd.read_csv(f"{DATA}/methylation/meta/benchmark_accessions.tsv", sep="\t", dtype=str)
    acc["acc_split"] = ["test" if g == "italy_balkan_caucasus" else "val" if g == "asia" else "train"
                        for g in acc.admixture_group.fillna("")]
    wl = WindowLoader(f"{DATA}/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa",
                      f"{DATA}/methylation/snv_arrays.h5", window=512)
    n_acc = len(acc)
    counts = np.zeros((len(pick), n_acc, 2), np.int16)
    for i, r in pick.iterrows():
        lo, hi = int(r.pos) - 257, int(r.pos) + 255
        dos = []
        for e in acc.ecotype_id:
            pos, alt = wl._snvs(e, str(r.chrom))
            i0, i1 = np.searchsorted(pos, [lo + 1, hi + 1])
            dos.append(pos[i0:i1])
        allpos = np.unique(np.concatenate(dos))
        D = np.zeros((n_acc, len(allpos)))
        for j, p in enumerate(dos):
            D[j, np.searchsorted(allpos, p)] = 1
        beta = rng.normal(0, 1.5, len(allpos))
        logit = np.log(r.mean_p / (1 - r.mean_p + 1e-6) + 1e-6) + D @ beta + rng.normal(0, 0.5, n_acc)
        p = 1 / (1 + np.exp(-logit))
        total = rng.poisson(15, n_acc).astype(np.int16)
        counts[i, :, 1] = total
        counts[i, :, 0] = rng.binomial(total, p)
    sd = h5py.string_dtype()
    with h5py.File(a.out, "w") as h5:
        gs = h5.create_group("sites")
        for c in ["chrom", "strand", "context", "subtask", "split_role", "pos_split", "annotation_class"]:
            gs.create_dataset(c, data=pick[c].astype(str).to_numpy(), dtype=sd)
        for c, dt in [("pos", np.int64), ("mean_p", np.float32), ("var_obs", np.float32), ("noise", np.float32),
                      ("n_obs", np.int16), ("cg_density", np.int16), ("site_idx", np.int64)]:
            gs.create_dataset(c, data=pick[c].to_numpy(dt))
        h5.create_dataset("counts", data=counts)
        ga = h5.create_group("accessions")
        for c in ["ecotype_id", "admixture_group", "tissue", "gsm", "source_series", "acc_split"]:
            ga.create_dataset(c, data=acc[c].fillna("").astype(str).to_numpy(), dtype=sd)
    print("wrote", a.out, counts.shape)


if __name__ == "__main__":
    main()

"""
The honest linear bar for the arabidopsis cache result: per-gene elastic net on
cis-SNP genotypes, predicting the SAME kinship-subtracted target, on the SAME
200 genes and the SAME accession split the cache run used.

Unlike SIEVE (private mutations -> per-gene enet is degenerate), 1001G variants
recur across accessions, so PrediXcan-style per-gene elastic net is well-posed.
This is the model the variant cache has to beat to justify itself: if a linear
map from the same cis-SNPs reaches the same pooled val pearson, the pretrained
representation is buying nothing here.

Matched to training_ge/run.py --dataset ath --holdout accessions
--kinship-residual:
  * genes: source.sample_genes(200, split="train"), seed 42 (identical set)
  * target: z (train-accession standardized) minus train-fitted GBLUP (lambda=3)
  * window: TSS +/- hw (default 4000), biallelic SNPs, alt-carrier = geno>0
  * fit on acc_split train, score pooled pearson on val, test excluded

    python -m training_ge.ath_elasticnet_control --n-genes 200
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
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--hw", type=int, default=4000)
    ap.add_argument("--gblup-lambda", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-key", default="acc_split",
                    choices=["acc_split", "kin_split"])
    ap.add_argument("--no-kinship-residual", action="store_true",
                    help="fit/score on raw z (for kin_split, where the "
                         "split itself controls relatedness)")
    ap.add_argument("--enet-residual", action="store_true",
                    help="linear-exhaustion check: fit a SECOND elastic net "
                         "on the double residual (kinship + enet #1 removed). "
                         "~0 means enet #1 was CV-optimal and whatever the "
                         "cache finds there is beyond linear reach.")
    args = ap.parse_args()

    import h5py
    import pgenlib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV
    from scipy.stats import pearsonr
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from transformers import AutoTokenizer
    from training_ge.ath_data import ArabidopsisWindowSource, load_pvar

    # Build the source exactly as run.py does, so sample_genes returns the
    # identical 200-gene set the cache trained on (its rng state, including the
    # verify_id6 draw, is reproduced by construction).
    tok = AutoTokenizer.from_pretrained("HuggingFaceBio/Carbon-500M",
                                        trust_remote_code=True)
    src = ArabidopsisWindowSource(tok, half_window=args.hw, seed=args.seed,
                                  kinship_residual=not args.no_kinship_residual,
                                  gblup_lambda=args.gblup_lambda,
                                  split_key=args.split_key)
    gene_ix = src.sample_genes(args.n_genes, split="train")
    if args.enet_residual:
        src.subtract_enet(gene_ix)          # enet #1, identical recipe
    z = src.z_all                           # kinship-(and enet-)subtracted

    with h5py.File(H5, "r") as f:
        dev = f["deviation"][:]
        chrom = f["genes/chrom"][:].astype(str)
        tss = f["genes/tss"][:]
        eco = f["accessions/ecotype_id"][:].astype(str)
        acc_split = f[f"accessions/{args.split_key}"][:].astype(str)
    print(f"split={args.split_key} kinship_residual={not args.no_kinship_residual}")

    tr, va = acc_split == "train", acc_split == "val"

    psam = [l.split()[0] for l in open(PFILE + ".psam") if not l.startswith("#")]
    col = {s: i for i, s in enumerate(psam)}
    panel_cols = np.array([col[e] for e in eco])
    tr_cols, va_cols = panel_cols[tr], panel_cols[va]

    v_chrom, v_pos, v_ref, v_alt = load_pvar(PFILE + ".pvar")
    by_chrom = {c: np.flatnonzero(v_chrom == c) for c in np.unique(v_chrom)}
    reader = pgenlib.PgenReader(str(PFILE + ".pgen").encode())
    n_psam = len(psam)

    preds, targs, per_gene, novel = [], [], [], []
    preds_tr, targs_tr = [], []
    n_skip = 0
    for k, gi in enumerate(gene_ix):
        c = chrom[gi]
        ix = by_chrom[c]
        pos_c = v_pos[ix]
        a, b = np.searchsorted(pos_c, [tss[gi] - args.hw, tss[gi] + args.hw])
        if b <= a:
            n_skip += 1
            continue
        vsel = ix[a:b]
        keep = np.array([len(v_ref[v]) == 1 and len(v_alt[v]) == 1 for v in vsel])
        vsel = vsel[keep]
        if len(vsel) == 0:
            n_skip += 1
            continue
        lo, hi = int(vsel[0]), int(vsel[-1]) + 1
        geno = np.empty((hi - lo, n_psam), dtype=np.int8)
        reader.read_range(lo, hi, geno)
        geno = geno[vsel - lo]
        alt = (geno > 0)
        X_tr = alt[:, tr_cols].T.astype(np.float32)
        X_va = alt[:, va_cols].T.astype(np.float32)
        # drop monomorphic-in-train columns; a val row carrying an alt at an
        # absent-in-train column is a 'novel-allele' row (same definition as
        # ath_data.GeneBatch.novel, so the cache evaluator's subsets match)
        p = X_tr.mean(0)
        keepc = (p > 0) & (p < 1)
        novel.append((X_va[:, p == 0] > 0).any(1))
        X_tr, X_va = X_tr[:, keepc], X_va[:, keepc]
        if X_tr.shape[1] == 0:
            n_skip += 1
            continue
        y_tr, y_va = z[gi, tr], z[gi, va]
        enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=20, cv=5,
                            max_iter=3000, n_jobs=-1, random_state=0)
        enet.fit(X_tr, y_tr)
        p_va = enet.predict(X_va)
        preds_tr.append(enet.predict(X_tr)); targs_tr.append(y_tr)
        preds.append(p_va)
        targs.append(y_va)
        if p_va.std() > 0:
            per_gene.append(pearsonr(y_va, p_va).statistic)
        if (k + 1) % 25 == 0:
            pp, tt = np.concatenate(preds), np.concatenate(targs)
            print(f"  {k+1}/{len(gene_ix)}: pooled val pearson so far "
                  f"{pearsonr(tt, pp).statistic:+.4f}")

    P, T = np.concatenate(preds), np.concatenate(targs)
    Nv = np.concatenate(novel)
    r = pearsonr(T, P)
    print(f"\nscored {len(preds)}/{len(gene_ix)} genes ({n_skip} skipped), "
          f"{len(T):,} val pairs")
    print(f"POOLED val pearson (elastic net) = {r.statistic:+.4f} "
          f"(p={r.pvalue:.1e})")
    print(f"  train (in-sample) pooled pearson = "
          f"{pearsonr(np.concatenate(targs_tr), np.concatenate(preds_tr)).statistic:+.4f}")
    for tag, m in (("novel-allele rows", Nv), ("seen-allele rows", ~Nv)):
        if m.sum() >= 30 and P[m].std() > 0:
            print(f"  {tag:18s} n={m.sum():,}  pooled pearson="
                  f"{pearsonr(T[m], P[m]).statistic:+.4f}")
    print(f"per-gene val pearson: median {np.nanmedian(per_gene):+.4f}, "
          f"mean {np.nanmean(per_gene):+.4f}")
    if args.enet_residual:
        print("\nlinear-exhaustion check: cache scored ~+0.05 on this same "
              "double-residual target; enet #2 near 0 => that signal is "
              "beyond linear reach, enet #2 >= 0.05 => enet #1 underfit")
    else:
        print("\ncompare: variant cache (r=1, wd=3) plateaued ~+0.063 pooled "
              "val pearson on the same target/split")
    return 0


if __name__ == "__main__":
    sys.exit(main())

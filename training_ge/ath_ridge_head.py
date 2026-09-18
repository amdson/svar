"""
Closed-form per-gene ridge readout on frozen cache features.

The per-gene head in run.py (--per-gene-head) gets one Adam update per gene
per epoch, i.e. ~8 updates in a run — it never fits. With the encoder frozen
the head is a ridge regression per gene, so fit it exactly instead:

  1. one inference pass: pooled (mutant - ref) delta for EVERY train/val row
     of the same 200 genes, cached to --features (npz, ~200 x 610 x 1024)
  2. per gene: RidgeCV on train rows (efficient LOO over an alpha grid — the
     counterpart to the elastic net's CV), predict val rows
  3. pooled val pearson, all / novel-allele / seen-allele rows, plus a
     shared (one ridge across all genes) reference and per-gene stats

--ckpt optionally loads a trained LoRA so the same fit can be run on
fine-tuned features (is the LoRA moving the readable signal?). Step 2 is
CPU-only once the features exist: re-run with --features to skip the GPU.

    python -m training_ge.ath_ridge_head --split-key kin_split \
        --features $SVAR_SCRATCH/runs/feat_kin_frozen.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch


def extract(args, feats_path):
    from transformers import AutoTokenizer  # noqa: F401  (tokenizer via loader)
    from CARBON_modules import load_carbon_variant_lora
    from training_ge.ath_data import ArabidopsisWindowSource
    from training_ge.run import _delta_rows

    device = "cuda"
    r, alpha, dropout = 8, 16.0, 0.0
    ck = None
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        a = ck["args"]
        r, alpha, dropout = a["lora_r"], a["lora_alpha"], a["lora_dropout"]
    model, tok = load_carbon_variant_lora(device=device, base_dtype=torch.bfloat16,
                                          r=r, alpha=alpha, dropout=dropout)
    if ck is not None:
        missing, unexpected = model.load_state_dict(ck["lora"], strict=False)
        assert not unexpected and not [k for k in missing if "lora_" in k]
        print(f"loaded LoRA from {args.ckpt} (epoch {ck['epoch']})")
    else:
        print("frozen pretrained Carbon (LoRA at init = identity)")
    model.eval()
    model.encoder.variant_checkpointing = False

    src = ArabidopsisWindowSource(tok, half_window=args.hw, seed=args.seed,
                                  kinship_residual=args.kinship_residual,
                                  split_key=args.split_key)
    gene_ix = src.sample_genes(args.n_genes, split="train")
    sp = dict(zip(src.eco, src.acc_split))
    out = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        for k, gi in enumerate(gene_ix):
            b = src.build(int(gi))
            if b is None:
                continue
            rows = torch.arange(len(b.z))
            F = torch.cat([_delta_rows(model, b, device, ch).cpu()
                           for ch in rows.split(args.rows_per_call)])
            out[b.gene_id] = dict(
                F=F.numpy().astype(np.float32), z=b.z.numpy(),
                novel=b.novel.numpy(),
                split=np.array([sp[l] for l in b.lines]))
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(gene_ix)} genes, {time.perf_counter()-t0:.0f}s")
    np.savez(feats_path, genes=np.array(list(out)),
             **{f"{g}/{key}": v for g, d in out.items() for key, v in d.items()})
    print(f"saved features for {len(out)} genes -> {feats_path}")


def fit(args, feats_path):
    from scipy.stats import pearsonr
    from sklearn.linear_model import RidgeCV

    d = np.load(feats_path)
    genes = d["genes"]
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    P, T, Nv, per_gene, chosen = [], [], [], [], []
    Ptr, Ttr = [], []
    Ftr_all, ztr_all, Fva_all = [], [], []
    for g in genes:
        F, z = d[f"{g}/F"], d[f"{g}/z"]
        sp, nv = d[f"{g}/split"], d[f"{g}/novel"]
        tr, va = sp == "train", sp == "val"
        if tr.sum() < 20 or va.sum() == 0:
            continue
        m = RidgeCV(alphas=alphas).fit(F[tr], z[tr])
        p = m.predict(F[va])
        P.append(p); T.append(z[va]); Nv.append(nv[va]); chosen.append(m.alpha_)
        Ptr.append(m.predict(F[tr])); Ttr.append(z[tr])
        if p.std() > 0:
            per_gene.append(pearsonr(z[va], p).statistic)
        Ftr_all.append(F[tr]); ztr_all.append(z[tr]); Fva_all.append(F[va])
    P, T, Nv = map(np.concatenate, (P, T, Nv))
    print(f"\nper-gene ridge on {len(chosen)} genes, {len(T):,} val pairs; "
          f"alpha chosen: median {np.median(chosen):.3g} "
          f"(grid {alphas[0]:.3g}..{alphas[-1]:.3g})")
    print(f"POOLED val pearson (per-gene ridge) = {pearsonr(T, P).statistic:+.4f}")
    Ptr, Ttr = np.concatenate(Ptr), np.concatenate(Ttr)
    print(f"  train (in-sample) pooled pearson = {pearsonr(Ttr, Ptr).statistic:+.4f}")
    for tag, mm in (("novel-allele rows", Nv), ("seen-allele rows", ~Nv)):
        if mm.sum() >= 30:
            print(f"  {tag:18s} n={mm.sum():,}  pooled pearson="
                  f"{pearsonr(T[mm], P[mm]).statistic:+.4f}")
    print(f"per-gene val pearson: median {np.nanmedian(per_gene):+.4f}, "
          f"mean {np.nanmean(per_gene):+.4f}")

    # shared-head reference: one ridge over all genes' rows (what run.py's
    # plain linear head is, minus the LoRA)
    m = RidgeCV(alphas=alphas).fit(np.concatenate(Ftr_all), np.concatenate(ztr_all))
    Ps = m.predict(np.concatenate(Fva_all))
    print(f"shared ridge (one head for all genes): pooled val pearson = "
          f"{pearsonr(T, Ps).statistic:+.4f} (alpha {m.alpha_:.3g}); train "
          f"{pearsonr(Ttr, m.predict(np.concatenate(Ftr_all))).statistic:+.4f}")
    print("\ncompare: per-gene elastic net on genotypes, same rows: "
          "kin_split raw +0.243 / acc_split kinship-resid +0.233")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="npz cache path")
    ap.add_argument("--ckpt", default=None, help="LoRA checkpoint (else frozen)")
    ap.add_argument("--split-key", default="kin_split")
    ap.add_argument("--kinship-residual", action="store_true")
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--hw", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows-per-call", type=int, default=64)
    ap.add_argument("--alpha-min", type=float, default=-1)
    ap.add_argument("--alpha-max", type=float, default=5)
    ap.add_argument("--n-alphas", type=int, default=13)
    ap.add_argument("--refit", action="store_true", help="re-extract features")
    args = ap.parse_args()
    if args.refit or not os.path.exists(args.features):
        extract(args, args.features)
    else:
        print(f"using cached features {args.features}")
    fit(args, args.features)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
training/snp_gated/run.py
-------------------------
Train a GatedRidge (sequence-conditioned ridge) on the SNP dosage matrix, where
each SNP column is rescaled by a learned gate a_j = exp(φ(E_j)) predicted from
that SNP's frozen Carbon window embedding. See training/snp_gated/model.py.

    python -m training.snp_gated.run --dataset soy_dev \\
        --cache $SVAR_SCRATCH/caches/soy/carbon500m_hw500.ckpt.pt \\
        --epochs 200 --lr 1e-2 --ridge 1e-2 --lam-gate 1.0 \\
        --early-stopping --output trained_heads/soy_dev_gated/model.pt

The per-SNP embedding E is built with NO extra GPU work and NO VCF genotype load:
the SNP→window map is rebuilt from the .pvar positions alone (the same greedy
SNPWindowPartitioner that made the cache), and E_j is that window's across-sample
mean embedding (train-only) read from the cache. `--freeze-gate` pins a≡1 for the
ordinary-ridge sanity baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from crop_embed.data.vcf import _parse_chrom
from crop_embed.models.fp_head_model import window_means, window_position_features_from_cache
from crop_embed.train import masked_mse, _compute_metrics
from crop_embed import MetricLogger, metrics_path_for

from training.common import features as feat, metrics as cmetrics, run_record
from training.common.datasets import get_dataset
from training.common.splits import default_split_path, get_or_build_split
from training.snp_gated.model import GatedRidge

from crop_embed.partitioner import SNPWindowPartitioner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a sequence-gated ridge on the SNP matrix.")
    p.add_argument("--dataset", required=True, help="registry dataset name (soy/soy_dev/...).")
    p.add_argument("--cache", required=True,
                   help="window-embedding cache (.ckpt.pt) supplying the per-SNP embeddings E.")
    p.add_argument("--impute", default="ref", choices=["ref", "mean"],
                   help="missing-genotype fill for the dosage matrix.")
    # Gate architecture
    p.add_argument("--hidden", type=int, default=256, help="gate MLP hidden width.")
    p.add_argument("--per-trait", action="store_true",
                   help="a separate gate per (SNP, trait) instead of one shared gate per SNP.")
    p.add_argument("--freeze-gate", action="store_true",
                   help="keep a≡1 (never train the gate) — the ordinary-ridge sanity baseline.")
    # Regularization
    p.add_argument("--ridge", type=float, default=10.0,
                   help="explicit L2 penalty coefficient on the SNP effects β (added to the "
                        "loss as ridge·‖β‖²). This is the ridge/RR-BLUP penalty; in the p≫n "
                        "regime it needs to be large (closed-form ridge picks alpha ~1e4–1e5).")
    p.add_argument("--lam-gate", type=float, default=1.0,
                   help="weight on mean(log a)² — the prior pulling the gate toward 1 (ridge). "
                        "Large ⇒ ordinary ridge; 0 ⇒ free gate.")
    p.add_argument("--gate-weight-decay", type=float, default=0.0,
                   help="optional L2 on the gate MLP weights (separate from --lam-gate).")
    # Optimization
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--early-stopping", action="store_true",
                   help="restore the best-val-PCC epoch before the final test eval / save.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    # wandb / IO
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--output", required=True, help="path to write model.pt.")
    return p


def _read_pvar_positions(pvar_path: str) -> list[tuple[int, int]]:
    """[(chrom_int, pos0), …] in .pvar (=.pgen) column order.

    The .pvar POS column is 1-based (VCF convention), but the cache's windows were
    built by load_snps_from_vcf from pysam's 0-based ``rec.start``. So convert POS→
    0-based (``-1``) here to match the cache's window coordinates exactly."""
    out: list[tuple[int, int]] = []
    with open(pvar_path) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            parts = ln.split()
            out.append((_parse_chrom(parts[0]), int(parts[1]) - 1))
    return out


def build_snp_embeddings(
    spec, cache_path: str, variant_ids: list[str], train_idx: torch.Tensor, device
) -> torch.Tensor:
    """(n_snp, D) per-SNP window embedding, aligned to `variant_ids`.

    No VCF genotype load and no extra GPU embedding: rebuild the SNP→window map
    from the .pvar positions with the same greedy SNPWindowPartitioner that made
    the cache, then take each window's across-sample MEAN embedding (fit on TRAIN
    only, so val/test never leak into E) as the SNP's neighborhood vector."""
    cached = feat.load_cached_windows(cache_path)
    if cached is None:
        raise SystemExit(f"cache {cache_path} predates sample_ids; rebuild it.")
    hw = int(cached.metadata["half_window"])
    buf = int(cached.metadata.get("buffer", 0))

    positions = _read_pvar_positions(spec.pgen_prefix + ".pvar")
    if len(positions) != len(variant_ids):
        raise SystemExit(f".pvar has {len(positions)} variants but the SNP matrix has "
                         f"{len(variant_ids)} — mismatched filesets.")

    # Rebuild windows from positions alone (partitioner only reads .pos).
    class _P:
        __slots__ = ("pos",)
        def __init__(self, pos): self.pos = pos
    snps_by_chrom: dict[int, list] = {}
    for chrom, pos in positions:
        snps_by_chrom.setdefault(chrom, []).append(_P(pos))
    for chrom in snps_by_chrom:                       # partitioner assumes sorted-by-pos
        snps_by_chrom[chrom].sort(key=lambda r: r.pos)
    part = SNPWindowPartitioner(snps_by_chrom, half_window=hw, buffer=buf)

    n_windows = cached.sample_fp_index.shape[1]
    if len(part) != n_windows:
        raise SystemExit(f"rebuilt {len(part)} windows but the cache has {n_windows}; "
                         "the .pvar SNP set doesn't match the cache's windowing.")
    # Sanity: the rebuilt windows must line up with the cache's window columns.
    chroms_c, pos_c = window_position_features_from_cache(
        cached.unique_fingerprints, cached.sample_fp_index)
    w0 = part.windows[0]
    assert int(chroms_c[0]) == w0.chrom and int(pos_c[0]) == (w0.start + w0.end) // 2, \
        "rebuilt window 0 disagrees with the cache — windowing mismatch"

    # Per-window across-sample mean embedding, TRAIN-only (no val/test leakage).
    wmean = window_means(cached.cache.to(device), cached.sample_fp_index.to(device),
                         train_idx.to(device)).cpu()                      # (n_windows, D)

    # Each variant → its window column → that window's mean embedding.
    win_of_var = torch.tensor(
        [part.snp_to_window_idx[(chrom, pos)] for chrom, pos in positions], dtype=torch.long)
    E = wmean[win_of_var]                                                # (n_snp, D)
    # Standardize per-dim ACROSS SNPs so the gate MLP sees zero-mean/unit-var inputs.
    # Raw Carbon embeddings have a large common-mode component and small per-SNP
    # contrast, which otherwise collapses φ(E) to a near-constant gate (no per-SNP
    # differentiation). This conditioning is over the SNP axis only — no label, no leak.
    E = (E - E.mean(0)) / E.std(0).clamp_min(1e-6)
    return E


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = MetricLogger(metrics_path_for(args.output),
                          wandb_project=args.wandb_project, wandb_name=args.wandb_name,
                          config=vars(args))
    print(f"Logging metrics to {logger.metrics_path}")

    spec = get_dataset(args.dataset)
    split = get_or_build_split(args.dataset, seed=args.seed)
    samples = split.sample_ids
    trait_cols = split.trait_cols

    # SNP dosage matrix + per-trait z-scored targets, both aligned to `samples`.
    X_np, variant_ids = feat.snp_matrix(spec, samples, impute=args.impute)
    Y_np, _ = feat.scaled_targets(spec, samples, trait_cols)
    X = torch.from_numpy(np.ascontiguousarray(X_np)).float()
    Y = torch.from_numpy(np.ascontiguousarray(Y_np)).float()
    n_traits = Y.shape[1]

    train_idx = torch.tensor(split.indices("train", samples), dtype=torch.long)
    val_idx = torch.tensor(split.indices("val", samples), dtype=torch.long)
    test_idx = torch.tensor(split.indices("test", samples), dtype=torch.long)
    print(f"SNP matrix: {X.shape[0]} samples × {X.shape[1]} variants; "
          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)} train/val/test")

    # Standardize SNP columns on TRAIN stats (no val/test leakage). Un-standardized
    # 0/1/2 dosage over ~40k columns makes the linear layer's output variance huge and
    # SGD diverges (closed-form ridge sidesteps this; SGD does not). Centering + unit
    # scaling is also the usual GBLUP genotype preprocessing.
    # CRUCIAL: columns that are ~constant in TRAIN (rare variants absent from the 1012
    # train samples but present in val/test) must be ZEROED everywhere — dividing a
    # nonzero val genotype by a ~0 train std sends the feature to ~1e8 and detonates the
    # val/test predictions (train looks fine because the column is constant there).
    mu = X[train_idx].mean(0)
    sd = X[train_idx].std(0)
    keep = sd > 1e-6
    X = (X - mu) / torch.where(keep, sd, torch.ones_like(sd))
    X[:, ~keep] = 0.0
    print(f"  standardized SNP columns; dropped {int((~keep).sum())} train-constant of {X.shape[1]}")

    E = build_snp_embeddings(spec, args.cache, variant_ids, train_idx, device)
    print(f"Per-SNP embeddings E: {tuple(E.shape)} (train-only window means)")

    model = GatedRidge(E, n_traits, hidden=args.hidden, per_trait=args.per_trait).to(device)
    if args.freeze_gate:
        for p in model.gate.parameters():
            p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GatedRidge ({'frozen a≡1' if args.freeze_gate else 'learned gate'}, "
          f"{'per-trait' if args.per_trait else 'shared'}; {n_params:,} trainable params)")

    # β's ridge penalty is applied EXPLICITLY in the loss (ridge·‖β‖²), not as AdamW
    # weight decay: p≫n ridge needs an alpha ~1e4–1e5 that decoupled weight decay can't
    # reach. The gate MLP keeps its own (small) optional weight decay; its pull toward
    # a≡1 comes from --lam-gate.
    opt = torch.optim.AdamW([
        {"params": model.beta.parameters(), "weight_decay": 0.0},
        {"params": model.gate.parameters(), "weight_decay": args.gate_weight_decay},
    ], lr=args.lr)

    Xtr, Ytr = X[train_idx], Y[train_idx]
    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True)

    def _predict(idx: torch.Tensor) -> torch.Tensor:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), args.batch_size):
                out.append(model(X[idx[i:i + args.batch_size]].to(device)).cpu())
        return torch.cat(out, dim=0)

    print(f"\nTraining {args.epochs} epochs at lr={args.lr}, ridge={args.ridge}, "
          f"lam_gate={args.lam_gate} …")
    step = 0
    best_val_pcc, best_state, best_epoch = float("-inf"), None, -1
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            # ridge·‖β‖² is the explicit RR-BLUP penalty; β is the rescaled-unit effect,
            # so with a learned gate this is exactly differential shrinkage λ(β/a)².
            loss = (masked_mse(pred, yb)
                    + args.ridge * model.beta.weight.pow(2).sum()
                    + args.lam_gate * model.gate_penalty())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if step % args.log_every == 0:
                tm = _compute_metrics(pred.detach(), yb, trait_cols, "train")
                print(f"epoch {epoch:4d} step {step:6d}  loss={loss.item():.4f}"
                      f"  pcc={tm.get('train/mean_pcc', float('nan')):.3f}")
                logger.log({"epoch": epoch, "step": step, **tm})
            step += 1

        val_m = _compute_metrics(_predict(val_idx), Y[val_idx], trait_cols, "val")
        vp = val_m.get("val/mean_pcc", float("nan"))
        print(f"epoch {epoch:4d}  [val] val_pcc={vp:.3f}")
        logger.log({"epoch": epoch, "step": step, **val_m})
        if args.early_stopping and vp > best_val_pcc:
            best_val_pcc, best_epoch = vp, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if args.early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Early stopping: restored epoch {best_epoch} (best val_pcc={best_val_pcc:.3f})")

    val_metrics = cmetrics.evaluate(Y[val_idx].numpy(), _predict(val_idx).numpy(), trait_cols)
    test_metrics = cmetrics.evaluate(Y[test_idx].numpy(), _predict(test_idx).numpy(), trait_cols)
    tm = test_metrics.get("mean", {})
    print(f"\n[test] mean pcc={tm.get('pearson', float('nan')):.3f}  "
          f"mse={tm.get('mse', float('nan')):.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"hidden": args.hidden, "per_trait": args.per_trait, "freeze_gate": args.freeze_gate,
              "ridge": args.ridge, "lam_gate": args.lam_gate, "n_traits": n_traits,
              "early_stopping": args.early_stopping, "best_epoch": best_epoch}
    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "cache_path": args.cache, "variant_ids": variant_ids,
                "split_path": str(default_split_path(args.dataset, args.seed)),
                "trait_cols": trait_cols, "sample_ids": samples, "args": vars(args)}, out_path)
    print(f"Saved to {out_path}")

    rec = run_record.build(
        dataset=args.dataset, features="snp_gated", model="gated_ridge", seed=args.seed,
        traits=trait_cols, hyperparams={**config, "lr": args.lr, "epochs": args.epochs},
        metrics={"val": val_metrics, "test": test_metrics},
        split_path=str(default_split_path(args.dataset, args.seed)), cache_path=args.cache)
    run_record.write(rec, out_path.parent)
    logger.close()


if __name__ == "__main__":
    main()

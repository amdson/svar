"""
training/emb_nn/run.py  (was train_pipeline/train_head.py)
----------------------------------------------------------
Train a head model on top of precomputed window embeddings (fixed cache). The GLM
head machinery is unchanged from the original train_head.py — only the data layer
is swapped onto training/common: pick a dataset by name, build its window dataset,
load the shared 70/15/15 split, and evaluate on val AND test. A lightweight run
record is written alongside the usual model.pt + metrics.jsonl.

By default trains an FPSumHeadModel (pool absolute window embeddings). Pass
--subtract-reference for FPRefDeltaSumHeadModel (per-window delta-from-reference)
or --center-windows for FPCenteredSumHeadModel. Head class, optimizer, and loop
are identical across these — only the pooled table and head class differ.

    python -m training.emb_nn.run --dataset rice \\
        --cache checkpoints/carbon/sativas413_carbon500m_hw500.ckpt.pt \\
        --head mlp --epochs 100 --lr 1e-3 --warm-start-standardizer \\
        --output trained_heads/mlp_sum/model.pt
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from crop_embed.models.fp_head_model import (
    MLPModel, LinearModel, FPSumHeadModel, FPCenteredSumHeadModel,
    FPRefDeltaSumHeadModel, window_means)
from crop_embed import FixedWindowEmbedder, MetricLogger, metrics_path_for, ActivationTracker
from crop_embed.train import masked_mse, _compute_metrics

from training.common import features as feat, metrics as cmetrics, run_record
from training.common.datasets import get_dataset
from training.common.splits import default_split_path, get_or_build_split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a head on precomputed window embeddings.")
    p.add_argument("--dataset", default="rice", help="registry dataset name (rice/soy/...).")
    p.add_argument("--cache", type=str, required=True,
                   help="FixedWindowEmbedder cache (.pt) from embed_windows.py.")
    # Windowing — must match how the cache was generated (checked below).
    p.add_argument("--half-window", type=int, default=500)
    p.add_argument("--buffer", type=int, default=0)
    # Head architecture
    p.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--hidden-dim", type=int, default=None, help="MLP hidden width (default emb_dim).")
    p.add_argument("--n-layers", type=int, default=2, help="MLP residual blocks (--head mlp).")
    p.add_argument("--dropout", type=float, default=0.0, help="MLP dropout (--head mlp).")
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable the learned de-mean/rescale standardizer in the head.")
    p.add_argument("--standardizer", choices=["perdim", "rms"], default="perdim",
                   help="Standardizer when normalize is on. 'perdim' rescales each dim by "
                        "its own std; 'rms' de-means per-dim but rescales by one RMS scalar "
                        "(prefer for --pool mean).")
    p.add_argument("--freeze-standardizer", action=argparse.BooleanOptionalAction, default=None,
                   help="Freeze the standardizer after warm-start. Default: frozen whenever "
                        "--warm-start-standardizer is set (recommended for --pool mean).")
    p.add_argument("--subtract-reference", action="store_true",
                   help="FPRefDeltaSumHeadModel: subtract each window's variant-free reference "
                        "embedding before pooling (learn deltas-from-reference).")
    p.add_argument("--center-windows", action="store_true",
                   help="FPCenteredSumHeadModel: subtract each window's across-sample MEAN "
                        "(fit on TRAIN) before pooling. Mutually exclusive with --subtract-reference.")
    p.add_argument("--pool", choices=["sum", "mean"], default="sum",
                   help="Window pooling into the per-sample vector (saved in head_config).")
    p.add_argument("--center-ln-pool", action="store_true",
                   help="DIAGNOSTIC: mean-pool per-window-centered, per-window-LayerNorm'd windows "
                        "(uniform-attention floor). Forces mean pool; not reproduced at inference.")
    # Warm-starting (mutually exclusive)
    p.add_argument("--warm-start-head", type=str, default=None,
                   help="Pretrained head checkpoint to load_state_dict from.")
    p.add_argument("--warm-start-standardizer", action="store_true",
                   help="Fit the standardizer on the training embeddings before training.")
    # Optimization
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="L2 decay, applied only to inner Linear weights.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--track-activations", action="store_true",
                   help="Log per-layer activation stats via forward hooks (on --log-every steps).")
    p.add_argument("--seed", type=int, default=42)
    # wandb
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    # I/O
    p.add_argument("--output", type=str, required=True, help="Path to write model.pt.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.warm_start_head and args.warm_start_standardizer:
        parser.error("--warm-start-head and --warm-start-standardizer are mutually exclusive.")
    if args.subtract_reference and args.center_windows:
        parser.error("--subtract-reference and --center-windows are mutually exclusive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = MetricLogger(
        metrics_path_for(args.output),
        wandb_project=args.wandb_project, wandb_name=args.wandb_name, config=vars(args))
    print(f"Logging metrics to {logger.metrics_path}")

    # ── Dataset, targets, and the shared 3-way split (via training/common) ────────
    spec = get_dataset(args.dataset)
    # Cache the (slow) window-dataset build, keyed on dataset + windowing.
    _data_cache = Path(".prepare_data_cache") / f"{args.dataset}_hw{args.half_window}_buf{args.buffer}.pkl"
    if _data_cache.exists():
        print(f"Loading cached window dataset from {_data_cache}")
        with open(_data_cache, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = feat.build_window_dataset(spec, args.half_window, args.buffer)
        _data_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(_data_cache, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Cached window dataset to {_data_cache}")

    Y_np, trait_cols = feat.scaled_targets(spec, dataset.samples)   # aligned to dataset.samples, z-scored
    Y = torch.tensor(Y_np, dtype=torch.float32)

    split = get_or_build_split(args.dataset, seed=args.seed)
    train_idx = torch.tensor(split.indices("train", dataset.samples), dtype=torch.long)
    val_idx = torch.tensor(split.indices("val", dataset.samples), dtype=torch.long)
    test_idx = torch.tensor(split.indices("test", dataset.samples), dtype=torch.long)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    # ── 1. Load fixed window cache ────────────────────────────────────────────────
    print(f"\nLoading cache from {args.cache} …")
    embedder = FixedWindowEmbedder.from_file(args.cache, dataset)
    cache = embedder.cache.float()               # (n_fps, D)
    sample_fp_index = embedder.sample_fp_index   # (n_samples, n_windows)
    emb_dim = cache.shape[1]
    n_traits = Y.shape[1]

    if (sample_fp_index.shape != dataset.sample_fp_index.shape
            or not torch.equal(sample_fp_index, dataset.sample_fp_index)):
        raise SystemExit(
            "Cache sample→fingerprint index doesn't match the dataset built from the "
            "split's VCF/windowing. Regenerate the cache for this windowing.")
    print(f"  {cache.shape[0]:,} fingerprints × {emb_dim} dims; {sample_fp_index.shape[0]} samples")

    # ── 1.a Pre-sum into one embedding per sample ─────────────────────────────────
    if args.center_ln_pool:
        if args.subtract_reference:
            parser.error("--center-ln-pool is incompatible with --subtract-reference")
        ref_index = None
        cache_d = cache.to(device)
        sfi_d = sample_fp_index.to(device)
        wmean = window_means(cache_d, sfi_d, train_idx.to(device))       # (n_windows, D), train-only
        parts = []
        for i in range(0, sfi_d.shape[0], 16):
            g = cache_d[sfi_d[i:i + 16]] - wmean                          # center  (b, nw, D)
            g = F.layer_norm(g, (g.shape[-1],))                          # param-free per-window LN
            parts.append(g.mean(dim=1).cpu())                            # mean-pool over windows
        summed = torch.cat(parts, dim=0)                                 # (n_samples, D)
        print("  center-ln-pool: mean over centered+LayerNorm'd windows (uniform-attention floor)")
    elif args.subtract_reference:
        ref_index = FPRefDeltaSumHeadModel.build_ref_index(dataset.unique_fingerprints)
        table = FPRefDeltaSumHeadModel.subtract_reference(cache, ref_index)   # (n_fps, D)
        summed = F.embedding_bag(sample_fp_index, table, mode=args.pool)      # (n_samples, D)
    elif args.center_windows:
        ref_index = None
        window_center = FPCenteredSumHeadModel.build_window_center(
            cache.to(device), sample_fp_index.to(device), train_idx.to(device)).cpu()
        table = FPCenteredSumHeadModel.subtract_center(cache, window_center)  # (n_fps, D)
        summed = F.embedding_bag(sample_fp_index, table, mode=args.pool)      # (n_samples, D)
        print("  center-windows: pool each window's deviation from its across-sample mean")
    else:
        ref_index = None
        table = cache
        summed = F.embedding_bag(sample_fp_index, table, mode=args.pool)      # (n_samples, D)

    train_ds = TensorDataset(summed[train_idx], Y[train_idx])
    val_x = summed[val_idx].to(device)
    val_y = Y[val_idx].to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # ── 2. Build head (+ optimizer) ───────────────────────────────────────────────
    if args.head == "linear":
        inner: torch.nn.Module = LinearModel(emb_dim, n_traits)
    else:
        inner = MLPModel(emb_dim, n_traits, hidden_dim=args.hidden_dim,
                         n_layers=args.n_layers, dropout=args.dropout)

    warm_start_embeddings = summed[train_idx] if args.warm_start_standardizer else None
    freeze_standardizer = (args.warm_start_standardizer if args.freeze_standardizer is None
                           else args.freeze_standardizer)
    if args.subtract_reference:
        head = FPRefDeltaSumHeadModel(
            inner, emb_dim=emb_dim, ref_index=ref_index, normalize=not args.no_normalize,
            warm_start_embeddings=warm_start_embeddings, pool=args.pool,
            standardizer=args.standardizer, freeze_standardizer=freeze_standardizer).to(device)
    elif args.center_windows:
        head = FPCenteredSumHeadModel(
            inner, emb_dim=emb_dim, window_center=window_center, normalize=not args.no_normalize,
            warm_start_embeddings=warm_start_embeddings, pool=args.pool,
            standardizer=args.standardizer, freeze_standardizer=freeze_standardizer).to(device)
    else:
        head = FPSumHeadModel(
            inner, emb_dim=emb_dim, normalize=not args.no_normalize,
            warm_start_embeddings=warm_start_embeddings, pool=args.pool,
            standardizer=args.standardizer, freeze_standardizer=freeze_standardizer).to(device)

    if args.warm_start_head:
        print(f"Warm-starting head from {args.warm_start_head}")
        ckpt = torch.load(args.warm_start_head, map_location=device, weights_only=False)
        head.load_state_dict(ckpt["head_state_dict"])

    linear_weight_ids = {id(m.weight) for m in head.modules() if isinstance(m, torch.nn.Linear)}
    decay, nodecay = [], []
    for p in head.parameters():
        if not p.requires_grad:
            continue
        (decay if id(p) in linear_weight_ids else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}], lr=args.lr)

    n_params = sum(p.numel() for p in head.parameters())
    head_label = f"refdelta-{args.head}" if args.subtract_reference else (
        f"centered-{args.head}" if args.center_windows else args.head)
    norm_desc = (f"{args.standardizer}{'(frozen)' if freeze_standardizer else ''}"
                 if not args.no_normalize else "off")
    print(f"Head: {head_label} ({n_params:,} params)  normalize={norm_desc}")

    tracker = ActivationTracker(head) if args.track_activations else None

    # ── 3. Train ──────────────────────────────────────────────────────────────────
    n_train = len(train_ds)
    print(f"\nTraining {args.epochs} epochs over {n_train} samples at lr={args.lr} …")
    step = 0
    for epoch in range(args.epochs):
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            is_log = step % args.log_every == 0
            if tracker is not None and is_log:
                tracker.arm()
            pred = head.forward_postsum(x)
            loss = masked_mse(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if is_log:
                train_m = _compute_metrics(pred.detach(), y, trait_cols, "train")
                act_m = tracker.collect() if tracker is not None else {}
                print(f"epoch {epoch:4d} step {step:6d}"
                      f"  loss={train_m.get('train/mean_mse', loss.item()):.4f}"
                      f"  pcc={train_m.get('train/mean_pcc', float('nan')):.3f}")
                logger.log({"epoch": epoch, "step": step, **train_m, **act_m})
            step += 1

        head.eval()
        with torch.no_grad():
            val_m = _compute_metrics(head.forward_postsum(val_x), val_y, trait_cols, "val")
        print(f"epoch {epoch:4d}  [val]"
              f"  val_loss={val_m.get('val/mean_mse', float('nan')):.4f}"
              f"  val_pcc={val_m.get('val/mean_pcc', float('nan')):.3f}")
        logger.log({"epoch": epoch, "step": step, **val_m})

    # ── 3.a Final held-out TEST evaluation ────────────────────────────────────────
    head.eval()
    with torch.no_grad():
        val_pred = head.forward_postsum(summed[val_idx].to(device)).cpu().numpy()
        test_pred = head.forward_postsum(summed[test_idx].to(device)).cpu().numpy()
    val_metrics = cmetrics.evaluate(Y[val_idx].numpy(), val_pred, trait_cols)
    test_metrics = cmetrics.evaluate(Y[test_idx].numpy(), test_pred, trait_cols)
    tm = test_metrics.get("mean", {})
    print(f"\n[test] mean pcc={tm.get('pearson', float('nan')):.3f}  mse={tm.get('mse', float('nan')):.4f}")

    # ── 4. Save head + reconstruction metadata (unchanged format) ─────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    head_config = {
        "head": args.head,
        "model_class": ("FPRefDeltaSumHeadModel" if args.subtract_reference
                        else "FPCenteredSumHeadModel" if args.center_windows
                        else "FPSumHeadModel"),
        "subtract_reference": args.subtract_reference,
        "center_windows": args.center_windows,
        "pool": args.pool, "emb_dim": emb_dim, "n_traits": n_traits,
        "hidden_dim": args.hidden_dim, "n_layers": args.n_layers, "dropout": args.dropout,
        "normalize": not args.no_normalize, "standardizer": args.standardizer,
        "freeze_standardizer": freeze_standardizer,
    }
    torch.save({
        "head_state_dict": head.state_dict(),
        "head_config": head_config,
        "cache_path": args.cache,
        "split_path": str(default_split_path(args.dataset, args.seed)),
        "trait_cols": trait_cols,
        "sample_ids": dataset.samples,
        "args": vars(args),
    }, out_path)
    print(f"Saved to {out_path}")

    # ── 4.a Lightweight run record → runs/index.jsonl (shared with sklearn runs) ──
    rec = run_record.build(
        dataset=args.dataset, features="emb_nn", model=head_label, seed=args.seed,
        traits=trait_cols, hyperparams={**head_config, "lr": args.lr, "epochs": args.epochs,
                                        "weight_decay": args.weight_decay, "pool": args.pool},
        metrics={"val": val_metrics, "test": test_metrics},
        half_window=args.half_window,
        split_path=str(default_split_path(args.dataset, args.seed)),
        cache_path=args.cache)
    run_record.write(rec, out_path.parent)

    if tracker is not None:
        tracker.remove()
    logger.close()


if __name__ == "__main__":
    main()

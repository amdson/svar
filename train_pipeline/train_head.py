"""
train_head.py
-----------------
Train a head model on top of precomputed window embeddings. Kept separate
from end to end training because training on the precomputed cache is much faster
than training on a fixed model.
The output is a saved ckpt file with the head's state_dict and metadata needed to reconstruct the head and embedder for inference.

By default this trains an FPSumHeadModel (pool absolute window embeddings). Pass
--subtract-reference to train an FPRefDeltaSumHeadModel instead: each window's
variant-free *reference* embedding is subtracted before pooling, so the head
learns from per-window deltas-from-reference. The cache, split, optimizer, and
training loop are identical either way — only the pooled table (raw vs. delta)
and the head class differ.

Pipeline
--------
1.   Load the fixed window cache (from embed_windows.py).
1.a  Pre-sum it into one (n_samples, D) embedding per sample — for a frozen
     cache the per-sample sum never changes, so we do the embedding_bag once
     up front and train through forward_postsum (no re-gather every step).
     Wrapped in a TensorDataset for batching/shuffling. With --subtract-reference
     the reference-subtracted table is pooled instead (it's likewise frozen).
2.   Build an FPSumHeadModel (LinearModel/MLPModel inner) + optimizer.
2.a  Optionally warm-start the whole head from a pretrained checkpoint.
2.b  Optionally warm-start just the standardizer from the training embeddings
     (mutually exclusive with 2.a — a pretrained head's standardizer was fit on
     different embeddings).
3.   Train against the precomputed embeddings.
4.   Save head state_dict + metadata to reconstruct head and embedder.

Example
-------
    python train_pipeline/train_head.py \\
        --cache checkpoints/v2/sativas413_embeddings.ckpt.pt \\
        --head mlp --epochs 100 --lr 1e-3 --warm-start-standardizer \\
        --output trained_heads/mlp_sum/model.pt
python train_pipeline/train_head.py --cache checkpoints/sativas413_embeddings.ckpt.pt --head linear --epochs 100 --lr 1e-3 --warm-start-standardizer --output trained_heads/linear_sum/model.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from crop_embed.models.fp_head_model import (
    MLPModel, LinearModel, FPSumHeadModel, FPCenteredSumHeadModel)
from crop_embed import FixedWindowEmbedder, MetricLogger, metrics_path_for, ActivationTracker
from crop_embed.data.loading import prepare_data
from crop_embed.train import masked_mse, _compute_metrics

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train a head on precomputed window embeddings.")
parser.add_argument("--cache", type=str, required=True,
                    help="FixedWindowEmbedder cache (.pt) from embed_windows.py / generate_cache.py.")

# Windowing — must match how the cache was generated. The dataset rebuilt here
# (for the split + targets) is windowed independently of the cache, so a mismatch
# trips the sample_fp_index identity check below.
parser.add_argument("--half-window", type=int, default=500,
                    help="Half-window the cache was generated with (must match embed_windows).")
parser.add_argument("--buffer", type=int, default=0,
                    help="Buffer the cache was generated with (must match embed_windows).")

# Head architecture
parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
parser.add_argument("--hidden-dim", type=int, default=None, help="MLP hidden width (default emb_dim).")
parser.add_argument("--n-layers", type=int, default=2, help="MLP residual blocks (--head mlp).")
parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout (--head mlp).")
parser.add_argument("--no-normalize", action="store_true",
                    help="Disable the learned de-mean/rescale standardizer in the head.")
parser.add_argument("--subtract-reference", action="store_true",
                    help="Train an FPRefDeltaSumHeadModel: subtract each window's "
                         "variant-free reference embedding before pooling, so the head "
                         "learns from deltas-from-reference. Default is the absolute "
                         "FPSumHeadModel.")
parser.add_argument("--center-windows", action="store_true",
                    help="Train an FPCenteredSumHeadModel: subtract each window's "
                         "across-sample MEAN (fit on TRAIN samples) before pooling — the "
                         "symmetric counterpart to --subtract-reference (no window is "
                         "forced to zero). NOTE: with the learned standardizer on this is "
                         "a constant shift it absorbs, so it equals the absolute head; it "
                         "only differs from absolute under --no-normalize (the meaningful "
                         "comparison vs --subtract-reference). Mutually exclusive with it.")
parser.add_argument("--pool", choices=["sum", "mean"], default="sum",
                    help="How windows are pooled into the per-sample vector. 'sum' "
                         "magnitude grows with #windows; 'mean' is window-count "
                         "invariant. The pre-pool here and the head's forward use this "
                         "same mode, and it's saved in head_config for reconstruction.")
parser.add_argument("--center-ln-pool", action="store_true",
                    help="DIAGNOSTIC aggregator: instead of plain pooling, mean-pool the "
                         "per-window-CENTERED, per-window-LayerNorm'd (parameter-free) "
                         "embeddings — exactly what an AttentionHead sees under uniform "
                         "('random') attention weights. Gives the floor attention should "
                         "beat. Forces mean pooling; incompatible with --subtract-reference. "
                         "The saved checkpoint does NOT reproduce this transform at inference "
                         "(it's for measuring val pcc, not deployment).")

# Warm-starting (2.a / 2.b — mutually exclusive)
parser.add_argument("--warm-start-head", type=str, default=None,
                    help="Path to a pretrained head checkpoint to load_state_dict from.")
parser.add_argument("--warm-start-standardizer", action="store_true",
                    help="Fit the standardizer on the training embeddings before training.")

# Optimization
parser.add_argument("--epochs", type=int, default=100,
                    help="Number of full passes over the training set.")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 decay, applied only to inner Linear weights.")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--log-every", type=int, default=20,
                    help="Log train metrics every N optimizer steps. Val metrics "
                         "are logged once per epoch over the full split.")
parser.add_argument("--track-activations", action="store_true",
                    help="Log per-layer activation/input summary stats (mean/std/absmax/"
                         "frac_zero) alongside train metrics, via forward hooks on the "
                         "head's leaf modules. Captured only on --log-every steps.")
parser.add_argument("--seed", type=int, default=42)

# wandb
parser.add_argument("--wandb-project", type=str, default=None,
                    help="If set, log training to this wandb project.")
parser.add_argument("--wandb-name", type=str, default=None,
                    help="Optional wandb run name; defaults to wandb's auto name.")

parser.add_argument("--split", type=str, default="splits/sativas413_seed42.pt",
                    help="Path to a cached train/val split .pt file. Must be built from "
                         "the same VCF and windowing as the cache. Generate with "
                         "python scripts/cache_split.py --output <SPLIT_PATH>.")
# I/O
parser.add_argument("--output", type=str, required=True, help="Path to write model.pt.")

args = parser.parse_args()
torch.manual_seed(args.seed)

if args.warm_start_head and args.warm_start_standardizer:
    parser.error(
        "--warm-start-head and --warm-start-standardizer are mutually exclusive: a "
        "pretrained head already carries a standardizer fit on its own embeddings."
    )

if args.subtract_reference and args.center_windows:
    parser.error("--subtract-reference and --center-windows are mutually exclusive.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local JSONL metrics sidecar (always on) + optional wandb. See
# crop_embed/logging_utils.py and notebooks/track_training.ipynb.
logger = MetricLogger(
    metrics_path_for(args.output),
    wandb_project=args.wandb_project,
    wandb_name=args.wandb_name,
    config=vars(args),
)
print(f"Logging metrics to {logger.metrics_path}")

# ── Dataset, targets, and the shared train/val split ──────────────────────────
# Generate the split first with: python scripts/cache_split.py --output <SPLIT_PATH>


SPLIT_PATH = args.split
data = prepare_data(split_path=SPLIT_PATH,
                    half_window=args.half_window, buffer=args.buffer)

dataset    = data["dataset"]
Y          = data["Y"]
trait_cols = data["trait_cols"]
train_idx  = data["train_idx"]
val_idx    = data["val_idx"]

# ── 1. Load fixed window cache ────────────────────────────────────────────────

print(f"\nLoading cache from {args.cache} …")
embedder = FixedWindowEmbedder.from_file(args.cache, dataset)
cache           = embedder.cache.float()       # (n_fps, D)
sample_fp_index = embedder.sample_fp_index      # (n_samples, n_windows)
emb_dim  = cache.shape[1]
n_traits = Y.shape[1]

# The cache's sample→fingerprint index must line up with the dataset the split
# was built from; otherwise train_idx/val_idx point at the wrong samples.
if (sample_fp_index.shape != dataset.sample_fp_index.shape
        or not torch.equal(sample_fp_index, dataset.sample_fp_index)):
    raise SystemExit(
        "Cache sample→fingerprint index doesn't match the dataset built from the "
        "split's VCF/windowing. Regenerate the cache for this windowing."
    )
print(f"  {cache.shape[0]:,} fingerprints × {emb_dim} dims; {sample_fp_index.shape[0]} samples")

# ── 1.a Pre-sum into one embedding per sample ─────────────────────────────────
# Frozen cache → the per-sample sum is constant, so compute it once. embedding_bag
# fuses the gather+sum so the (n_samples, n_windows, D) intermediate never exists.
# With --subtract-reference we pool each window's delta-from-reference instead of its
# absolute embedding; ref_index[i] is the cache row of fingerprint i's variant-free
# reference window. The delta table is likewise frozen, so it pre-sums the same way.

if args.center_ln_pool:
    # Diagnostic: mean-pool the per-window-centered, per-window-LayerNorm'd windows
    # — the exact representation an AttentionHead averages under uniform attention.
    # Done in sample-chunks so the (n_samples, n_windows, D) gather never materializes.
    if args.subtract_reference:
        parser.error("--center-ln-pool is incompatible with --subtract-reference")
    ref_index = None
    cache_d = cache.to(device)
    sfi_d   = sample_fp_index.to(device)
    wmean   = window_means(cache_d, sfi_d, train_idx.to(device))      # (n_windows, D), train-only
    parts = []
    for i in range(0, sfi_d.shape[0], 16):
        g = cache_d[sfi_d[i:i + 16]] - wmean                          # center  (b, nw, D)
        g = F.layer_norm(g, (g.shape[-1],))                          # param-free per-window LN
        parts.append(g.mean(dim=1).cpu())                            # mean-pool over windows
    summed = torch.cat(parts, dim=0)                                 # (n_samples, D)
    print(f"  center-ln-pool: mean over centered+LayerNorm'd windows (uniform-attention floor)")
elif args.subtract_reference:
    ref_index = FPRefDeltaSumHeadModel.build_ref_index(dataset.unique_fingerprints)
    table = FPRefDeltaSumHeadModel.subtract_reference(cache, ref_index)   # (n_fps, D)
    summed = F.embedding_bag(sample_fp_index, table, mode=args.pool)      # (n_samples, D)
elif args.center_windows:
    ref_index = None
    window_center = FPCenteredSumHeadModel.build_window_center(
        cache.to(device), sample_fp_index.to(device), train_idx.to(device)
    ).cpu()                                                               # (n_fps, D), train-only
    table = FPCenteredSumHeadModel.subtract_center(cache, window_center)  # (n_fps, D)
    summed = F.embedding_bag(sample_fp_index, table, mode=args.pool)      # (n_samples, D)
    print(f"  center-windows: pool each window's deviation from its across-sample mean")
else:
    ref_index = None
    table = cache
    # Pre-pool with the SAME mode the head's forward uses, so forward_postsum trains on
    # the same scale the head would see at inference (--pool, default "sum").
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

# 2.b warm-start the standardizer from the *training* embeddings only (no val leak).
# When --subtract-reference, `summed` is the per-sample delta sum, so the standardizer
# is fit on the delta distribution the head actually sees.
warm_start_embeddings = summed[train_idx] if args.warm_start_standardizer else None
if args.subtract_reference:
    head = FPRefDeltaSumHeadModel(
        inner, emb_dim=emb_dim, ref_index=ref_index,
        normalize=not args.no_normalize,
        warm_start_embeddings=warm_start_embeddings,
        pool=args.pool,
    ).to(device)
elif args.center_windows:
    head = FPCenteredSumHeadModel(
        inner, emb_dim=emb_dim, window_center=window_center,
        normalize=not args.no_normalize,
        warm_start_embeddings=warm_start_embeddings,
        pool=args.pool,
    ).to(device)
else:
    head = FPSumHeadModel(
        inner, emb_dim=emb_dim,
        normalize=not args.no_normalize,
        warm_start_embeddings=warm_start_embeddings,
        pool=args.pool,
    ).to(device)

# 2.a warm-start the whole head from a pretrained checkpoint (same architecture).
if args.warm_start_head:
    print(f"Warm-starting head from {args.warm_start_head}")
    ckpt = torch.load(args.warm_start_head, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["head_state_dict"])

# Weight decay only on inner Linear weights (not biases, not the standardizer).
linear_weight_ids = {id(m.weight) for m in head.modules() if isinstance(m, torch.nn.Linear)}
decay, nodecay = [], []
for p in head.parameters():
    (decay if id(p) in linear_weight_ids else nodecay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay},
     {"params": nodecay, "weight_decay": 0.0}],
    lr=args.lr,
)

n_params = sum(p.numel() for p in head.parameters())
head_label = f"refdelta-{args.head}" if args.subtract_reference else args.head
print(f"Head: {head_label} ({n_params:,} params)  normalize={not args.no_normalize}")

# Optional per-layer activation/input tracking via forward hooks (no forward-pass
# edits). Armed only on --log-every steps so non-logged steps stay overhead-free.
tracker = ActivationTracker(head) if args.track_activations else None

# ── 3. Train ──────────────────────────────────────────────────────────────────
# Explicit epochs: each pass over train_loader shows the model every training
# sample exactly once, so --epochs is the number of times it sees each sample.
# Train metrics log k steps; val is the full split once per epoch. 

n_train = len(train_ds)
print(f"\nTraining {args.epochs} epochs over {n_train} samples at lr={args.lr} …")
step = 0
for epoch in range(args.epochs):
    head.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        is_log = step % args.log_every == 0
        if tracker is not None and is_log:
            tracker.arm()                       # record this forward's activations
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

    # End-of-epoch validation over the full split.
    head.eval()
    with torch.no_grad():
        val_m = _compute_metrics(head.forward_postsum(val_x), val_y, trait_cols, "val")
    print(f"epoch {epoch:4d}  [val]"
          f"  val_loss={val_m.get('val/mean_mse', float('nan')):.4f}"
          f"  val_pcc={val_m.get('val/mean_pcc', float('nan')):.3f}")
    logger.log({"epoch": epoch, "step": step, **val_m})

# ── 4. Save head + reconstruction metadata ────────────────────────────────────

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "head_state_dict": head.state_dict(),
    "head_config": {
        "head": args.head,
        "model_class": ("FPRefDeltaSumHeadModel" if args.subtract_reference
                        else "FPCenteredSumHeadModel" if args.center_windows
                        else "FPSumHeadModel"),
        "subtract_reference": args.subtract_reference,
        "center_windows": args.center_windows,
        "pool": args.pool,
        "emb_dim": emb_dim,
        "n_traits": n_traits,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "normalize": not args.no_normalize,
    },
    "cache_path": args.cache,
    "split_path": SPLIT_PATH,
    "trait_cols": trait_cols,
    "sample_ids": dataset.samples,
    "args": vars(args),
}, out_path)
print(f"\nSaved to {out_path}")

if tracker is not None:
    tracker.remove()
logger.close()

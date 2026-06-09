"""
train_refdelta_head.py
----------------------
Like train_head.py, but trains an FPRefDeltaSumHeadModel: each window's
variant-free *reference* embedding is subtracted from the fingerprint embedding
before pooling, so the head learns from per-window deltas-from-reference rather
than absolute embeddings. See crop_embed.models.fp_head_model.FPRefDeltaSumHeadModel.

Kept separate from train_head.py so the plain FPSumHeadModel path is untouched.
The cache, split, dataset, optimizer, and training loop are otherwise identical;
the only differences are (1) building the reference index, (2) pre-summing the
reference-subtracted table, and (3) warm-starting the standardizer on summed
*deltas* (the distribution this head actually sees).

Example
-------
    python train_pipeline/train_refdelta_head.py \\
        --cache checkpoints/sweep/sativas413_hw500.ckpt.pt \\
        --half-window 500 --head mlp --epochs 100 --lr 1e-3 \\
        --warm-start-standardizer \\
        --output trained_heads/sweep_refdelta/mlp_hw500/model.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_embed.models.fp_head_model import (
    MLPModel, LinearModel, FPRefDeltaSumHeadModel,
)
from crop_embed import FixedWindowEmbedder, MetricLogger, metrics_path_for
from crop_embed.data.loading import prepare_data
from crop_embed.train import masked_mse, _compute_metrics

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train a reference-delta head on precomputed window embeddings.")
parser.add_argument("--cache", type=str, required=True,
                    help="FixedWindowEmbedder cache (.pt) from embed_windows.py / generate_cache.py.")

# Windowing — must match how the cache was generated (mirrors train_head.py).
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

# Warm-starting (mutually exclusive)
parser.add_argument("--warm-start-head", type=str, default=None,
                    help="Path to a pretrained head checkpoint to load_state_dict from.")
parser.add_argument("--warm-start-standardizer", action="store_true",
                    help="Fit the standardizer on the training (delta) embeddings before training.")

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
parser.add_argument("--seed", type=int, default=42)

# wandb
parser.add_argument("--wandb-project", type=str, default=None,
                    help="If set, log training to this wandb project.")
parser.add_argument("--wandb-name", type=str, default=None,
                    help="Optional wandb run name; defaults to wandb's auto name.")

# I/O
parser.add_argument("--output", type=str, required=True, help="Path to write model.pt.")

args = parser.parse_args()
torch.manual_seed(args.seed)

if args.warm_start_head and args.warm_start_standardizer:
    parser.error(
        "--warm-start-head and --warm-start-standardizer are mutually exclusive: a "
        "pretrained head already carries a standardizer fit on its own embeddings."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = MetricLogger(
    metrics_path_for(args.output),
    wandb_project=args.wandb_project,
    wandb_name=args.wandb_name,
    config=vars(args),
)
print(f"Logging metrics to {logger.metrics_path}")

# ── Dataset, targets, and the shared train/val split ──────────────────────────

SPLIT_PATH = "splits/sativas413_seed42.pt"
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
# was built from; this also guarantees dataset.unique_fingerprints is in the same
# cache-row order the reference index is built against.
if (sample_fp_index.shape != dataset.sample_fp_index.shape
        or not torch.equal(sample_fp_index, dataset.sample_fp_index)):
    raise SystemExit(
        "Cache sample→fingerprint index doesn't match the dataset built from the "
        "split's VCF/windowing. Regenerate the cache for this windowing."
    )
print(f"  {cache.shape[0]:,} fingerprints × {emb_dim} dims; {sample_fp_index.shape[0]} samples")

# ── 1.a Reference index + reference-subtracted, pre-summed embeddings ──────────
# ref_index[i] = cache row of fingerprint i's variant-free reference window. We
# subtract that baseline, then pre-sum per sample exactly like train_head does
# (frozen cache → the per-sample sum of deltas is constant).

ref_index   = FPRefDeltaSumHeadModel.build_ref_index(dataset.unique_fingerprints)
delta_cache = FPRefDeltaSumHeadModel.subtract_reference(cache, ref_index)   # (n_fps, D)
summed      = F.embedding_bag(sample_fp_index, delta_cache, mode="sum")     # (n_samples, D)

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

# 2.b warm-start the standardizer from the *training* deltas only (no val leak).
warm_start_embeddings = summed[train_idx] if args.warm_start_standardizer else None
head = FPRefDeltaSumHeadModel(
    inner, emb_dim=emb_dim, ref_index=ref_index,
    normalize=not args.no_normalize,
    warm_start_embeddings=warm_start_embeddings,
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
print(f"Head: refdelta-{args.head} ({n_params:,} params)  normalize={not args.no_normalize}")

# ── 3. Train ──────────────────────────────────────────────────────────────────
# Identical loop to train_head.py — only the inputs (summed deltas) differ.

n_train = len(train_ds)
print(f"\nTraining {args.epochs} epochs over {n_train} samples at lr={args.lr} …")
step = 0
for epoch in range(args.epochs):
    head.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        pred = head.forward_postsum(x)
        loss = masked_mse(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            train_m = _compute_metrics(pred.detach(), y, trait_cols, "train")
            print(f"epoch {epoch:4d} step {step:6d}"
                  f"  loss={train_m.get('train/mean_mse', loss.item()):.4f}"
                  f"  pcc={train_m.get('train/mean_pcc', float('nan')):.3f}")
            logger.log({"epoch": epoch, "step": step, **train_m})
        step += 1

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
    "head_state_dict": head.state_dict(),   # includes the ref_index buffer
    "head_config": {
        "head": args.head,
        "model_class": "FPRefDeltaSumHeadModel",
        "subtract_reference": True,
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

logger.close()

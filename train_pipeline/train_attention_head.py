"""
train_attention_head.py
------------------------
Train an *attention* aggregator head on precomputed window embeddings — the
set-based counterpart to train_head.py's mean/sum pooling.

Why a separate script. train_head.py exploits a frozen cache: for sum/mean
pooling the per-sample vector never changes, so it pre-pools the whole cache
into one (n_samples, D) tensor *once* and trains through `forward_postsum`. An
attention aggregator can't be pre-pooled — the aggregation is *learned* and
depends on every window's embedding — so we keep the per-window axis and feed
the full (B, n_windows, D) set to the head each step.

Model. The head is `FPGatherHeadModel(AttentionHead(...))`: the gather wrapper
materializes (B, n_windows, D) from the cache (cache[gather_ind]) and hands it to
the AttentionHead, where K learnable queries cross-attend to the windows →
(B, K, D) → MLP → n_traits. Same `(fp_emb, gather_ind)` call signature as the
pooling heads, so the saved checkpoint reconstructs the same way. The cache is
frozen (no grad through the gather); only the head trains.

Memory. The (B, n_windows, D) gather is the cost — at hw1000 that's ~20.6k
windows × 768 dims. So train batches are small (--batch-size, default 8) and the
validation pass is chunked (--eval-chunk) rather than run over the whole split at
once. AttentionHead's K queries make the attention itself cheap (scores are
(B, K, n_windows)); the linear K/V projections over all windows dominate.

Metrics land in the same JSONL sidecar (train/val mean_mse, mean_pcc, and
per-trait mse/pcc) as train_head.py, so the track_training notebook plots these
runs identically.

Example
-------
    python train_pipeline/train_attention_head.py \\
        --cache checkpoints/sweep/sativas413_hw1000.ckpt.pt \\
        --half-window 1000 --buffer 0 \\
        --n-queries 8 --n-heads 4 --epochs 50 --lr 1e-3 \\
        --output trained_heads/attention/hw1000_abs/model.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_embed.models.fp_head_model import (
    AttentionHead, FPGatherHeadModel, window_position_features,
)
from crop_embed import FixedWindowEmbedder, MetricLogger, metrics_path_for, ActivationTracker
from crop_embed.data.loading import prepare_data
from crop_embed.train import masked_mse, _compute_metrics

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train an attention aggregator head on precomputed window embeddings.")
parser.add_argument("--cache", type=str, required=True,
                    help="FixedWindowEmbedder cache (.pt) from embed_windows.py / generate_cache.py.")

# Windowing — must match how the cache was generated (same identity check as train_head.py).
parser.add_argument("--half-window", type=int, default=1000,
                    help="Half-window the cache was generated with (must match embed_windows).")
parser.add_argument("--buffer", type=int, default=0,
                    help="Buffer the cache was generated with (must match embed_windows).")

# Attention head architecture
parser.add_argument("--n-queries", type=int, default=8,
                    help="K — number of learnable query vectors (distinct genome 'summaries').")
parser.add_argument("--n-heads", type=int, default=4,
                    help="Attention heads in the cross-attention layer (must divide emb_dim).")
parser.add_argument("--hidden-dim", type=int, default=None,
                    help="MLP hidden width after attention. Defaults to emb_dim.")
parser.add_argument("--positional", action="store_true",
                    help="Add a learnable per-chromosome embedding + fixed sinusoidal "
                         "position encoding to each window before attention (else the "
                         "head treats windows as an unordered bag).")

# Optimization
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of full passes over the training set.")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 decay, applied only to inner Linear weights.")
parser.add_argument("--batch-size", type=int, default=8,
                    help="Samples per step. Small by default — each sample materializes "
                         "an (n_windows, D) set (~20.6k×768 at hw1000).")
parser.add_argument("--eval-chunk", type=int, default=None,
                    help="Samples per forward during the chunked val pass. Defaults to "
                         "--batch-size; raise/lower to trade speed for memory.")
parser.add_argument("--log-every", type=int, default=10,
                    help="Log train metrics every N optimizer steps. Val metrics "
                         "are logged once per epoch over the full split.")
parser.add_argument("--track-activations", action="store_true",
                    help="Log per-layer activation/input summary stats via forward hooks "
                         "on the head's leaf modules. Captured only on --log-every steps.")
parser.add_argument("--seed", type=int, default=42)

# wandb
parser.add_argument("--wandb-project", type=str, default=None,
                    help="If set, log training to this wandb project.")
parser.add_argument("--wandb-name", type=str, default=None,
                    help="Optional wandb run name; defaults to wandb's auto name.")

parser.add_argument("--split", type=str, default="splits/sativas413_seed42.pt",
                    help="Path to a cached train/val split .pt file. Must be built from "
                         "the same VCF and windowing as the cache.")
# I/O
parser.add_argument("--output", type=str, required=True, help="Path to write model.pt.")

args = parser.parse_args()
torch.manual_seed(args.seed)
eval_chunk = args.eval_chunk or args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local JSONL metrics sidecar (always on) + optional wandb.
logger = MetricLogger(
    metrics_path_for(args.output),
    wandb_project=args.wandb_project,
    wandb_name=args.wandb_name,
    config=vars(args),
)
print(f"Logging metrics to {logger.metrics_path}")

# ── Dataset, targets, and the shared train/val split ──────────────────────────

SPLIT_PATH = args.split
data = prepare_data(split_path=SPLIT_PATH,
                    half_window=args.half_window, buffer=args.buffer)

dataset    = data["dataset"]
Y          = data["Y"]
trait_cols = data["trait_cols"]
train_idx  = data["train_idx"]
val_idx    = data["val_idx"]

# ── 1. Load fixed window cache (kept per-window — NOT pre-pooled) ──────────────

print(f"\nLoading cache from {args.cache} …")
embedder = FixedWindowEmbedder.from_file(args.cache, dataset)
cache           = embedder.cache.float().to(device)   # (n_fps, D) — frozen, no grad
sample_fp_index = embedder.sample_fp_index            # (n_samples, n_windows)
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
sample_fp_index = sample_fp_index.to(device)
n_windows = sample_fp_index.shape[1]
print(f"  {cache.shape[0]:,} fingerprints × {emb_dim} dims; "
      f"{sample_fp_index.shape[0]} samples × {n_windows:,} windows/sample")

# ── 2. Build the attention head (+ optimizer) ─────────────────────────────────
# FPGatherHeadModel materializes cache[gather_ind] = (B, n_windows, D) and feeds
# the full set to the AttentionHead. The cache is frozen, so the only trained
# params are in the head.

pos_kwargs = {}
if args.positional:
    # Per-window (chrom, center_bp) aligned to the n_windows axis (the columns of
    # sample_fp_index == partitioner.windows order).
    w_chroms, w_positions = window_position_features(dataset)
    pos_kwargs = dict(positional=True, window_chroms=w_chroms, window_positions=w_positions)

inner = AttentionHead(
    emb_dim, n_traits,
    n_queries=args.n_queries, n_heads=args.n_heads, hidden_dim=args.hidden_dim,
    **pos_kwargs,
)
head = FPGatherHeadModel(inner).to(device)

# Weight decay only on inner Linear weights (not biases, LayerNorm, queries).
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
print(f"Head: attention (n_queries={args.n_queries}, n_heads={args.n_heads}, "
      f"positional={args.positional}) — {n_params:,} params")

tracker = ActivationTracker(head) if args.track_activations else None

# ── 3. Train ──────────────────────────────────────────────────────────────────
# Batch by SAMPLE INDEX (not pre-pooled vectors): each step gathers the batch's
# (B, n_windows, D) window sets from the frozen cache and runs the attention head.

train_loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)
Y = Y.to(device)


@torch.no_grad()
def predict_chunked(sample_idx: torch.Tensor) -> torch.Tensor:
    """Forward over sample_idx in --eval-chunk groups → (len, n_traits) on device.
    Chunked because (len_val, n_windows, D) won't fit in one gather."""
    head.eval()
    preds = []
    for i in range(0, len(sample_idx), eval_chunk):
        b = sample_idx[i:i + eval_chunk].to(device)
        preds.append(head(cache, sample_fp_index[b]))
    return torch.cat(preds, dim=0)


n_train = len(train_idx)
print(f"\nTraining {args.epochs} epochs over {n_train} samples at lr={args.lr} "
      f"(batch_size={args.batch_size}, eval_chunk={eval_chunk}) …")
step = 0
for epoch in range(args.epochs):
    head.train()
    for (b_idx,) in train_loader:
        b_idx = b_idx.to(device)
        gather_ind = sample_fp_index[b_idx]          # (B, n_windows)
        y = Y[b_idx]

        is_log = step % args.log_every == 0
        if tracker is not None and is_log:
            tracker.arm()
        pred = head(cache, gather_ind)               # (B, n_traits)
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

    # End-of-epoch validation over the full split (chunked forward).
    val_pred = predict_chunked(val_idx)
    val_m = _compute_metrics(val_pred, Y[val_idx], trait_cols, "val")
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
        "head": "attention",
        "model_class": "FPGatherHeadModel",
        "inner_class": "AttentionHead",
        "emb_dim": emb_dim,
        "n_traits": n_traits,
        "n_queries": args.n_queries,
        "n_heads": args.n_heads,
        "hidden_dim": args.hidden_dim,
        "positional": args.positional,
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

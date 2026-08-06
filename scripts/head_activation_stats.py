"""
scripts/head_activation_stats.py
--------------------------------
Signal-propagation + fit diagnostic for the emb_nn MLP head (the exact model
training/emb_nn/run.py builds: mean-pool windows -> learned per-dim standardizer
-> MLPModel of `input_proj` + N residual blocks -> linear output).

Reports, for an 8-layer (default) head on a given embedding cache:

  1. Average per-dim activation mean & var at EACH layer -- the initial
     (standardized) embedding, the input projection, and every residual block --
     AT INITIALIZATION (random weights, standardizer UNFIT).
  2. The same POST-INITIALIZATION, i.e. after the warm-start standardizer is fit
     to the train embeddings (the init the trainer actually uses), still before
     any gradient step. [Also a post-training snapshot, for reference.]
  3. Per-trait (per-"class") train & validation MSE and Pearson (PCC) after
     training, using the shared 70/15/15 split and z-scored targets.

"Average per-dim mean/var" = take the per-dimension mean (and variance) across
samples, then average those D numbers into one summary per layer.

    python scripts/head_activation_stats.py           # uses the CONFIG below
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from crop_embed.models.fp_head_model import MLPModel, FPSumHeadModel
from crop_embed.train import _compute_metrics, masked_mse
from training.common import features as feat
from training.common.datasets import get_dataset
from training.common.splits import get_or_build_split

# ── CONFIG ────────────────────────────────────────────────────────────────────
import os
_SCRATCH = os.environ.get("SVAR_SCRATCH", os.path.expanduser("~/svar_scratch"))
CACHE        = f"{_SCRATCH}/caches/soy/carbon3b_hw500.ckpt.pt"
DATASET      = "soy"
SEED         = 42
N_LAYERS     = 8            # residual blocks (the "8-layer network")
HIDDEN_DIM   = None         # None -> emb_dim
DROPOUT      = 0.0
POOL         = "mean"
STANDARDIZER = "perdim"     # "perdim" | "rms"
FREEZE_STD   = True         # freeze standardizer after warm-start (trainer default)
LR           = 3e-4
EPOCHS       = 60
BATCH        = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_head(warm_start_emb):
    """Fresh head with identical MLP init each call (seed reset by caller)."""
    inner = MLPModel(EMB_DIM, N_TRAITS, hidden_dim=HIDDEN_DIM,
                     n_layers=N_LAYERS, dropout=DROPOUT)
    head = FPSumHeadModel(
        inner, emb_dim=EMB_DIM, normalize=True,
        warm_start_embeddings=warm_start_emb, pool=POOL,
        standardizer=STANDARDIZER, freeze_standardizer=FREEZE_STD)
    return head.to(DEVICE)


def layer_stats(head, X):
    """avg per-dim (mean, var) at each layer for input X (B, D). Dropout off."""
    acts: "OrderedDict[str, tuple]" = OrderedDict()

    def mk(name):
        def hook(_m, _inp, out):
            a = out.detach().float()
            acts[name] = (a.mean(0).mean().item(), a.var(0, unbiased=False).mean().item())
        return hook

    hooks = [head.norm.register_forward_hook(mk("embedding (standardized)")),
             head.model.input_proj.register_forward_hook(mk("input_proj"))]
    for i, b in enumerate(head.model.blocks):
        hooks.append(b.register_forward_hook(mk(f"block_{i + 1}")))
    head.eval()
    with torch.no_grad():
        head.forward_postsum(X.to(DEVICE))
    for h in hooks:
        h.remove()
    return acts


def train_head(head):
    lin_w = {id(m.weight) for m in head.modules() if isinstance(m, nn.Linear)}
    decay, nodecay = [], []
    for p in head.parameters():
        if p.requires_grad:
            (decay if id(p) in lin_w else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-4}, {"params": nodecay, "weight_decay": 0.0}], lr=LR)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        head.train()
        for x, y in loader:
            loss = masked_mse(head.forward_postsum(x.to(DEVICE)), y.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def print_stats(title, acts, raw=None):
    print(f"\n=== {title} ===")
    print(f"  {'layer':24s} {'avg per-dim mean':>18s} {'avg per-dim var':>16s}")
    if raw is not None:
        print(f"  {'pooled (pre-norm)':24s} {raw[0]:18.4f} {raw[1]:16.4f}")
    for name, (m, v) in acts.items():
        print(f"  {name:24s} {m:18.4f} {v:16.4f}")


# ── DATA (shared with training/emb_nn) ───────────────────────────────────────
print(f"Loading cache {CACHE} …")
cached = feat.load_cached_windows(CACHE)
assert cached is not None, "cache lacks sample_ids (legacy); use a cache built by embed_windows.py"
cache = cached.cache                       # (n_fp, D) float
sfi = cached.sample_fp_index               # (n_samples, n_windows) long
samples = cached.samples
EMB_DIM = cache.shape[1]

spec = get_dataset(DATASET)
Y_np, trait_cols = feat.scaled_targets(spec, samples)   # z-scored, aligned to samples
Y = torch.tensor(Y_np, dtype=torch.float32)
N_TRAITS = Y.shape[1]

split = get_or_build_split(DATASET, seed=SEED)
tr = torch.tensor(split.indices("train", samples), dtype=torch.long)
va = torch.tensor(split.indices("val", samples), dtype=torch.long)

print(f"Pooling ({POOL}) {sfi.shape[0]} samples × {sfi.shape[1]} windows → per-sample vectors …")
summed = F.embedding_bag(sfi, cache, mode=POOL)         # (n_samples, D)
Xtr, Ytr = summed[tr], Y[tr]
Xva, Yva = summed[va], Y[va]
raw_stats = (Xtr.mean(0).mean().item(), Xtr.var(0, unbiased=False).mean().item())
print(f"emb_dim={EMB_DIM}  traits={N_TRAITS}  train={len(tr)}  val={len(va)}  "
      f"n_layers={N_LAYERS}  hidden={HIDDEN_DIM or EMB_DIM}  dropout={DROPOUT}")

# ── 1. AT INITIALIZATION (random weights, standardizer UNFIT) ────────────────
torch.manual_seed(SEED)
head_init = build_head(warm_start_emb=None)
print_stats("1. at initialization (random weights, standardizer UNFIT)",
            layer_stats(head_init, Xtr), raw=raw_stats)

# ── 2. POST-INITIALIZATION (warm-started standardizer, pre-training) ─────────
torch.manual_seed(SEED)                                 # identical MLP init
head = build_head(warm_start_emb=Xtr)                   # standardizer fit to train
print_stats("2. post-initialization (warm-started standardizer, pre-training)",
            layer_stats(head, Xtr), raw=raw_stats)

# ── train, then a post-training snapshot for reference ───────────────────────
print(f"\nTraining {EPOCHS} epochs (lr={LR}) …")
train_head(head)
print_stats("(reference) post-training", layer_stats(head, Xtr), raw=raw_stats)

# ── 3. PER-TRAIT (per-class) TRAIN / VAL  MSE & PCC ──────────────────────────
head.eval()
with torch.no_grad():
    ptr = head.forward_postsum(Xtr.to(DEVICE))
    pva = head.forward_postsum(Xva.to(DEVICE))
mtr = _compute_metrics(ptr, Ytr.to(DEVICE), trait_cols, "train")
mva = _compute_metrics(pva, Yva.to(DEVICE), trait_cols, "val")

print("\n=== 3. per-trait train / val  MSE & PCC (z-scored targets) ===")
print(f"  {'trait':10s} {'train_mse':>10s} {'train_pcc':>10s} {'val_mse':>10s} {'val_pcc':>10s}")
for t in trait_cols:
    print(f"  {t:10s} {mtr.get(f'train/mse/{t}', float('nan')):10.4f} "
          f"{mtr.get(f'train/pcc/{t}', float('nan')):10.4f} "
          f"{mva.get(f'val/mse/{t}', float('nan')):10.4f} "
          f"{mva.get(f'val/pcc/{t}', float('nan')):10.4f}")
print(f"  {'MEAN':10s} {mtr.get('train/mean_mse', float('nan')):10.4f} "
      f"{mtr.get('train/mean_pcc', float('nan')):10.4f} "
      f"{mva.get('val/mean_mse', float('nan')):10.4f} "
      f"{mva.get('val/mean_pcc', float('nan')):10.4f}")

"""
train_attention_head.py
------------------------
Train the cross-attention head (AttentionHead) over ALL cached windows/SNPs,
using the same simple epoch/DataLoader structure as the baseline scripts.

Unlike the pooled-vector models, this head needs the full (B, n_windows, D)
window matrix per sample. Materializing it for every sample at once is ~24 GB
for the default rice config, so instead the DataLoader carries sample *indices*
and we expand each batch on the fly via FixedWindowEmbedder.forward(idx).

Uses the same single 70/30 split, masked MSE, and per-trait MSE/PCC metrics as
model_comparison_70_30.ipynb so the result is directly comparable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crop_embed import FixedWindowEmbedder, AttentionHead

CACHE_PATH = Path("/home/andrew.dickson/svar/checkpoints/sativas413_1000.ckpt.pt")
PHENO_PATH = Path("/home/andrew.dickson/rice_data/RiceDiversity_44K_Phenotypes_34traits_PLINK.txt")

SEED      = 45      # match the comparison notebook's split
TEST_SIZE = 0.10
EPOCHS    = 15
LR        = 1e-4
WD        = 1e-4
BATCH     = 8        # per-batch matrix peaks at BATCH * n_windows * D * 4 bytes; lower if OOM
N_QUERIES = 8
N_HEADS   = 4
FREEZE_ATTENTION = False   # zero+freeze queries -> exactly uniform attention; only the MLP
                          # trains, so the head reduces to mean-pool + MLP (~ LinearHead).
BYPASS_NORM = True        # drop the per-window LayerNorm: it re-normalizes each window across
                          # the embedding dim and undoes the §2b standardization before pooling.

# Where to persist results for the comparison notebook to load.
RESULTS_DIR = Path(__file__).resolve().parent / "comparison_results"
SUMMARY_CSV = RESULTS_DIR / "attention_head_summary.csv"
DETAIL_NPZ  = RESULTS_DIR / "attention_head_detail.npz"

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Generate data up front ──────────────────────────────────────────────
cache_blob = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
cache_t         = cache_blob["cache"].float()            # (n_unique_windows, emb_dim)
sample_fp_index = cache_blob["sample_fp_index"].long()   # (n_samples, n_windows)
cache_samples   = cache_blob.get("sample_ids")
if cache_samples is None:
    raise ValueError("Cache file is missing 'sample_ids' — regenerate with generate_cache.py.")

n_samples_cache, n_windows = sample_fp_index.shape
n_unique, emb_dim          = cache_t.shape
print(f"cache {tuple(cache_t.shape)}  sample_fp_index {tuple(sample_fp_index.shape)}  emb_dim={emb_dim}")

pheno_raw = pd.read_csv(PHENO_PATH, sep="\t")
pheno_raw["NSFTVID"] = pheno_raw["NSFTVID"].astype(int)
TRAIT_COLS = [c for c in pheno_raw.columns if c not in ("HybID", "NSFTVID")]

cache_nsftvid  = [int(s.rsplit("_", 1)[-1]) for s in cache_samples]
cache_id_to_ix = {nid: i for i, nid in enumerate(cache_nsftvid)}

pheno_matched = (
    pheno_raw[pheno_raw["NSFTVID"].isin(cache_id_to_ix)]
    .copy()
    .reset_index(drop=True)
)
cache_row_order = np.asarray(
    [cache_id_to_ix[nid] for nid in pheno_matched["NSFTVID"]], dtype=np.int64
)

Y_raw    = pheno_matched[TRAIT_COLS].values.astype(float)
Y_scaled = np.full_like(Y_raw, np.nan)
for j in range(Y_raw.shape[1]):
    col  = Y_raw[:, j]
    mask = ~np.isnan(col)
    if mask.sum() < 2:
        continue
    mu = col[mask].mean()
    sd = col[mask].std(ddof=0)
    Y_scaled[mask, j] = (col[mask] - mu) / (sd if sd > 0 else 1.0)

n_traits = Y_scaled.shape[1]
print(f"matched {len(pheno_matched)} / {len(pheno_raw)}   Y_scaled {Y_scaled.shape}")


# ── 2. Single 70/30 split (same partition as the comparison notebook) ──────
# Split row indices into pheno_matched order; train_test_split with the same
# (test_size, random_state) yields the identical partition the notebook uses.
all_rows = np.arange(len(pheno_matched))
train_rows, test_rows = train_test_split(all_rows, test_size=TEST_SIZE, random_state=SEED)

# Map matched rows -> cache sample indices (what FixedWindowEmbedder expects).
train_idx = torch.as_tensor(cache_row_order[train_rows], dtype=torch.long)
test_idx  = torch.as_tensor(cache_row_order[test_rows],  dtype=torch.long)
y_train   = Y_scaled[train_rows]
y_test    = Y_scaled[test_rows]
print(f"train {len(train_idx)}  test {len(test_idx)}  traits {n_traits}")


# ── 2b. Per-feature standardization (fit on train) ─────────────────────────
# The raw mean-pooled embeddings differ by only ~5e-6 across samples vs a ~7e-2
# magnitude — the cross-sample signal is ~1/13000 of the input scale, so an
# unstandardized head sits at a near-constant input and can't fit (flat train
# MSE). Every model that works (LinearHead/MLP/sklearn) runs on StandardScaler'd
# features. Mean-pool is linear, so normalizing each window by the *pooled* train
# stats makes the pooled summary unit-variance and the per-window signal O(1).
fixed = FixedWindowEmbedder(cache_t, sample_fp_index).to(device).eval()
with torch.no_grad():
    pooled = torch.empty((n_samples_cache, emb_dim))
    for i in range(0, n_samples_cache, BATCH):
        j = min(i + BATCH, n_samples_cache)
        pooled[i:j] = fixed(torch.arange(i, j, device=device)).mean(dim=1).cpu()

train_pooled = pooled[train_idx]
emb_mu = train_pooled.mean(dim=0, keepdim=True)
emb_sd = train_pooled.std(dim=0, keepdim=True).clamp_min(1e-8)
cache_t = (cache_t - emb_mu) / emb_sd                       # bake standardization into the cache
fixed = FixedWindowEmbedder(cache_t, sample_fp_index).to(device).eval()
print(f"raw pooled cross-sample std {pooled.std(0).mean():.2e} -> standardized to ~1.0")


# ── 3. Metric harness (identical to the comparison notebook) ───────────────
def masked_mse(pred, target):
    m = ~torch.isnan(target)
    diff = (pred - torch.nan_to_num(target)) * m
    return (diff ** 2).sum(dim=0) / m.sum(dim=0).clamp(min=1)


def trait_metrics(pred, targ):
    n_t = targ.shape[1]
    mse = np.full(n_t, np.nan)
    pcc = np.full(n_t, np.nan)
    for j in range(n_t):
        m = ~np.isnan(targ[:, j]) & ~np.isnan(pred[:, j])
        if m.sum() < 2:
            continue
        p, t = pred[m, j], targ[m, j]
        mse[j] = np.mean((p - t) ** 2)
        if p.std() > 0 and t.std() > 0:
            pcc[j] = np.corrcoef(p, t)[0, 1]
    return {'mse_mean': float(np.nanmean(mse)), 'pcc_mean': float(np.nanmean(pcc)),
            'mse': mse, 'pcc': pcc}


# ── 4. Model + dataloaders ─────────────────────────────────────────────────
# Datasets carry sample INDICES (+ targets); the (B, n_windows, D) matrix is
# built per batch via fixed(idx) inside the loop. `fixed` (standardized cache)
# was built in §2b above.
head  = AttentionHead(
    emb_dim=emb_dim, n_traits=n_traits, n_queries=N_QUERIES, n_heads=N_HEADS,
).to(device)

if BYPASS_NORM:
    # forward() calls self.norm(window_emb); Identity makes it a no-op so the
    # standardized embeddings reach the pooling unmodified.
    head.norm = torch.nn.Identity()

if FREEZE_ATTENTION:
    # queries=0 -> all attention logits equal -> softmax is exactly uniform, so
    # pooled = a fixed linear map of the window mean. Freezing queries + the
    # attn projections leaves only norm + MLP trainable: the head becomes
    # mean-pool + MLP and should track LinearHead/MLPHead.
    with torch.no_grad():
        head.queries.zero_()
    head.queries.requires_grad_(False)
    for p in head.attn.parameters():
        p.requires_grad_(False)

trainable = [p for p in head.parameters() if p.requires_grad]
n_frozen  = sum(p.numel() for p in head.parameters() if not p.requires_grad)
print(f"trainable params {sum(p.numel() for p in trainable):,}  frozen {n_frozen:,}")
optim = torch.optim.Adam(trainable, lr=LR, weight_decay=WD)

train_ds = TensorDataset(train_idx, torch.as_tensor(y_train, dtype=torch.float32))
test_ds  = TensorDataset(test_idx,  torch.as_tensor(y_test,  dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)


def run_epoch(loader, train_mode):
    head.train(train_mode)
    sse = torch.zeros(n_traits, device=device)
    cnt = torch.zeros(n_traits, device=device)
    for idx, yb in loader:
        idx, yb = idx.to(device), yb.to(device)
        with torch.no_grad():
            window_emb = fixed(idx)              # (B, n_windows, D) built on the fly
        pred = head(window_emb)
        if train_mode:
            optim.zero_grad()
            masked_mse(pred, yb).mean().backward()
            optim.step()
        with torch.no_grad():
            m = ~torch.isnan(yb)
            sse += torch.where(m, (pred - yb) ** 2, 0.0).sum(dim=0)
            cnt += m.sum(dim=0)
    return (sse / cnt.clamp(min=1)).cpu().numpy()


# ── 5. Training loop ───────────────────────────────────────────────────────
train_hist, val_hist = [], []
for epoch in range(EPOCHS):
    tr = run_epoch(train_loader, train_mode=True)
    with torch.no_grad():
        va = run_epoch(test_loader, train_mode=False)
    train_hist.append(tr)
    val_hist.append(va)
    print(f"epoch {epoch:3d} | train MSE {tr.mean():.4f} | val MSE {va.mean():.4f}")

train_hist = np.array(train_hist)
val_hist   = np.array(val_hist)


# ── 6. Final metrics (eval mode) ───────────────────────────────────────────
def predict(idx_all):
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx_all), BATCH):
            b = idx_all[i:i + BATCH].to(device)
            out.append(head(fixed(b)).cpu())
    return torch.cat(out).numpy()


p_tr, p_te = predict(train_idx), predict(test_idx)
tr_metrics = trait_metrics(p_tr, y_train)
te_metrics = trait_metrics(p_te, y_test)

metrics_dict = {"Model": "Attention_head",
                "Train MSE Mean": tr_metrics['mse_mean'], "Val MSE Mean": te_metrics['mse_mean'],
                "Train PCC Mean": tr_metrics['pcc_mean'], "Val PCC Mean": te_metrics['pcc_mean']}
for i, trait in enumerate(TRAIT_COLS):
    metrics_dict[f'Train_{trait}_mse'] = tr_metrics['mse'][i]
    metrics_dict[f'Val_{trait}_mse']   = te_metrics['mse'][i]
    metrics_dict[f'Train_{trait}_pcc'] = tr_metrics['pcc'][i]
    metrics_dict[f'Val_{trait}_pcc']   = te_metrics['pcc'][i]

summary = pd.DataFrame([metrics_dict]).round(4)
print()
print(summary[["Model", "Train MSE Mean", "Val MSE Mean",
               "Train PCC Mean", "Val PCC Mean"]].to_string(index=False))


# ── 7. Persist results for the comparison notebook ─────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(SUMMARY_CSV, index=False)        # one row, same columns as the notebook table
np.savez(
    DETAIL_NPZ,
    train_hist=train_hist, val_hist=val_hist,    # (EPOCHS, n_traits) MSE curves
    pred_train=p_tr, pred_test=p_te,             # (n, n_traits) predictions
    y_train=y_train, y_test=y_test,              # matching targets (NaN-masked)
    trait_cols=np.array(TRAIT_COLS),
)
print(f"\nwrote {SUMMARY_CSV}\n      {DETAIL_NPZ}")

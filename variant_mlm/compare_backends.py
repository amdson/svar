"""
variant_mlm/compare_backends.py
-------------------------------
Run the EXACT model and a CACHE backend over the *same* masked variant windows
in one process and quantify the cache approximation directly: corpus CE gap,
per-window CE (exact vs cache), and per-token KL(exact || cache) — the
information the approximation loses on the masked-token prediction.

Both models get byte-identical inputs (deterministic full masking, same
collator), so every difference is the frozen-reference approximation. Saves a
matplotlib figure (+ a .npz of the raw arrays) next to --output.

Example
-------
    python variant_mlm/compare_backends.py --half-window 500 --limit 2000 \\
        --output variant_mlm/backend_compare.png
    # CPU / no flash-attn:
    python variant_mlm/compare_backends.py --attn-impl torch --device cpu --limit 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH
from DNABERT2_modules import load_dnabert2_mlm, load_dnabert2_variant_cache_mlm
from variant_mlm.data import (
    VariantMLMCollator,
    build_variant_window_dataset,
    make_loader,
)

DEFAULT_VCF_PATH = str(Path(__file__).resolve().parents[1] / "sativas413_msu7_final.vcf")

parser = argparse.ArgumentParser(
    description="Compare exact vs cache variant-MLM loss and plot the gap.")
parser.add_argument("--model-path", type=str, default="zhihan1996/DNABERT-2-117M")
parser.add_argument("--cache-backend", choices=["efficient", "bruteforce"],
                    default="efficient", help="Which cache impl to compare to exact.")
parser.add_argument("--max-length", type=int, default=2048)
parser.add_argument("--attn-impl", choices=["flash", "triton", "torch"], default=None,
                    help="Override attention impl; use 'torch' on CPU / without flash-attn.")
parser.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
parser.add_argument("--fasta-path", type=str, default=FASTA_PATH)
parser.add_argument("--half-window", type=int, default=500)
parser.add_argument("--buffer", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--limit", type=int, default=None,
                    help="Compare only the first N variant windows.")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--output", type=str, default="variant_mlm/backend_compare.png",
                    help="PNG path; raw arrays saved alongside as <output>.npz.")
args = parser.parse_args()

device = torch.device(
    args.device if args.device is not None
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
overrides = {"attn_impl": args.attn_impl} if args.attn_impl is not None else None
print(f"Device: {device}  cache-backend: {args.cache_backend}  half_window={args.half_window}")

# ── Data ──────────────────────────────────────────────────────────────────────
dataset, indices = build_variant_window_dataset(
    args.vcf_path, args.fasta_path, half_window=args.half_window, buffer=args.buffer)
if args.limit is not None:
    indices = indices[: args.limit]
print(f"  {len(indices):,} variant windows")

# ── Models (both load the same DNABERT-2 weights) ─────────────────────────────
exact, tokenizer = load_dnabert2_mlm(args.model_path, device=device,
                                     config_overrides=overrides)
cache, _ = load_dnabert2_variant_cache_mlm(
    args.model_path, device=device, backend=args.cache_backend,
    config_overrides=overrides)
exact.eval(); cache.eval()

collator = VariantMLMCollator(
    tokenizer, max_length=args.max_length,
    select_prob=1.0, mask_replace_prob=1.0, random_replace_prob=0.0)
loader = make_loader(dataset, indices, args.batch_size, shuffle=False)

# ── Compare loop ──────────────────────────────────────────────────────────────
win_ce_exact, win_ce_cache, win_kl = [], [], []   # per-window means
tok_ce_exact, tok_ce_cache, tok_kl = [], [], []   # per-token

with torch.no_grad():
    for batch in tqdm(loader, desc="compare", unit="batch"):
        enc = collator(batch)
        if enc["n_targets"] == 0:
            continue
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = enc["labels"].to(device)

        le = exact(input_ids, attention_mask=attention_mask, labels=labels,
                   return_dict=True).logits
        lc = cache(input_ids, attention_mask=attention_mask, labels=labels,
                   return_dict=True).logits

        for b in range(input_ids.shape[0]):
            tb = labels[b] > 0
            if not tb.any():
                continue
            lab = labels[b][tb]
            ce_e = F.cross_entropy(le[b][tb].float(), lab, reduction="none")
            ce_c = F.cross_entropy(lc[b][tb].float(), lab, reduction="none")
            # KL(exact || cache) per masked token.
            logp = F.log_softmax(le[b][tb].float(), dim=-1)
            logq = F.log_softmax(lc[b][tb].float(), dim=-1)
            kl = (logp.exp() * (logp - logq)).sum(-1)

            tok_ce_exact.append(ce_e); tok_ce_cache.append(ce_c); tok_kl.append(kl)
            win_ce_exact.append(ce_e.mean().item())
            win_ce_cache.append(ce_c.mean().item())
            win_kl.append(kl.mean().item())

tok_ce_exact = torch.cat(tok_ce_exact).cpu().numpy()
tok_ce_cache = torch.cat(tok_ce_cache).cpu().numpy()
tok_kl = torch.cat(tok_kl).cpu().numpy()
win_ce_exact = np.array(win_ce_exact)
win_ce_cache = np.array(win_ce_cache)
win_kl = np.array(win_kl)

# ── Summary ───────────────────────────────────────────────────────────────────
ce_e_corpus = float(tok_ce_exact.mean())
ce_c_corpus = float(tok_ce_cache.mean())
gap = win_ce_cache - win_ce_exact
print("\n── exact vs cache ──")
print(f"  windows / tokens     : {len(win_ce_exact):,} / {len(tok_ce_exact):,}")
print(f"  corpus CE  exact     : {ce_e_corpus:.5f}")
print(f"  corpus CE  cache     : {ce_c_corpus:.5f}")
print(f"  corpus CE  gap       : {ce_c_corpus - ce_e_corpus:+.5f}")
print(f"  per-window gap       : mean={gap.mean():+.5f}  median={np.median(gap):+.5f}  "
      f"p95|gap|={np.percentile(np.abs(gap), 95):.5f}")
print(f"  per-token KL(e||c)   : mean={tok_kl.mean():.5e}  median={np.median(tok_kl):.5e}  "
      f"max={tok_kl.max():.5e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

lo = min(win_ce_exact.min(), win_ce_cache.min())
hi = max(win_ce_exact.max(), win_ce_cache.max())
axes[0].scatter(win_ce_exact, win_ce_cache, s=8, alpha=0.4)
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
axes[0].set_xlabel("exact per-window CE")
axes[0].set_ylabel("cache per-window CE")
axes[0].set_title("per-window CE: exact vs cache")
axes[0].legend()

axes[1].hist(gap, bins=60, color="C0", alpha=0.8)
axes[1].axvline(0, color="k", lw=1)
axes[1].axvline(gap.mean(), color="r", ls="--", lw=1, label=f"mean={gap.mean():+.4f}")
axes[1].set_xlabel("cache CE − exact CE (per window)")
axes[1].set_ylabel("# windows")
axes[1].set_title("per-window CE gap")
axes[1].legend()

kl_pos = tok_kl[tok_kl > 0]
if kl_pos.size:
    axes[2].hist(np.log10(kl_pos), bins=60, color="C2", alpha=0.8)
    axes[2].set_xlabel("log10 KL(exact || cache)  [per masked token]")
else:
    axes[2].text(0.5, 0.5, "KL ~ 0 everywhere", ha="center", va="center")
axes[2].set_ylabel("# tokens")
axes[2].set_title("per-token prediction divergence")

fig.suptitle(
    f"variant-MLM: exact vs {args.cache_backend} cache | half_window={args.half_window} | "
    f"{len(win_ce_exact):,} windows | corpus CE {ce_e_corpus:.4f} → {ce_c_corpus:.4f} "
    f"(gap {ce_c_corpus - ce_e_corpus:+.4f})")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=130)
print(f"\nSaved plot to {out_path}")

npz_path = str(out_path) + ".npz"
np.savez(npz_path, win_ce_exact=win_ce_exact, win_ce_cache=win_ce_cache,
         win_kl=win_kl, tok_ce_exact=tok_ce_exact, tok_ce_cache=tok_ce_cache,
         tok_kl=tok_kl)
print(f"Saved raw arrays to {npz_path}")

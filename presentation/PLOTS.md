# Presentation plots — how to generate and display each result

This expands [todo.txt](todo.txt) into concrete, runnable recipes. Each section
says **what to show**, **where the data/code already lives**, the **exact
commands** to produce the numbers, and a **matplotlib snippet** to draw the
figure. All paths are relative to the repo root (`/home/andrew.dickson/svar`).

**Implemented scripts (goals 1, 3, 4).** These are written and runnable in
`presentation/` (use the env Python `/home/andrew.dickson/.conda/envs/svar/bin/python`):
- Goal 1: `count_windows.py` (→ `window_sweep.csv`) + `plot_window_sweep.py`
  (→ `figs/01_window_sweep.png`). **Generated.**
- Goal 3: `pool_cache.py` (helper) + `umap_preprocessing.py`
  (→ `figs/03_umap_grid.png`, `--reducer tsne` for t-SNE). **Generated.**
- Goal 4: `--csv-out` added to `model_dev/compare_variant_cache_embeddings.py`;
  `plot_vc_degradation.py` (→ `figs/04_vc_degradation.png`). **Scripts ready —
  the data-generation step needs the GPU.**

Conventions used below:
- Run everything in the `svar` conda env: `conda activate svar`.
- VCF: `sativas413_msu7_final.vcf` (the repo copy) or
  `/home/andrew.dickson/rice_data/sativas413_msu7_final.vcf` (full path used in
  the embed scripts). FASTA comes from `crop_embed/data/coords.py:FASTA_PATH`.
- Save figures into `presentation/figs/` (create it: `mkdir -p presentation/figs`).
- The window length `L` and the partitioner's `half_window` are related by
  `L = 2 * half_window`. The partitioner requires `buffer < half_window`.

---

## 1. Unique windows / tokens / variants vs. window length

**What to show.** Three curves as window length `L` sweeps from ~2 bp to 1e6 bp
(log x-axis):
1. **# unique windows in the dataset** — `len(UniqueWindowDataset)` =
   number of distinct fingerprints (deduplicated `(chrom, start, end,
   alt_positions)` tuples). See [pipeline_summary.md](../pipeline_summary.md) §3.
2. **mean tokens per window** — average tokenizer token count over the unique
   window sequences (saturates at `--max-length`).
3. **# unique window variants per window** — `len(dataset) / len(partitioner)`,
   the mean number of distinct haplotypes sharing a window footprint.

**Where it comes from.** No embedding or model forward is needed for curves 1
and 3 — only [crop_embed/partitioner.py](../crop_embed/partitioner.py) +
[crop_embed/dataset.py](../crop_embed/dataset.py) (which builds
`unique_fingerprints` from the VCF). Curve 2 needs a tokenizer; use Carbon's
(cheap, no model weights) via `AutoTokenizer`.

**Why a tradeoff curve.** At `L → 1` every SNP is isolated, so unique windows ≈
2×(#SNPs) (ref + alt per site) and variants/window ≈ 2. At `L → 1e6` windows
merge toward one-per-chromosome, so unique windows falls toward the number of
distinct whole-region haplotypes (≤ n_samples) while variants/window climbs.

**Script** — save as `presentation/count_windows.py`:

```python
"""Sweep window length and record unique-window / token / variant counts."""
import csv, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.data.coords import FASTA_PATH, DEFAULT_VCF_PATH
from crop_embed.partitioner import SNPWindowPartitioner
from crop_embed.dataset import UniqueWindowDataset

VCF, FASTA = DEFAULT_VCF_PATH, str(FASTA_PATH)
MAX_LEN = 2048                                   # tokenizer cap (matches embed runs)
TOK_SAMPLE = 400                                 # windows to tokenize per length (speed)
LENGTHS = [2, 10, 100, 1000, 10_000, 100_000, 1_000_000]   # add more for a smoother curve

snps_by_chrom, samples = load_snps_from_vcf(VCF)
tok = AutoTokenizer.from_pretrained("HuggingFaceBio/Carbon-500M",
                                    trust_remote_code=True)

rows = []
for L in LENGTHS:
    hw = max(1, L // 2)
    part = SNPWindowPartitioner(snps_by_chrom, half_window=hw, buffer=0)
    ds = UniqueWindowDataset(VCF, FASTA, part)
    n_windows = len(part)
    n_unique = len(ds)                           # distinct fingerprints
    # mean tokens per window over a sample of unique windows ("<dna>" + 6-mer)
    idxs = range(0, n_unique, max(1, n_unique // TOK_SAMPLE))
    tok_counts = [
        len(tok(f"<dna>{ds[i]['sequence']}", add_special_tokens=False,
                truncation=True, max_length=MAX_LEN)["input_ids"])
        for i in idxs
    ]
    mean_tokens = sum(tok_counts) / len(tok_counts)
    rows.append({
        "length": L, "n_windows": n_windows, "n_unique_windows": n_unique,
        "mean_tokens": mean_tokens, "variants_per_window": n_unique / n_windows,
    })
    print(rows[-1])

out = Path(__file__).parent / "window_sweep.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out)
```

```bash
conda activate svar
python presentation/count_windows.py
```

> Note: large `L` (1e5–1e6) builds a `UniqueWindowDataset` that loads each
> chromosome into memory and materializes long sequences — run on the GPU box /
> a machine with enough RAM, and expect the 1e6 point to take the longest. If
> curve 2 is too slow at large `L`, drop `TOK_SAMPLE` or skip tokenizing the
> top length and note the saturation at `MAX_LEN` analytically (Carbon tokens ≈
> `1 + ceil(L/6)`, capped at `--max-length`).

**Plot:**

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("presentation/window_sweep.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, col, title, ylab in [
    (axes[0], "n_unique_windows", "Unique windows in dataset", "# unique windows"),
    (axes[1], "mean_tokens", "Mean tokens per window", "mean tokens (cap 2048)"),
    (axes[2], "variants_per_window", "Unique variants per window", "unique fps / window"),
]:
    ax.plot(df["length"], df[col], "o-")
    ax.set_xscale("log"); ax.set_xlabel("window length (bp)")
    ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("presentation/figs/01_window_sweep.png", dpi=150)
```

---

## 2. Effect of each training diff on MSE / PCC curves (train + val)

**What to show.** For each head-training "diff" (the sweep axes), the train and
val **MSE** and **PCC** learning curves, plus per-trait versions for chosen
traits.

**The "diffs" are exactly the sweep axes** in
[train_pipeline/run_head_sweep.sh](../train_pipeline/run_head_sweep.sh):
- **variant**: `absolute` / `centered` (`--center-windows`) / `refdelta`
  (`--subtract-reference`) — see [train_head.py](../train_pipeline/train_head.py).
- **head**: `linear` vs `mlp`.
- **pool**: `mean` vs `sum`.
- **warm-start standardizer**: `ws0` vs `ws1` (`--warm-start-standardizer`).
- plus the **embedding cache** itself (variant-cache vs full-forward, window
  size, mean vs snp-only pooling) — one cache per curve family.

**Where the data already is.** Every `train_head.py` run writes a JSONL metrics
sidecar next to its `model.pt` (via
[crop_embed/logging_utils.py](../crop_embed/logging_utils.py)):
`trained_heads/head_sweep/<variant>/<run>/model.metrics.jsonl`. There are
already 14+ such files on disk. Each **train** row has `step`, `train/mean_mse`,
`train/mean_pcc`, and `train/mse/<trait>` + `train/pcc/<trait>`; each **val** row
(logged once per epoch) has the `val/...` equivalents.

**To (re)generate the full grid** (resumable — existing runs are skipped):

```bash
conda activate svar
# Preview the exact commands without running:
DRYRUN=1 bash train_pipeline/run_head_sweep.sh
# Run it (edit the CACHES/VARIANTS/HEADS/POOL arrays in the script to scope it):
bash train_pipeline/run_head_sweep.sh
```

To produce a single curve on demand (e.g. refdelta vs absolute on one cache):

```bash
python train_pipeline/train_head.py \
  --cache checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt \
  --half-window 500 --head mlp --pool mean --warm-start-standardizer \
  --epochs 50 --lr 1e-3 --subtract-reference \
  --output trained_heads/demo/refdelta_mlp/model.pt
```

**Plot helper** — reads any set of metrics JSONLs and overlays them:

```python
import json, glob, os
import matplotlib.pyplot as plt

def load_metrics(path):
    train, val = [], []
    for line in open(path):
        r = json.loads(line)
        (val if any(k.startswith("val/") for k in r) else train).append(r)
    return train, val

def series(rows, key):
    xs = [r["step"] for r in rows if key in r]
    ys = [r[key]   for r in rows if key in r]
    return xs, ys

# Pick the runs (diffs) to compare — one label -> one metrics file.
runs = {
    "absolute": "trained_heads/head_sweep/absolute/sativas413_carbon_vc_w250__absolute_mlp_mean_ws1_lr1e-3_wd1e-4_ep15/model.metrics.jsonl",
    "centered": "trained_heads/head_sweep/centered/sativas413_carbon_vc_w250__centered_mlp_mean_ws1_lr1e-3_wd1e-4_ep15/model.metrics.jsonl",
    "refdelta": "trained_heads/head_sweep/refdelta/sativas413_carbon_vc_w250__refdelta_mlp_mean_ws1_lr1e-3_wd1e-4_ep15/model.metrics.jsonl",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
panels = [("train/mean_mse", axes[0,0], "train MSE"),
          ("val/mean_mse",   axes[0,1], "val MSE"),
          ("train/mean_pcc", axes[1,0], "train PCC"),
          ("val/mean_pcc",   axes[1,1], "val PCC")]
for label, path in runs.items():
    tr, va = load_metrics(path)
    for key, ax, title in panels:
        rows = va if key.startswith("val/") else tr
        ax.plot(*series(rows, key), label=label)
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=.3)
axes[0,0].legend()
fig.suptitle("Effect of window-centering diff (mlp / mean / ws1, vc_w250)")
fig.tight_layout(); fig.savefig("presentation/figs/02_diff_curves.png", dpi=150)
```

**Per-trait version.** Swap the metric keys to the per-trait ones, e.g.
`train/pcc/Seed length` and `val/pcc/Seed length`. The full trait list is the
column set in `RiceDiversity_44K_Phenotypes_34traits_PLINK.txt` (also visible as
the `*/mse/<trait>` keys in any metrics file). Loop over a chosen subset:

```python
TRAITS = ["Seed length", "Flowering time at Arkansas", "Amylose content"]
fig, axes = plt.subplots(1, len(TRAITS), figsize=(5*len(TRAITS), 4), squeeze=False)
for j, trait in enumerate(TRAITS):
    ax = axes[0][j]
    for label, path in runs.items():
        _, va = load_metrics(path)
        ax.plot(*series(va, f"val/pcc/{trait}"), label=label)
    ax.set_title(f"val PCC — {trait}"); ax.set_xlabel("step"); ax.grid(alpha=.3)
axes[0][0].legend()
fig.tight_layout(); fig.savefig("presentation/figs/02_per_trait_pcc.png", dpi=150)
```

> Tip: to compare a *single* axis cleanly (e.g. only `pool=mean` vs `sum`), hold
> every other token in the run-dir name fixed and point `runs` at the two files
> that differ in just that token. The `run` naming in `run_head_sweep.sh`
> (`<cache>__<variant>_<head>_<pool>_<ws>_lr..._wd..._ep...`) makes this a glob.

---

## 3. UMAP / t-SNE of embedding preprocessing options (+ Carbon vs DNABERT-2)

**What to show.** 2-D UMAP (and/or t-SNE) of the **per-sample** embedding under
each preprocessing combination, points colored by inferred subpopulation:
- **all windows** vs **snp-only** windows (`--snp-only` at embed time).
- **Carbon** vs **DNABERT-2** backend.
- **centered + normed → average** vs **summed and standardized** (the two
  per-sample aggregation recipes).

**Mapping each option to code.** A cache (`embed_windows.py` /
`embed_windows_vc.py` output) gives the per-window table; the per-sample vector
is produced by pooling exactly as [train_head.py](../train_pipeline/train_head.py)
does:
- **summed and standardized** = `F.embedding_bag(sample_fp_index, cache,
  mode="sum")` then z-score the columns (the head's learned standardizer warm-
  start is just column mean/std — replicate with `StandardScaler`).
- **centered + normed → average** = the `--center-ln-pool` diagnostic in
  train_head.py: per-window center (subtract train-set window mean), parameter-
  free LayerNorm, then **mean** over windows.
- **all vs snp-only** and **Carbon vs DNABERT-2** are just different cache
  files. Available caches include
  `checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt` (all) and
  `..._hw500_snponly.ckpt.pt` (snp-only); generate DNABERT-2 ones with
  `embed_windows.py --backend dnabert2` (default).

**First, make sure the caches exist** (see the commented commands in
[embed_all_windows.sh](../embed_all_windows.sh)). Minimal set for this figure:

```bash
conda activate svar
VCF=/home/andrew.dickson/rice_data/sativas413_msu7_final.vcf
# Carbon, all vs snp-only (full-forward)
python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" \
  --half-window 500 --max-length 2048 --batch-size 64 \
  --output checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt
python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" \
  --half-window 500 --max-length 2048 --batch-size 64 --snp-only \
  --output checkpoints/manual/sativas413_carbon500m_hw500_snponly.ckpt.pt
# DNABERT-2, all vs snp-only
python train_pipeline/embed_windows.py --backend dnabert2 --vcf-path "$VCF" \
  --half-window 500 --max-length 2048 --batch-size 64 \
  --output checkpoints/manual/sativas413_dnabert2_hw500.ckpt.pt
python train_pipeline/embed_windows.py --backend dnabert2 --vcf-path "$VCF" \
  --half-window 500 --max-length 2048 --batch-size 64 --snp-only \
  --output checkpoints/manual/sativas413_dnabert2_hw500_snponly.ckpt.pt
```

**Pooling helper** — turns a cache into a `(n_samples, D)` matrix under either
recipe. Save as `presentation/pool_cache.py`:

```python
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_embed.data.loading import prepare_data
from crop_embed import FixedWindowEmbedder

SPLIT = "splits/sativas413_seed42.pt"

def per_sample_matrix(cache_path, recipe, half_window=500):
    data = prepare_data(split_path=SPLIT, half_window=half_window, buffer=0,
                        verbose=False)
    ds, train_idx = data["dataset"], data["train_idx"]
    emb = FixedWindowEmbedder.from_file(cache_path, ds)
    cache = emb.cache.float()                       # (n_fps, D)
    sfi = emb.sample_fp_index                       # (n_samples, n_windows)
    if recipe == "sum_std":
        X = F.embedding_bag(sfi, cache, mode="sum") # (n_samples, D)
        mu, sd = X[train_idx].mean(0), X[train_idx].std(0).clamp_min(1e-6)
        return ((X - mu) / sd).numpy(), ds.samples
    elif recipe == "center_ln_mean":
        # per-window centering on TRAIN samples + param-free LN + mean pool
        wmean = torch.zeros(sfi.shape[1], cache.shape[1])
        g_all = cache[sfi]                          # (n_samples, n_windows, D)
        wmean = g_all[train_idx].mean(0)            # (n_windows, D)
        g = g_all - wmean
        g = F.layer_norm(g, (g.shape[-1],))
        return g.mean(1).numpy(), ds.samples        # (n_samples, D)
    raise ValueError(recipe)
```

> The `center_ln_mean` branch materializes `(n_samples, n_windows, D)`; for the
> 413×~26k×D case do it in sample-chunks (mirror the loop in
> `train_head.py:--center-ln-pool`) if memory is tight.

**Subpopulation labels.** Follow
[notebooks/snp_emb_subpop_analysis.ipynb](../notebooks/snp_emb_subpop_analysis.ipynb):
TruncatedSVD on the raw SNP matrix → quartile bins on PC1 as inferred
subpopulations. (If you have the RiceDiversity 44K true subpopulation
annotations, color by those instead.)

**Plot — grid of UMAPs:**

```python
import umap, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import sys; sys.path.insert(0, ".")
from presentation.pool_cache import per_sample_matrix
from crop_embed.data.preprocessing import load_vcf_sparse

# subpop labels from SNP PC1 quartiles, aligned to dataset.samples order
X_snp, vcf_samples, _, _ = load_vcf_sparse()
pc1 = TruncatedSVD(n_components=10, random_state=42).fit_transform(X_snp)[:, 0]
samp_to_pop = dict(zip(vcf_samples, pd.qcut(pc1, 4, labels=False)))

panels = [
    ("Carbon all / sum+std",      "checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt",          "sum_std"),
    ("Carbon all / center+ln",    "checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt",          "center_ln_mean"),
    ("Carbon snp-only / sum+std", "checkpoints/manual/sativas413_carbon500m_hw500_snponly.ckpt.pt",   "sum_std"),
    ("DNABERT2 all / sum+std",    "checkpoints/manual/sativas413_dnabert2_hw500.ckpt.pt",             "sum_std"),
]
fig, axes = plt.subplots(1, len(panels), figsize=(5*len(panels), 4.5))
for ax, (title, cache, recipe) in zip(axes, panels):
    X, samples = per_sample_matrix(cache, recipe)
    coords = umap.UMAP(n_components=2, random_state=42,
                       n_neighbors=15, min_dist=0.05).fit_transform(X)
    pops = np.array([samp_to_pop[s] for s in samples])
    sc = ax.scatter(coords[:,0], coords[:,1], c=pops, cmap="tab10",
                    s=15, alpha=.85, linewidths=0)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("UMAP of per-sample embeddings across preprocessing options")
fig.tight_layout(); fig.savefig("presentation/figs/03_umap_grid.png", dpi=150)
```

For **t-SNE**, swap the reducer: `from sklearn.manifold import TSNE;
coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)`.

---

## 4. SNP-embedding degradation under the variant-cache approximation (1–10 SNPs)

**What to show.** How close the variant-cache embedding stays to the true
(full-forward) embedding as the number of concurrent SNPs in a window grows from
1 to ~10 — and how that compares to the reference-only baseline.

**Where it comes from — this already exists.**
[model_dev/compare_variant_cache_embeddings.py](../model_dev/compare_variant_cache_embeddings.py)
computes, per variant window and **binned by #SNP tokens**:
- `var_cos` — cosine(cache, true) at the recomputed SNP token positions.
- `cache_pool_cos` / `ref_pool_cos` — cosine of the mean-pooled window vector
  for the cache vs the reference-only baseline, both vs the true alt forward.
- `cache_relerr` / `ref_relerr` — relative L2 error of the pooled vectors.

It currently **prints a table**. To plot, add a CSV dump. Apply this small patch
to `_report(records)` (it already has the `records` list of tuples
`(n_snp, var_cos, cache_pool_cos, ref_pool_cos, cache_relerr, ref_relerr)`):

```python
# add near the top of _report(), after `by_n` is built:
import csv
with open("presentation/vc_degradation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_snp","var_cos","cache_pool_cos","ref_pool_cos",
                "cache_relerr","ref_relerr"])
    w.writerows(records)
```

(Or add a `--csv-out PATH` arg and write there.) Then run it — `--min-snps 1`
keeps all windows, and a higher `--limit`/`--max-eval` gives more windows per
bin so the per-#SNP means are stable:

```bash
conda activate svar
python -m model_dev.compare_variant_cache_embeddings \
  --half-window 500 --buffer 0 --max-length 2048 \
  --limit 4000 --min-snps 1
# -> prints the table AND writes presentation/vc_degradation.csv
```

> Needs CUDA realistically (two Carbon-500M forwards per window). The exact and
> cache models are loaded via `load_carbon_local` / `load_carbon_variant_cache`
> (fp32). Higher `--limit` = more windows scanned = denser high-#SNP bins.

**Plot — degradation vs #SNPs:**

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("presentation/vc_degradation.csv")
g = df[df["n_snp"].between(1, 10)].groupby("n_snp").mean(numeric_only=True)

fig, (axc, axe) = plt.subplots(1, 2, figsize=(13, 4.5))
# cosine similarity to true (higher = better)
axc.plot(g.index, g["var_cos"],        "o-", label="cache, per-SNP-token")
axc.plot(g.index, g["cache_pool_cos"], "s-", label="cache, window-pooled")
axc.plot(g.index, g["ref_pool_cos"],   "^-", label="reference-only (baseline)")
axc.set_xlabel("# SNPs in window"); axc.set_ylabel("cosine vs true alt forward")
axc.set_title("Variant-cache fidelity vs #SNPs"); axc.legend(); axc.grid(alpha=.3)
# relative L2 error (lower = better)
axe.plot(g.index, g["cache_relerr"], "s-", label="cache pooled")
axe.plot(g.index, g["ref_relerr"],   "^-", label="reference-only")
axe.set_xlabel("# SNPs in window"); axe.set_ylabel("relative L2 error (pooled)")
axe.set_title("Pooled-embedding error vs #SNPs"); axe.legend(); axe.grid(alpha=.3)
fig.tight_layout(); fig.savefig("presentation/figs/04_vc_degradation.png", dpi=150)
```

> Interpretation (from the script's own note): 1 SNP is near-exact at the
> per-token level (`var_cos ≈ 1`); error accumulates with concurrent SNPs. The
> *pooled* cache vs reference gap can look small because mean-pooling dilutes the
> few SNP tokens — so present `var_cos` (per-SNP-token) as the headline fidelity
> curve, with the pooled curves as context.

---

## 5. Carbon AR fine-tuning curves on the 90% SNP-window subset (variant-cache)

**What to show.** Train + val **log-likelihood on SNP tokens** vs. training
step/epoch, for Carbon fine-tuned through the variant cache on the AR objective.
"90% subset" = the train split with `--val-frac 0.1`; "variant cache method" =
`--backend cache`. Log-likelihood = `−NLL` (the loss in
[variant_ar/loss.py](../variant_ar/loss.py) is the token-weighted next-token NLL
at SNP positions; perplexity `= exp(NLL)`).

**How to run** ([variant_ar/train.py](../variant_ar/train.py)):

```bash
conda activate svar
python variant_ar/train.py \
  --backend cache --half-window 500 --buffer 0 --max-length 2048 \
  --val-frac 0.1 --epochs 5 --batch-size 4 --lr 1e-5 --precision fp32 \
  --output checkpoints/variant_ar/carbon_vc.pt
```

This prints `train_loss` every `--log-every` steps and a per-epoch
`[val] nll=... ppl=...`, and saves the best checkpoint.

**One gap to close first:** unlike `train_head.py`, `variant_ar/train.py` does
**not** yet write a JSONL sidecar — it only prints. Two options:

**(a) Add `MetricLogger`** (recommended; matches the rest of the repo). Minimal
edits in `variant_ar/train.py`:

```python
# imports
from crop_embed import MetricLogger, metrics_path_for
# after args are parsed:
logger = MetricLogger(metrics_path_for(args.output), config=vars(args))
# in the train loop, where it prints train_loss:
if step % args.log_every == 0:
    logger.log({"epoch": epoch, "step": step, "train/nll": loss.item(),
                "train/ll": -loss.item()})
# after each val eval:
logger.log({"epoch": epoch, "step": step, "val/nll": val["mean_nll"],
            "val/ll": -val["mean_nll"], "val/perplexity": val["perplexity"]})
# at the very end:
logger.close()
```

Then plot from `checkpoints/variant_ar/carbon_vc.metrics.jsonl` with the
`load_metrics` helper from §2:

```python
import matplotlib.pyplot as plt
# reuse load_metrics/series from section 2
tr, va = load_metrics("checkpoints/variant_ar/carbon_vc.metrics.jsonl")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(*series(tr, "train/ll"), label="train LL (SNP tokens)", alpha=.8)
ax.plot(*series(va, "val/ll"),   "o-", label="val LL (SNP tokens)")
ax.set_xlabel("step"); ax.set_ylabel("log-likelihood (−NLL)")
ax.set_title("Carbon AR fine-tune via variant cache (90% train)")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("presentation/figs/05_ar_curves.png", dpi=150)
```

**(b) Parse stdout** instead (no code change): tee the run to a log and regex the
`step .. train_loss=` and `[val] nll=` lines:

```bash
python variant_ar/train.py --backend cache --val-frac 0.1 ... \
  --output checkpoints/variant_ar/carbon_vc.pt 2>&1 | tee logs/variant_ar_run.log
```

```python
import re, matplotlib.pyplot as plt
tr_s, tr_ll, va_e, va_ll = [], [], [], []
for line in open("logs/variant_ar_run.log"):
    m = re.search(r"step (\d+)\s+train_loss=([\d.]+)", line)
    if m: tr_s.append(int(m[1])); tr_ll.append(-float(m[2]))
    m = re.search(r"epoch (\d+)\s+\[val\] nll=([\d.]+)", line)
    if m: va_e.append(int(m[1])); va_ll.append(-float(m[2]))
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(tr_s, tr_ll, label="train LL", alpha=.7)
ax.plot(va_e, va_ll, "o-", label="val LL (per epoch)")  # x is epoch here
ax.set_xlabel("step / epoch"); ax.set_ylabel("log-likelihood (−NLL)")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig("presentation/figs/05_ar_curves.png", dpi=150)
```

> Optional baseline: re-run with `--backend exact` to overlay the ground-truth
> objective the cache is approximating — they share the identical loss so the
> curves are directly comparable (see the train.py docstring).

---

## Quick checklist

| # | Result | Primary code | Output fig |
|---|--------|--------------|------------|
| 1 | Unique windows / tokens / variants vs length | `partitioner.py` + `dataset.py` (new `count_windows.py`) | `figs/01_window_sweep.png` |
| 2 | Training-diff MSE/PCC curves (train+val, per-trait) | `train_head.py` + `run_head_sweep.sh` metrics JSONL | `figs/02_diff_curves.png`, `figs/02_per_trait_pcc.png` |
| 3 | UMAP/t-SNE across preprocessing + Carbon vs DNABERT-2 | `embed_windows*.py` caches + pooling recipes | `figs/03_umap_grid.png` |
| 4 | Variant-cache degradation vs #SNPs | `model_dev/compare_variant_cache_embeddings.py` (+CSV) | `figs/04_vc_degradation.png` |
| 5 | Carbon AR fine-tune curves (variant cache, 90%) | `variant_ar/train.py` (+MetricLogger) | `figs/05_ar_curves.png` |
</content>
</invoke>

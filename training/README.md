# `training/` — how to run jobs

---

## 0. Environment

Everything runs in the `svar` conda env, and all data/caches/runs live under
`$SVAR_SCRATCH` (defaults to `~/svar_scratch` on this box — the cluster path is
dead here, so just leave it unset or export the local one):

```bash
conda activate svar
export SVAR_SCRATCH=$HOME/svar_scratch      # optional; this is already the default
cd ~/svar
```

Run everything as a module from the repo root (`python -m training.<dir>.run`),
never as a file path — the package imports depend on it.

Scratch layout:

```
$SVAR_SCRATCH/
├── datasets/<crop>/      # pgen/pvar/psam, VCF, FASTA, <crop>_pheno_aligned.csv
├── caches/<crop>/        # window-embedding caches: <backbone>_hw<HW>.ckpt.pt
└── runs/index.jsonl      # append-only manifest of every run (the compare surface)
```

Splits are the exception — they're tiny and **committed in-repo** at
`splits/<crop>_seed42.pt`, built automatically on first use.

---

- **Dataset** — one registry entry (`soy`, `rice`, `arabidopsis`, `wheat`).
  Selected with `--dataset`.
- **Modality** — how features reach the model: `snp` (0/1/2 dosage), `emb`
  (per-sample pooled embeddings), or `window` (per-window embedding cache the NN
  head pools internally). You don't pass this — each runner *is* a modality.
- **Split** — one 70/15/15 nested split per `(dataset, seed)`, keyed by sample
  ID, identical across every modality 
- **Run record** — every run appends one row to `$SVAR_SCRATCH/runs/index.jsonl`
  with its final hyperparameters + val/test metrics. Comparison is one read.

Two layers of hyperparameter search: **within-run** (each sklearn model is a
`GridSearchCV`, tunes itself per trait) and **across-run** (`sweep.py` drives a
grid over pipeline axes). You rarely touch the first.

---

## 2. Runners — one job at a time

Each runner loops all requested traits internally and writes a run record. All
share `--dataset`, `--traits` (`all` or comma list), `--seed` (default 42).

### `snp_sklearn` — classical models on the SNP matrix
Models: `ridge` (RR-BLUP), `svr` (RBF), `krr` (RBF kernel-ridge), `rf`, `gbm`, `pls`.

```bash
# dense dosage (good for ridge/pls); one trait
python -m training.snp_sklearn.run --dataset soy --model ridge --traits protein

# all 11 traits
python -m training.snp_sklearn.run --dataset soy --model pls --traits all

# kernels/trees on 40k SNPs are intractable dense → sparse then TruncatedSVD-500
python -m training.snp_sklearn.run --dataset soy --model svr --sparse --svd 500
```

### `emb_sklearn` — the same classical suite on pooled embeddings
Needs an embedding cache (see §3). The pooled vector is low-dim + dense, so no
`--sparse/--svd` — every model runs straight on it.

```bash
python -m training.emb_sklearn.run --dataset soy --model ridge \
    --backbone carbon500m --half-window 500 \
    --cache $SVAR_SCRATCH/caches/soy/carbon500m_hw500.ckpt.pt \
    --recipe center_ln_mean          # or sum_std
```

### `emb_nn` — trained neural heads on the window cache
Heads: `linear` / `mlp`, with pooling + standardizer knobs. This is the one to
tune (`--warm-start-standardizer` is essentially mandatory — without it the head
doesn't train).

```bash
python -m training.emb_nn.run --dataset soy \
    --cache $SVAR_SCRATCH/caches/soy/carbon500m_hw500.ckpt.pt --half-window 500 \
    --head mlp --pool mean --warm-start-standardizer \
    --lr 1e-3 --weight-decay 1e-4 --epochs 60 \
    --output trained_heads/soy_carbon500m/mlp_mean/model.pt
```

Key flags: `--head {linear,mlp}`, `--pool {sum,mean}`, `--center-windows`,
`--warm-start-standardizer`, `--n-layers`, `--hidden-dim`, `--dropout`,
`--lr`, `--weight-decay`, `--epochs`. Reads the window cache directly (VCF-skip
fast path), so setup is seconds, not minutes.

### `e2e` — encoder + head fine-tuning
End-to-end (two-pass activation checkpointing). Same head knobs plus backbone
fine-tuning; heavier, GPU-bound.

### `cropformer` — CNN + self-attention SNP baseline
Single architecture on the `snp` matrix, comparable to `snp_sklearn`. Model name
defaults to `cropformer`.

```bash
python -m training.cropformer.run --dataset soy --traits all --mic-k 10000 --max-epochs 100
python -m training.cropformer.run --dataset rice --mic-k 0     # use all SNPs
```

---

## 3. Building an embedding cache (prerequisite for `emb_*` / `e2e`)

`emb_sklearn`, `emb_nn`, and `e2e` all consume a window-embedding cache. Generate
one per `(dataset, backbone, half_window)` with `embed_windows.py`. It reads the
VCF + FASTA, windows every SNP, dedupes windows, and embeds the unique set with a
DNA-LM. **This is the slow, one-time step** (VCF parse + windowing is ~40 min on
soy before the GPU even starts; then the embedding loop).

```bash
# Carbon-500M, soy, half-window 500  (what all the soy emb jobs use)
CUDA_VISIBLE_DEVICES=0 python train_pipeline/embed_windows.py \
    --backend carbon --carbon-size 500M \
    --half-window 500 --buffer 0 --max-length 2048 --batch-size 16 \
    --vcf-path   $SVAR_SCRATCH/datasets/soy/soysnp50k_a2_final.vcf \
    --fasta-path $SVAR_SCRATCH/datasets/soy/Glycine_max.Glycine_max_v2.1.dna_sm.toplevel.fa \
    --output     $SVAR_SCRATCH/caches/soy/carbon500m_hw500.ckpt.pt

# Carbon-3B: same recipe, --carbon-size 3B, smaller batch (bigger model),
# and a resume checkpoint for the longer run:
CUDA_VISIBLE_DEVICES=0 python train_pipeline/embed_windows.py \
    --backend carbon --carbon-size 3B \
    --half-window 500 --buffer 0 --max-length 2048 --batch-size 8 \
    --vcf-path   $SVAR_SCRATCH/datasets/soy/soysnp50k_a2_final.vcf \
    --fasta-path $SVAR_SCRATCH/datasets/soy/Glycine_max.Glycine_max_v2.1.dna_sm.toplevel.fa \
    --checkpoint-every 2000 \
    --checkpoint-path $SVAR_SCRATCH/caches/soy/carbon3b_hw500.partial.pt \
    --output          $SVAR_SCRATCH/caches/soy/carbon3b_hw500.ckpt.pt
```

Backends: `carbon` (`--carbon-size 500M|3B|8B`), `dnabert2`, `plantcad`. The
output `emb_dim` (1024 for 500M, 3072 for 3B) is stored in the cache metadata, so
downstream runners read it — you never hardcode it.

---

## 4. Sweeps — run many points, resumable, comparable

`sweep.py` expands a Cartesian grid declared in a config module
(`training/sweeps/<name>.py` exporting `SWEEP`) and shells out to the right runner
per point. Always `--dry-run` first.

```bash
python -m training.sweep --config training/sweeps/soy_baselines.py --dry-run   # preview
python -m training.sweep --config training/sweeps/soy_baselines.py             # run
python -m training.sweep --config training/sweeps/soy_carbon3b_heads.py --gpus 0
python -m training.sweep --config training/sweeps/<cfg>.py --only svr          # label substring filter
python -m training.sweep --config training/sweeps/<cfg>.py --jobs 3            # worker pool
```

Flags: `--dry-run` / `--list` (preview), `--gpus 0,1,2` (round-robin GPU-pinned
blocks marked `gpu: True`), `--jobs N` (concurrent points), `--only SUBSTR`
(filter labels), `--force` (ignore the resume ledger and re-run everything).

**Resumable.** Each point is keyed by a hash of its resolved command in a per-sweep
`ledger.jsonl`; a re-run skips completed points. NN blocks with an
`output_template` also skip if the model file already exists. So you can Ctrl-C and
re-launch freely. Per-point logs land in `logs/sweeps/<cfg>/`.

### Writing a config
A block is a dict; `SWEEP` is one block or a list. Minimal:

```python
_COMMON = {"dataset": "soy", "traits": "all", "seed": 42}
block = {
    "runner": "snp_sklearn",              # snp_sklearn | emb_sklearn | emb_nn | e2e
    "name":   "dense",                    # label prefix
    "fixed":  _COMMON,                    # constants for every point
    "grid":   {"model": ["ridge", "pls"]},# swept axes (Cartesian product)
    # "gpu": True,                        # pin a --gpus id (NN runners)
    # "output_template": "trained_heads/.../{label}/model.pt",  # NN output + skip key
}
SWEEP = [block]
```
Booleans in `grid`/`fixed` become bare flags (`True` → `--warm-start-standardizer`,
`False` → omitted). See [`sweeps/example.py`](sweeps/example.py) for the annotated
template.

### Configs that exist now
| config | what it sweeps |
|---|---|
| `sweeps/soy_baselines.py` | 6 sklearn models on soy **SNPs** (dense ridge/pls; sparse→SVD500 krr/svr/rf/gbm) |
| `sweeps/soy_carbon500m_baselines.py` | cheap sklearn models (ridge/pls/svr/krr) on soy **Carbon-500M embeddings** |
| `sweeps/soy_carbon_heads.py` | minimal 4-point NN head (head×pool) on 500M cache |
| `sweeps/soy_carbon_heads_search.py` | 40-point NN head search (training hparams + arch) on 500M cache |
| `sweeps/soy_carbon3b_heads.py` | lean 10-point NN head search on the **Carbon-3B** cache |
| `sweeps/arabidopsis_baselines.py` | 6 sklearn models on arabidopsis (sparse→SVD500), top-20 traits |

---

## 5. Comparing results

Every run — single or swept — appends to `$SVAR_SCRATCH/runs/index.jsonl`. One read:

```python
from training.common.run_record import load_runs
df = load_runs()          # -> DataFrame, one row per run; metrics flattened to columns
# columns include: dataset, features, model, backbone, half_window,
#                  val.pearson, val.r2, val.mse, val.mae, test.pearson, ...
#                  (the *.pearson columns are already the trait-mean)
soy = df[df.dataset == "soy"]
print(soy.groupby(["features", "model"])["test.pearson"].max().sort_values(ascending=False))
```

NN head runs currently write their `run.json` under `trained_heads/<label>/`
rather than `runs/` — a known location inconsistency; read those directly if you
need them alongside `load_runs()`.

---

## 6. Running in the background + watching

Long jobs: launch with `nohup ... &`, log to a file, and tail it. Note Python
block-buffers stdout to a file, so a quiet log ≠ a stalled job — check the process
and (for embed jobs) GPU with `nvidia-smi`.

```bash
nohup python -m training.sweep --config training/sweeps/soy_carbon500m_baselines.py \
    > /tmp/emb_baselines.log 2>&1 &
tail -f /tmp/emb_baselines.log
nvidia-smi     # for GPU (embed / e2e / NN-head) jobs
```

Pick a **free** GPU with `CUDA_VISIBLE_DEVICES=N` (or `--gpus N` for sweeps) —
check occupancy with `nvidia-smi` first.

---

## 7. Dataset readiness

| dataset | genotypes | phenotypes | embedding cache | notes |
|---|---|---|---|---|
| **soy** | ✅ pgen | ✅ `soy_pheno_aligned.csv` (11 traits) | ✅ carbon500m_hw500 (3B building) | the reference dataset; everything is verified here |
| arabidopsis | ✅ pgen | ✅ `arabidopsis_pheno_aligned.csv` (replicate-collapsed, top-20 best-covered) | — | baseline sweep configured |
| rice | VCF | pheno not yet IID-aligned | — | phenotype join is a follow-up |
| wheat | build via `datasets/wheat` | 12 traits registered | — | ready for dataset build |

To (re)build a dataset's raw files, use its Makefile under
[`../datasets/`](../datasets/) (`make -C datasets/soy`). See
[`../datasets/README.md`](../datasets/README.md).

---

## Quick reference

```bash
# single SNP baseline
python -m training.snp_sklearn.run --dataset soy --model ridge --traits all
# single embedding NN head
python -m training.emb_nn.run --dataset soy --cache <cache> --half-window 500 \
    --head mlp --pool mean --warm-start-standardizer --output <out>/model.pt
# a whole sweep (preview then run)
python -m training.sweep --config training/sweeps/soy_baselines.py --dry-run
python -m training.sweep --config training/sweeps/soy_baselines.py
# build an embedding cache
python train_pipeline/embed_windows.py --backend carbon --carbon-size 500M \
    --half-window 500 --vcf-path <vcf> --fasta-path <fasta> --output <cache>
# compare
python -c "from training.common.run_record import load_runs; print(load_runs().shape)"
```

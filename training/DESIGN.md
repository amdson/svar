# `training/` — unified model pipeline

Standardized way to train **any model type on any dataset**, over data made **as
uniform as possible**, with per-run hyperparameters recorded for downstream
comparison. Built as an *additive* layer: `crop_embed/` stays the library,
`train_pipeline/` keeps its identity (embedding-cache generation + NN/e2e
training + sweeps) and is only *un-hardcoded* to consume this layer — nothing is
moved or reorganized.

## The two data shapes, one contract

Some models need a **SNP matrix** (samples × variants dosage); others need
**reconstructed sequence windows** → **fixed embeddings**. We express the
difference as a *feature modality* behind one contract, so every runner sees the
same shape: an `(n_samples, features)` matrix (or a window bundle) + aligned
targets + a split.

| modality | provider | consumed by |
|---|---|---|
| `snp`    | additive 0/1/2 dosage from pgen        | `snp_sklearn` |
| `emb`    | per-sample pooled fixed embeddings      | `emb_sklearn`, `emb_nn` (pooled heads) |
| `window` | the per-window cache + `sample_fp_index` | `emb_nn` (heads that pool internally), `e2e` |

## Layout

```
crop_embed/                 # library (+ data/genotype_matrix.py, generalized targets)
training/
├── common/                 # shared lib — all heavy lifting
│   ├── datasets.py         # DatasetSpec + DATASETS registry
│   ├── splits.py           # 3-way nested split build/load
│   ├── features.py         # snp / emb / window / targets providers
│   ├── metrics.py          # NaN-masked per-trait eval
│   ├── run_record.py       # lightweight per-run record + manifest
│   ├── artifacts.py        # run dirs + model/config IO
│   └── harness.py          # run(runner): parse → load → fit → eval → save
├── snp_sklearn/            # modality: snp     (run.py + estimators.py)
├── emb_sklearn/            # modality: emb     (run.py, shares estimators)
├── emb_nn/                 # was train_pipeline/train_head.py (moved here, on common)
└── e2e/                    # train_pipeline/train_end2end.py (pending move)
```

The GLM trainers move into `emb_nn`/`e2e` (files relocated, external refs updated),
keeping only the data layer swapped onto `training/common`; all head/backbone
machinery is unchanged.

Per-directory scripts own their CLI and hand a small `Runner` to
`harness.run(...)`, which does everything shared. Adding a model = a new file in
a directory; adding a dataset = one registry entry; adding a modality = one
provider.

## Canonical dataset form

Every crop is normalized to the same contract so the code has no per-crop
branches:

- **genotypes**: plink2 `.pgen/.pvar/.psam`, integer chromosomes, **IID sample keys**;
- **phenotypes**: a CSV keyed by those same IIDs (soy's `soy_pheno_aligned.csv` is the template);
- **splits/caches**: addressed by dataset name.

`DatasetSpec` (frozen dataclass) resolves paths under `$DATA_ROOT/<crop>`
(default `$SVAR_SCRATCH/datasets`, same convention as the Makefiles):

```python
DatasetSpec(name, pgen_prefix, vcf_path, fasta_path, pheno_csv, trait_cols=None)
DATASETS: dict[str, DatasetSpec]     # get_dataset("soy")
```

## Splits — 70/15/15 nested, precomputed

Hold out **15% test** by sample-ID first, then split the remaining 85% into
**70/15 train/val** (overall 70/15/15). Random, seeded (**42**), keyed by ID so
partitions survive reordering and are **identical across every modality** (no
cross-model leakage). One file per `(dataset, seed)`, **committed in-repo**
(`splits/<crop>_seed42.pt`, tiny). Built once by `training/common/splits.py`
(the evolved `cache_split.py`).

Stored keys: `sample_ids`, `train_sample_ids`, `val_sample_ids`,
`test_sample_ids`, `targets` (n×T, NaN=missing, aligned to `sample_ids`),
`trait_cols`, `metadata` (seed, ratios, generated_at).

## Feature providers (`features.py`)

All return arrays aligned to a requested sample order.

```python
snp_matrix(spec, samples, *, impute="ref")   -> (X f32 (n,V), variant_ids)   # missing→0 (ref) by default
pooled_embeddings(spec, backbone, half_window, samples, recipe="center_ln_mean") -> X f32 (n,D)
window_cache(spec, backbone, half_window)     -> WindowBundle(cache, sample_fp_index, sample_ids)
targets(spec, samples, traits)                -> (Y f32 (n,T), trait_cols)    # reindex pheno_csv by IID
```

**No canonical backbone/window** — `emb_*` runners take `--backbone` and
`--half-window` as required args. Caches are addressed at
`$SVAR_SCRATCH/caches/<crop>/<backbone>_hw<hw>[_snponly].ckpt.pt` (systematic
version of how `embed_windows.py` already names them), with an explicit
`--cache` override.

## Model runners

- **`snp_sklearn`** / **`emb_sklearn`** — Ridge (RR-BLUP), SVR (RBF), **KRR
  (RBF kernel-ridge)**, RF, GBM, PLS. `GridSearchCV` inside the train partition,
  select on val, report test. **One trait at a time** (loop); estimator
  definitions shared between the two. `snp_sklearn` takes either dense dosage or,
  with `--sparse --svd N`, the **raw sparse SNP matrix → TruncatedSVD → model**
  pipeline (SVD consumes the CSR directly). The SVD/sparse/impute knobs are
  recorded in each run's `run.json` for comparison.
- **`emb_nn`** — the existing `fp_head_model` heads (linear/mlp/attention,
  standardizers, pooling variants), moved from `train_head.py`, **multi-task**.
- **`e2e`** — encoder+head fine-tuning (two-pass activation checkpointing), moved
  from `train_end2end.py`, **multi-task**.

Metrics are per-trait everywhere: Pearson, R², MSE, MAE, NaN-masked, on **val and
test**.

## Run records (`run_record.py`) — lightweight, decoupled

A dataclass + `build/write/load`. No wandb, no search framework, no
orchestration. Records only the config **actually used** (not the search space —
that lives in the saved estimator's `cv_results_` if wanted):

```jsonc
{ "run_id","created_at","git_sha",              // auto (git_sha best-effort)
  "dataset","seed","split","features",
  "backbone","half_window",                     // only when relevant
  "traits","model","hyperparams",               // hyperparams = final params (best_params_)
  "metrics": {"val":..., "test":...} }
```

Defaults keep light experiments frictionless: everything above is auto-filled
from parsed args; `run_id` and the run dir are auto-named (config hash); the
manifest append (`runs/index.jsonl`) is automatic and silent. **Large-artifact
content hashing is off by default** (paths + size/mtime only); `--strict` turns
on cache/split content hashes for publication-grade comparison.

Downstream comparison is one read: `load_runs()` → DataFrame, group/filter by any
field. Runs are written under scratch (`$SVAR_SCRATCH/runs/`, gitignored like
`trained_heads/`); splits are committed.

## Reuse vs net-new

- **Reuse**: `UniqueWindowDataset`, `FixedWindowEmbedder`, `fp_head_model`,
  `pool_cache.per_sample_matrix`, `MetricLogger`, extended `make_split`.
- **Net-new (Phase 1)**: `genotype_matrix.py` dosage extractor, 3-way split,
  registry + IID-keyed targets, `features/metrics/run_record/harness`,
  `snp_sklearn` + `emb_sklearn`.
- **Later**: refactor `train_head.py`/`train_end2end.py` in place onto this
  layer (parity-checked); fold `predict_crop_phenotype.py` + baseline notebooks
  into the sklearn runners; delete the stale `scripts/generate_cache.py`.

## Status

- [x] Phase 1: `common` + `snp_sklearn` + `emb_sklearn`, verified on soy.
- [~] Phase 2: `emb_nn` done (train_head → `training/emb_nn/run.py`, on `common`,
  parity-checked on rice, sweep/repro refs updated); `e2e` (train_end2end) pending.
- [ ] Phase 3: cleanup + notebook/`predict_crop_phenotype.py` migration.

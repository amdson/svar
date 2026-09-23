# training_meth — state as of 2026-09-23

## Question

Do frozen gLM features predict the effect of natural sequence variation on DNA
methylation at a fixed cytosine across arabidopsis accessions, and do they beat
a per-site linear model on the cis SNVs in the same window? The methylation
benchmark (`ath_meth_benchmark/`) holds position fixed and varies the genome:
~456k evaluation sites (371k CG, 42k CHG, 43k CHH) x 811 accessions, with a
measurement ceiling of r ≈ 0.82 (KAPPA = 3). Dense supervision per window, a
mechanistically clean `context_changing` subtask, and a held-out-position axis
(chr4 val, chr5 test) that no per-site model can enter.

## Why "embedding style" and why any backbone works here

Windows are 512 bp centred on the cytosine. Across 811 accessions a site has only
~47 unique windows (mock run over 80 real sites), so a plain full forward per
unique window costs almost nothing: the variant cache is not needed for frozen
features at this window size. The only backbone-specific knowledge is where the
centre base lands in the token stream, which `backbones.SiteEmbedder` resolves
either from fast-tokenizer offsets or by probing a fixed-width tokenizer once.
Fine-tuning through the cache would again be Carbon/DNABERT2-only.

## Code

| File | Role |
| --- | --- |
| `backbones.py` | `SiteEmbedder(backend, model_path, layer, pool_bp, bidir)`: centre-token read-out (+- pool_bp) for `carbon`, `gpn`, `plantcad`, generic `hf`. `bidir` concatenates the same read-out on the reverse complement (needed for causal Carbon: forward centre token cannot see the right half; verified: right-side mutation changes the forward read-out by exactly 0 and the bidir one by 7.5) |
| `data.py` | `MethSiteSource`: sites/accessions/counts from `dataset.h5`, accession windows via the benchmark `WindowLoader`, unique-window dedup, reference (TAIR10) window always included, cis SNV dosage per site. Minus-strand sites are reverse-complemented so the centre is always the C |
| `extract_features.py` | one GPU pass -> npz (float16 unique-window features, inverse index, targets, dosage, site metadata) |
| `fit_sites.py` | CPU: per-site RidgeCV on features (`emb`, delta from reference by default), on SNV dosage (`geno`, the bar), permuted target (`null`), features on geno residuals (`stack`); per-site Pearson r on held-out accessions, median by context x subtask; `--transfer` = one shared ridge over train-position sites scored on unseen positions |
| `run_meth.sbatch`, `run_meth_cpu.sbatch` | Slurm wrappers; first arg `training_meth.<module>` picks the entry point |
| `tests/make_mock_dataset.py` | stage-3-schema h5 with genotype-driven synthetic targets on real windows, for end-to-end smoke tests |

## Pitfalls found

1. **Benchmark `sites/pos` is 0-based.** The allc files are 0-based and stage 1
   indexed them as 1-based, so the cytosine is at TAIR10 1-based `pos + 1` on
   both strands (1999/2000 sampled sites). `MethSiteSource(pos_offset=1)` applies
   the shift and refuses to start unless >= 99 % of sites show C ('+') / G ('-')
   there. Stage 3 counts are indexed the same way, so the dataset is internally
   consistent; only the coordinate label is off. Not fixed in the benchmark build.
2. A per-site r is nan when the prediction is constant on held-out accessions
   (all of them share one window, or no held-out accession carries any SNV).
   Summaries report both the nan-dropped median and a zero-filled median.
3. `gpn` and `plantcad` backends need packages not in the svar env (`gpn`,
   `mamba-ssm`). Carbon is the only one exercised so far.

## Smoke test (mock targets, real windows, 80 CG sites, Carbon-500M bidir, dim 2048)

Mock p = sigmoid(site base + SNV effects + noise), so `geno` is the generative
model: geno median r 0.74 / emb 0.65 on context_changing, null 0.00, shared
ridge on unseen positions 0.10 vs 0.30 on seen. Pipeline validated, numbers
meaningless.

## Running (2026-09-23)

| Job | What | Output |
| --- | --- | --- |
| 20734029 `ath_meth_stage3` | builds `dataset.h5` (stage 2 verdict was MARGINAL; Andrew chose to proceed) | `$DATA/methylation/dataset.h5`, ~30 min |
| 20734086 `meth_feat_cg` (after stage 3) | Carbon features, 2,500 CG sites, positions train+val | `$SVAR_SCRATCH/runs/meth_feat_cg_carbon.npz` |
| 20734087 `meth_feat_chgchh` | same, 2,000 CHG+CHH sites | `.../meth_feat_chgchh_carbon.npz` |
| 20734088 / 20734089 `meth_fit_*` | fit_sites with `--eval-splits val test --transfer` | `*_fit.csv`, `*_summary.csv`, `*_transfer.csv` next to the npz |

Read the results from `logs/meth_fit_cg_<jobid>.out`: the emb vs geno medians by
subtask, the null, and the unseen-position transfer block.

## Next

- Backbone sweep on the same sites once features exist (Carbon 3B; GPN after
  `pip install gpn`; PlantCAD after mamba-ssm). `--layer` sweep is free.
- Baselines the benchmark spec still owes: site mean + accession offset floor,
  per-site kinship BLUP (relatedness leaks harder for methylation than
  expression: epialleles).
- If emb beats geno on `context_changing` or transfer is clearly above null on
  unseen positions, the fine-tuning question reopens (cache, Carbon only).

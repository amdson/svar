# Cross-Accession Arabidopsis Methylation Benchmark

ML dataset where the **input** is accession-specific DNA sequence and the
**target** is DNA methylation at a fixed TAIR10 position, varying across ~811
*A. thaliana* accessions. Position is held fixed; the genome varies. The task
measures whether a model predicts the effect of natural sequence variation,
not whether it recognizes generally-methylated positions.

## v1 scope (deliberate decisions)

- **Targets: GSE43857 only** (Salk MethylC-seq, 22 °C, 927 samples).
  GSE54292 (GMI, 10/16 °C) is excluded — growth temperature shifts CHH
  methylation via RdDM/CMT2 and would be a lab×temperature confound perfectly
  correlated with a batch of accessions. `source_series` is recorded per
  accession so GSE54292 can be added later as a covariate-modelled extension.
- **SNVs only** — no indels, no SVs. This is why coordinates remain comparable
  across accessions, which is the premise of the benchmark. Heterozygous and
  missing genotype calls are left as TAIR10 reference (rates logged per
  accession in `snv_arrays.h5`).
- **512 bp window** centred on the target cytosine (configurable).
- **All coordinates TAIR10** (Ensembl Plants bare `1..5` naming). No liftover.
- **KAPPA = 3** site filter ⇒ retained-site reliability > 2/3, measurement
  ceiling r = sqrt((KAPPA−1)/KAPPA) ≈ **0.82**.
- Genotype patching uses the **unfiltered** biallelic SNP set
  (`arabidopsis_1001g_biallelic_snps.vcf.gz`, 10.7 M SNPs). The MAF/geno-QC'd
  `_final` set is NOT used: rare alleles are exactly the context-changing
  signal (§3b).
- CG dyads are **not merged**; minus-strand cytosines are separate sites.
- ChrC/ChrM dropped. Caveat: the GEO `mC_calls` files contain **no chloroplast
  rows**, so per-sample bisulfite non-conversion cannot be recomputed from
  ChrC here — use Kawakatsu et al. 2016 Table S1 conversion rates to exclude
  outlier samples (TODO before freezing v1).

## Data on disk

`DATA = /90daydata/small_grains/andrew.dickson/datasets/arabidopsis`

| Piece | Path |
|---|---|
| allc files (targets) | `DATA/methylation/allc/` (848 GSM files, GEO GSE43857) |
| Sample/accession metadata | `DATA/methylation/meta/` (SOFT, manifest, `benchmark_accessions.tsv`) |
| Stage 1 accumulators | `DATA/methylation/stage1/` (dense genome arrays + `allc_qc.tsv`) |
| Stage 2 site selection | `DATA/methylation/stage2/` (parquet site tables + `report_stage2.md`) |
| Final dataset | `DATA/methylation/dataset.h5` |
| Per-accession SNVs | `DATA/methylation/snv_arrays.h5` |
| Reference | `DATA/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa` (Ensembl r63) |
| Gene annotation | `DATA/annotation/Arabidopsis_thaliana.TAIR10.63.gff3.gz` |
| TE annotation | `DATA/annotation/TAIR10_Transposable_Elements.txt` (TAIR, 31,189 TEs) |

## Accession panel

811 = intersection of GSE43857 ecotype ids (887 distinct) with the 1001G v3.1
VCF panel (1,135). Tissue: 778 leaf, 33 inflorescence-only (recorded in
`accessions/tissue`; where an ecotype had both, leaf was chosen and the other
GSM noted in `alt_gsm`). Admixture groups from the 1001G master accession
list; one accession has no group assignment (goes to train).

**Accession-axis split defaults (review before freezing):**
test = `italy_balkan_caucasus` (88), val = `asia` (69), train = the rest
(incl. `admixed` and `relict`). Set via `--acc-test-groups/--acc-val-groups`
on `stage3_extract.py`. Position axis: chr1–3 train, chr4 val, chr5 test.

## Build stages

```
build/stage1_intersect.py        # accession intersection + manifest  [done]
build/download_allc.sh           # resumable bulk GEO download        [done]
build/stage1_accession_table.py  # benchmark_accessions.tsv           [done]
build/stage1_accumulate.{py,sbatch}  # streaming accumulator pass (§2)
                                 #   + CHH 100bp window stats (§3c)
build/stage2_select.py           # site selection, GO/NO-GO report (§3)
build/stage3_extract.py          # count extraction -> dataset.h5 (§4, §6)
build/stage4_snv_arrays.{py,sbatch}  # per-accession SNV arrays (§5)
build/window_loader.py           # 512bp window extraction w/ SNV patching
```

The stage1 sbatch chains stages 1→2 and runs stage 3 only if the §3a GO
criterion holds (CG retained > 500k); otherwise it stops for review of
`stage2/report_stage2.md`. All stages are resumable (partials/tmp files are
skipped on rerun).

## Expression benchmark core (S2E design doc, dataset #2)

`DATA/expression/expression_dataset.h5` — GSE80744 UQ+gene-normalized counts
(Kawakatsu 2016 published pipeline), 24,175 genes × 727 accessions in matrix,
**665 intersect the 1001G VCF panel** (626 also in the methylation benchmark —
the joint multi-task subset). Targets: `log2_expr` and `deviation`
(per-gene deviation from panel mean — never absolute level). 22,611 genes
carry Ensembl coordinates + TSS/TTS for window inputs. Splits mirror the
methylation benchmark: accessions test=italy_balkan_caucasus (82) /
val=asia (52) / train (531); genes chr1–3/4/5 = 13,868/3,470/5,273.
Built by `build/expr_build.py`.

**T1/T2 setup:**
- `build/expr_family_split.py` — T2 family-aware gene splits from PLAZA
  dicots-05 HOMFAM (all 22,611 genes matched; families never straddle
  splits): 15,817 train / 2,242 val / 4,552 test genes
  (6,028 / 762 / 1,527 families). Stored as `genes/family_split` in the h5.
- `baselines/t1_sandwich.{py,sbatch}` — the baseline sandwich: kinship BLUP
  (LD-pruned GRM, per-gene λ by closed-form LOO), cis elastic net
  (±100 kb, MAF ≥ 0.01, unfiltered SNP set), cis-h² ceiling via
  two-component Haseman–Elston (upgradeable to REML/GEMMA). Emits
  `t1_sandwich.parquet` + `t1_report.md` with the headroom number
  (median cis-h² − EN r² on the primary set) that decides whether T1
  discriminates above the linear baseline.
- `baselines/evaluate.py` — harness for any model's predictions
  (long-format or matrix), T1/T2 modes, per-gene Spearman across held-out
  accessions, median over genes (never pooled), strata by cis-h² bin.
  T2 = family_split=test genes × acc_split=test accessions; floor is 0 by
  construction, and seen-gene baselines (BLUP/EN) do not apply there.

## Still to do

- Conversion-rate outlier exclusion (Kawakatsu Table S1) before v1 freeze.
- §3b refinement: classify creation/destruction subtypes from `snv_arrays.h5`
  (allc-based `ctx_changed` catches context shifts; absence of a destroyed
  cytosine is not distinguishable from no-coverage without the VCF).
- Baselines + evaluation harness (§7): mu_i + g_a floor, per-site logistic
  on genotype, 2-layer CNN, fine-tuned PlantCAD. Metric: per-site
  across-accession Pearson on held-out accessions, median over sites, by
  context and subtask. Never pooled over (site, accession) pairs.
- §8 diagnostic: held-out error per accession vs genetic distance from Col-0
  (ecotype 6909).

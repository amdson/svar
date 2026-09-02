# T1 / T2 benchmark mechanics

Precise description of the cross-accession expression benchmark: what the
data objects are, what a model receives and must produce, how the splits are
constructed, exactly what each baseline computes, and how evaluation works.
Companion overview: `../DATASETS.md`. Code: `../build/expr_*.py`,
`../baselines/`.

## 1. The prediction problem

One (gene, accession) pair is one prediction unit.

- **Input** — accession-specific DNA sequence around the gene, in TAIR10
  coordinates. The reference is stored once; each accession's homozygous-ALT
  biallelic SNVs (`snv_arrays.h5`, from the *unfiltered* 1001G v3.1 call
  set) are patched into the window on the fly
  (`build/window_loader.py`). Window size is the model's choice; ±4–6 kb
  around the TSS captures most Arabidopsis cis-variation. Heterozygous and
  missing calls are left as reference (accessions are inbred; rates logged).
- **Target** — `deviation[g, a] = log2(normCount[g, a] + 1) − mean_a'(log2(normCount[g, a'] + 1))`,
  i.e. how accession *a*'s expression of gene *g* differs from the panel
  mean for that gene. Models predict deviations, never absolute levels, so
  cross-gene variance (highly predictable from reference sequence alone)
  cannot contribute to the score. The subtracted mean is a per-gene
  constant, so it has no effect on the per-gene rank metric.

Data: `expression/expression_dataset.h5` — `deviation` and `log2_expr`
matrices (22,611 genes × 665 accessions, float32), `genes/*` (coordinates,
TSS/TTS, strand, `pos_split`, `family_id`, `family_split`), `accessions/*`
(`ecotype_id`, `admixture_group`, `acc_split`, `in_methylation_benchmark`).
Source: GSE80744 UQ + gene-normalized counts (Kawakatsu 2016), intersected
with the 1001G VCF panel.

## 2. Splits

**Accession axis** (`accessions/acc_split`): held out by 1001G admixture
group, never randomly — test = `italy_balkan_caucasus` (82), val = `asia`
(52), train = everything else (531, incl. `admixed` and `relict`). A random
accession split fails because accessions are related: a held-out accession
shares long identical haplotypes with training accessions, so a model can
score well by recognizing haplotypes rather than understanding variants.
Group-level holdout forces prediction across population structure. It does
not remove structure entirely — that is what the kinship-BLUP baseline
measures (§4.2).

**Gene axis, positional** (`genes/pos_split`): chromosomes 1–3 train,
4 val, 5 test. Used by the methylation benchmark and available here.

**Gene axis, family-aware** (`genes/family_split`, used by T2): every gene
belongs to a PLAZA dicots-05 HOMFAM homologous family (all 22,611 matched).
Whole families are assigned to train/val/test (~70/10/20 by gene count:
15,817 / 2,242 / 4,552 genes in 6,028 / 762 / 1,527 families). Splitting
genes individually would leak: paralogs share promoter and regulatory
sequence, so a "held-out" gene with a training paralog is partially seen.

## 3. The two tasks

**T1 — seen genes × unseen accessions.** Training may use every gene's
(sequence, deviation) pairs for the 531 train accessions. At test time the
model predicts deviations for the same genes in the 82 test-group
accessions. This asks: *given everything you learned about this locus, can
you read what its variants do in a genome you have never seen?* Human
precedent: fine-tuned deep models reach parity with the cis elastic net but
rarely beat it. The interesting region is where linear models are weakest —
accessions carrying rare or low-MAF cis alleles, which an elastic net
cannot fit (too few observations per allele) but a model with real
regulatory grammar can score zero-shot.

**T2 — unseen gene families × unseen accessions.** Training must exclude
every gene whose family is in the test split (`family_split == "test"`).
Evaluation is on those excluded genes × the test accessions. The model
never sees the locus's own expression, so nothing gene-specific can be
memorized: performance requires regulatory grammar that transfers. The
floor is exactly zero, seen-gene baselines (BLUP, elastic net) cannot run
here by construction, and the human-precedent answer is ≈ 0. Any
reproducible positive signal is a result.

## 4. The baseline sandwich (`baselines/t1_sandwich.py`)

Every model result is read against four reference points, fit per gene on
the 531 train accessions and evaluated on the 82 test accessions:

**4.1 Population mean (floor).** Predicts deviation 0 for every accession.
Per-gene correlation with a constant is undefined and scored 0. The floor
is exactly zero *by construction of the target*.

**4.2 Kinship BLUP (structure-only).** A genomic relationship matrix
`K = Z·Zᵀ/p` is built from ~82k LD-pruned SNPs (plink2
`--indep-pairwise 50 5 0.2`, MAF ≥ 0.05; `Z` = column-standardized
dosages, all 665 accessions). For each gene, kernel ridge on the train
block: `ŷ_test = K_test,train (K_train,train + λI)⁻¹ y_train`, with λ
chosen per gene from a 30-point log grid by *exact* leave-one-out (the
hat-matrix identity `e_i = (y − Hy)_i / (1 − H_ii)`, computed once from the
eigendecomposition of `K_train`). This model sees **no local sequence at
all** — only genome-wide relatedness — so anything it captures is
population structure plus trans-polygenic background. It is the leak
detector: a sequence model that fails to beat it has learned nothing local.

**4.3 Cis elastic net (the model to beat).** PrediXcan-style: for each
gene, take biallelic SNPs within ±100 kb of the TSS with MAF ≥ 0.01 in the
665 panel (missing calls imputed to 2·AF, dosages standardized; if > 3,000
SNPs, sure-screening keeps the top 3,000 by |correlation with y| — with
n = 531 this doesn't change the fit, it bounds runtime). Fit
`ElasticNetCV` (l1_ratio 0.5, 12-alpha path, 3-fold CV) on train; predict
test. Each SNP is a free parameter: this is the strongest *local* linear
machine, the plant equivalent of what human deep models struggled to beat.
It cannot transfer to unseen genes (its parameters are per-SNP, per-gene),
which is why T2 has no elastic-net bar.

**4.4 cis-h² ceiling (Haseman–Elston).** For each gene, two variance
components on the train accessions: `K_cis` (GRM from the gene's cis-window
SNPs) and `K_glob` (the pruned genome-wide GRM). For all accession pairs
i < j, regress `y_i·y_j / var(y)` on `[K_cis[i,j], K_glob[i,j], 1]` by
OLS; the fitted coefficients estimate the fractions of expression variance
attributable to cis genotype and to genome-wide background (clipped to
[0, 1]). `cis_h2` bounds what *any* model reading only local sequence can
explain: max achievable r² ≈ cis-h², max r ≈ √cis-h². HE is fast and
unbiased but noisy per gene; a REML/GEMMA confirmatory pass on the primary
gene set is the planned upgrade.

**The headroom number.** On the primary gene set (cis-h² ≥ 0.1),
`median(cis_h2 − max(en_r_test, 0)²)` is the space above the elastic net
that a better model could occupy. If it is ≈ 0, T1 cannot discriminate
models beyond the linear baseline and the benchmark's teeth are T2, the
rare-allele stratum of T1, and sign-concordance (T3). This is the first
number to read in `t1_report.md`.

## 5. Evaluation (`baselines/evaluate.py`)

For each evaluated gene *g*: Spearman rank correlation across the eval
accessions between predicted and observed deviations,
`r_g = spearman(ŷ[g, A_eval], y[g, A_eval])`. Genes need ≥ 10 non-NaN
predictions and nonzero variance on both sides; otherwise r_g is undefined
and reported as such. The headline is the **median of r_g over genes**,
with strata by cis-h² bin (and, for T2, the family split). Predictions are
supplied long-format (`gene_id, ecotype_id, pred`) or as a matrix in
dataset order.

**Never pooled.** Correlation pooled over all (gene, accession) pairs is
banned: variance between genes (or between sites, in the methylation
benchmark) far exceeds variance across accessions within one gene, so a
pooled r rewards predicting *which genes vary* — position-level knowledge —
and can look excellent while carrying zero cross-accession information.
Per-gene correlation isolates exactly the thing this benchmark exists to
measure.

## 6. The methylation analog

The methylation benchmark (`../README.md`, `../DATASETS.md`) is the same
design at single-cytosine resolution: fixed TAIR10 position, 811
accessions, (mc_count, total) count-pair targets suitable for
coverage-weighted binomial loss, per-site across-accession Pearson (median
over sites, by CG/CHG/CHH context and by subtask), with its own baseline
floor (`mu_site + g_accession`) and a measurement ceiling set explicitly by
the KAPPA site filter (κ = 3 ⇒ max r ≈ 0.82). Its `context_changing`
subtask (SNP creates/destroys/shifts the cytosine context) plays the role
that rare-allele strata play here: the mechanistically clean, learnable end
of the difficulty ladder. 626 accessions overlap between the two datasets.

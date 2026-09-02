# The two cross-accession Arabidopsis datasets

Both datasets share one premise: **the genomic position is held fixed and the
genome varies across natural accessions**. Nearly all existing plant
sequence-to-function benchmarks do the opposite (vary position on a single
reference genome), which is why models that ace them can still fail to
predict the effect of natural sequence variation. Both use the same input
side — TAIR10 plus per-accession biallelic SNVs from the unfiltered 1001
Genomes v3.1 call set (`snv_arrays.h5`, patched into windows on the fly, no
liftover, no indels in v1) — the same accession-axis splits (held-out
admixture groups: test = italy_balkan_caucasus, val = asia), and 626
accessions appear in both, forming a joint multi-task subset.

## 1. Methylation benchmark (`methylation/dataset.h5`)

**Task:** given a 512 bp accession-specific window centred on a fixed TAIR10
cytosine, predict methylation level (mc_count/total) in that accession.

- **Targets:** GSE43857 MethylC-seq (Salk, 22 °C, single growth condition),
  811 accessions after intersection with the 1001G VCF panel (778 leaf, 33
  inflorescence). GSE54292 (10/16 °C) deliberately excluded — temperature ×
  lab is perfectly confounded with a batch of accessions.
- **Site selection:** streaming accumulator over ~31M cytosines × 811
  accessions; evaluation sites require coverage in ≥90 % of accessions and
  across-accession variance > KAPPA× binomial sampling noise (KAPPA = 3 ⇒
  retained-site reliability > 2/3, measurement ceiling r ≈ 0.82). The filter
  simultaneously removes unlearnably-noisy sites and invariant sites that
  carry no cross-accession information.
- **Subtasks:** `context_changing` (a SNP creates/destroys the cytosine or
  shifts CG/CHG/CHH context — mechanistically clean, easier) vs
  `invariant_context` (same local context everywhere — tests learning beyond
  trinucleotide context). The gap between them is itself a metric.
- **Structure:** per-site (mc, total) count pairs (never pre-divided ratios,
  so losses can be coverage-weighted binomial), int16, sites × accessions;
  train-only invariant sites (low/high methylation) sampled matched to the
  evaluation set's annotation-class × CG-density distribution so models
  learn the unmethylated end without a trivial "genic ⇒ unmethylated" rule;
  separate binned 100 bp CHH target where single-site CHH is too noisy.
- **Splits:** position axis chr1–3 / 4 / 5 = train/val/test × accession axis
  by admixture group. Headline: unseen chromosome × unseen accession group.
- **Metric:** per-site across-accession Pearson r on held-out accessions,
  median over sites, by context and subtask — never pooled over
  (site, accession) pairs.

## 2. Expression benchmark (`expression/expression_dataset.h5`)

**Task:** given accession-specific sequence around a gene (TSS/TTS windows
provided), predict that accession's expression *deviation from the panel
mean* for the gene — never absolute level, so cross-gene variance cannot
inflate scores.

- **Targets:** GSE80744 rosette-leaf RNA-seq (Kawakatsu 2016 published
  UQ+gene-normalized counts), 24,175 genes × 727 accessions, of which
  **665 intersect the 1001G VCF panel**; log2(count+1) and per-gene
  deviations; 22,611 genes carry Ensembl TAIR10 coordinates.
- **Splits:** accession axis as above (531/52/82); gene axis two ways —
  positional (chr1–3/4/5) and **family-aware** (PLAZA dicots-05 HOMFAM
  families assigned whole to train/val/test: 15,817/2,242/4,552 genes in
  6,028/762/1,527 families) so paralogs cannot leak into the unseen-gene
  task.
- **Tasks:** T1 = seen genes × held-out accessions (interesting region:
  rare/low-MAF cis alleles, where linear models are underpowered);
  T2 = unseen families × held-out accessions (floor is exactly zero; any
  reproducible signal is evidence of learned regulatory grammar — the
  human-precedent answer is ≈0).
- **Baseline sandwich** (`baselines/`): population mean (floor) ≤ kinship
  BLUP (structure-only) ≤ cis elastic net (±100 kb, the model to beat)
  ≤ cis-h² (Haseman–Elston two-component ceiling). The gap between elastic
  net and cis-h² on the cis-h² ≥ 0.1 primary gene set is the number that
  says whether T1 can discriminate models at all.
- **Metric:** per-gene Spearman r across held-out accessions, median over
  genes, stratified by cis-h² bin (`baselines/evaluate.py`).

## How they differ (and why both)

Methylation offers far more signal per unit — ~10⁵–10⁶ evaluation sites, a
known measurement ceiling, and a mechanistic easy subtask — so it
discriminates models well but answers a narrower question. Expression is the
headline task aligned with the human personal-genome negative results, but
most genes have little cis-heritability, so its discriminative power rests
on the rare-allele stratum of T1, on T2's zero floor, and on sign-concordance
evaluations. Running both on shared accessions and shared input pseudogenomes
lets transfer between the tasks be measured directly.

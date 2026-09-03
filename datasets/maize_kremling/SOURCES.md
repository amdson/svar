# Maize (Zea mays) Kremling eQTL panel — data sources & provenance

What this builds: the **Kremling et al. 2018** expression panel (*Nature* 555:520,
"Dysregulation of expression correlates with rare-allele burden and fitness loss
in maize") — 3' RNA-seq of **7 tissues × ~300 Goodman/Buckler 282-panel inbreds**
— paired with the matching **HapMap3.2.1 genotypes already subset to the RNA-seq
lines**, all on **AGPv3** (B73 RefGen_v3). Target use: cis expression prediction
from SNPs (PrediXcan-style, per tissue) through the Carbon variant cache, with a
tissue axis arabidopsis/1001G lacks and fast-LD causal structure closer to SIEVE.

Everything comes from one CyVerse Data Commons DOI directory (Buckler lab), which
is what makes this build light: **no full HapMap3 download** (83M sites, hundreds
of GB) — the RNAset subset is ~2.5 GB.

Retrieved: 2026-09-03.

## Sources (all verified reachable)

Base URL (CyVerse anonymous WebDAV):
`https://data.cyverse.org/dav-anon/iplant/projects/commons_repo/curated/Kremling_Nature3RNASeq282_March2018`

| Piece | File | Notes |
|------|------|-------|
| Expression | `Expression_matrix/df_STAR_HTSeq_counts_B73_match_based_on_genet_dist_DESeq2_normed_fpm_rounded.txt` | **FPM** matrix, DESeq2 library-size normalized, all 1,771 samples × AGPv3.29 gene models. Sample IDs encode tissue (LMAD/LMAN/GRoot/GShoot/L3Base/L3Tip/L3Mid/Kern) + line. 404 MB. |
| Expression (alt names) | `…_fpm_rounded_origNames_and_Altnames.txt` | Same matrix with extra taxa-name columns — used to reconcile expression sample names with genotype taxa names. |
| Genotypes | `RNAset_genos/merged_flt_c{1..10}.hmp321.onlyRNAset_MAFover005.KNNi.hmp.txt.tar.gz` | **HapMap3.2.1** (Bukowski 2018) subset to the RNA-seq lines; **MAF > 0.05**, KNN-imputed; TASSEL hapmap text, one per chromosome, AGPv3 coordinates. ~2.5 GB total. |
| Genotypes (rare) | `RNAset_genos/merged_flt_c1-10.hmp321.onlyRNAset_MAFover0under005.KNNi.hmp.txt.tar.gz` | Rare-variant companion (0 < MAF < 0.05), all chromosomes in one tarball. **Not downloaded by default** — see caveats. |
| Reference genome | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-29/fasta/zea_mays/dna/Zea_mays.AGPv3.29.dna_sm.genome.fa.gz` | Ensembl Plants release-29, **AGPv3**, soft-masked, chromosomes only (`dna_sm.genome`). |
| Annotation | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-29/gff3/zea_mays/Zea_mays.AGPv3.29.gff3.gz` | **AGPv3.29** gene models (GRMZM… ids) — the exact gene set the counts were made against (readme: STAR + HTSEQ on AGPv3.29). |
| Field phenotypes | `Holland_Field_BLUP_phenotypes/` | BLUPs for the panel; not part of this build, listed for later. |

## Why AGPv3 (not v4/v5)

The readme states counts were made against **AGPv3.29**, and hmp321 was called on
AGPv3. Keeping genome, GFF, genotypes, and expression all on AGPv3 avoids a
CrossMap uplift *and* a gene-ID remap (GRMZM→Zm00001d), each of which loses
entries. The cost is an older assembly; acceptable for cis-window modeling.

## Caveats

* **"KNN-imputed" is not complete.** Measured on a 50k-row sample of chr10:
  ~12% of calls at biallelic sites are still `N` genome-wide (5–6% within
  gene-proximal windows, per `model_dev/probe_kremling_windows.py`). The pgen
  carries these as missing; the training data layer must decide mask-vs-ref,
  as for arabidopsis.
* **Multiallelic sites are dropped** by `hmp_to_vcf.py` (~16% of hapmap rows
  have 3–4 alleles; ~25.0M biallelic SNPs kept of roughly 30M rows). This
  matches the repo convention (every other dataset here is biallelic-SNPs
  only); splitting or top-2-allele retention would be an extension.
* **MAF > 0.05 only** in the default genotype set — rare variants live in the
  companion tarball and are KNN-imputed with the same pipeline; a rare-variant
  extension is a follow-up, not part of the first build. (Kremling's own headline
  result is about rare-allele burden, so this matters eventually.)
* **KNN imputation** means no missing calls but imputation error at low MAF;
  hets in these inbreds are mostly residual heterozygosity or imputation noise —
  the converter maps IUPAC het codes to ALT carrier = 1 (dominant coding), and
  writes a het-rate report.
* Expression sample names carry batch suffixes (e.g. `LMAD1_`, `LMAD2_` harvest
  days) and duplicated lines; the builder averages duplicates per (line, tissue)
  after log2(1+FPM), matching the TASSEL `avgDups` convention.
* L3Mid has an order of magnitude fewer samples (readme) — excluded, as in the
  paper.

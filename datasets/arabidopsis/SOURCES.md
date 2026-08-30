# Arabidopsis (Arabidopsis thaliana) — data sources & provenance

What this builds: a filtered, REF-aligned **VCF + reference genome** on **TAIR10**,
the same way as soy — but using the **1001 Genomes** SNP/short-indel release as the
genotype source. Panel: **1,135 natural accessions** (Alonso-Blanco et al. 2016,
*Cell* 166:481, "1,135 Genomes Reveal the Global Pattern of Polymorphism in
*Arabidopsis thaliana*").

Retrieved: 2026-07-21.

## Sources (all verified reachable)

| Piece | File / URL | Notes |
|------|------------|-------|
| Genotypes | `https://1001genomes.org/data/GMI-MPI/releases/v3.1/1001genomes_snp-short-indel_only_ACGTN.vcf.gz` | 1001 Genomes **v3.1**, 1,135 accessions, SNP + short-indel, **already a VCF on TAIR10** with REF/ALT. **~18 GB.** CHROM = bare `1..5`. Companion `…​.vcfidx` (35 MB) and `.md5` are in the same directory. |
| Reference genome | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa.gz` | Ensembl Plants release-63, **TAIR10**, soft-masked (37 MB). Headers bare `1..5,Mt,Pt` — the `1..5` match the VCF CHROM exactly (no rename). |
| Phenotype index | `https://arapheno.1001genomes.org/rest/phenotype/list.csv` | AraPheno REST. Lists every public phenotype (id, name, study). ~1000+ phenotypes across ~28 studies. |
| Phenotype values | `https://arapheno.1001genomes.org/rest/phenotype/<id>/values.csv` | Per-phenotype table keyed by accession id. `make phenotypes` pulls `list.csv` then loops these. (Per-study grouping: `…/rest/study/<id>/phenotypes.csv`.) |

Because the genotypes are already a VCF in the target assembly, arabidopsis needs
**no ped/map conversion, no CrossMap, and no chrom-rename** (simpler than both rice
*and* soy) — just biallelic-SNP filter → QC → `--ref-from-fa` → check.

> **Alternate genotype source (not used here): AraGWAS imputed matrix.**
> `https://aragwas.1001genomes.org/api/genotypes/download` serves the AraGWAS
> imputed SNP matrix (2,029 accessions × 10,709,466 markers, BEAGLE-imputed, no
> missing calls) as **HDF5**. It's the source described in the methods note
> ("genotype matrix … combined with metadata from the 1001G VCF"), but it needs an
> HDF5→VCF conversion. We build from the full 1001G VCF directly (soy-style); the
> tradeoff is that our VCF is **not imputed** and so carries missing calls (the
> on-the-fly variant-application path tolerates them — same as soy).

## Build pipeline

`make` runs (output `$(DATA_ROOT)/arabidopsis/arabidopsis_1001g_final.vcf.gz`
+ `.vcf.gz.tbi` + `.pgen`):

> **All VCFs in this build are bgzipped.** Uncompressed, the biallelic-SNP
> intermediate is ~50 GB and the final export ~16 GB (≈11 M SNPs × 1,135 samples of
> `0/0<tab>` text); bgzip cuts each roughly 10x. Both consumers read it natively —
> plink2 accepts a bgzipped `--vcf`, and `crop_embed`'s loader is pysam-based
> ("plain or bgzipped", `crop_embed/dataset.py`). The final VCF also gets a tabix
> index so pysam can fetch a region instead of streaming the whole file.
> `training/common/datasets.py` resolves either `.vcf` or `.vcf.gz`.

1. **biallelic-SNP filter** — `bcftools view -t 1,2,3,4,5 --types snps
   --min-alleles 2 --max-alleles 2`. Restricts to the 5 nuclear chromosomes and
   drops indels + multi-allelic sites. This single stream reads the ~18 GB source
   once (`-t` needs no index). **No `bcftools sort`:** the 1001G header defines no
   `##contig` lines, so `bcftools sort` (which spills temp BCF, and BCF *requires*
   contigs in the header) dies with `[E::bcf_write] Unchecked error`. The release
   is already coordinate-sorted, and plink2 `--sort-vars` (step 4) re-orders
   variants without needing contig headers, so order is still guaranteed.
2. **QC** (plink2) — soy-equivalent defaults, overridable:
   - `MAF=0.01`   (minor-allele frequency floor)
   - `GENO=0.2`   (drop variants with >20% missing)
   - `MIND=`      (empty → no sample filter; set `make MIND=0.1` to also drop
     low-call-rate samples)
3. **LD pruning** (optional, off) — `make LD_PRUNE=1` runs
   `plink2 --indep-pairwise 50 5 0.4`. `--set-all-var-ids '@:#'` gives every SNP a
   `chr:pos` id first, so pruning works even if the source lacks variant IDs.
4. **REF-from-fa** — two plink2 calls: `--ref-from-fa force --sort-vars
   --make-pgen` builds a sorted, REF-aligned pgen, then `--pfile … --export vcf`
   writes the VCF. Split because plink2 forbids `--sort-vars` in any command that
   also carries `--export vcf` (even with `--make-pgen`).
5. `make check` — `bcftools norm --check-ref w` confirms REF vs the genome.

### Imputation — skipped
The 1001G release is **not** imputed; the final VCF therefore still contains missing
calls. (To start from an imputed matrix instead, use the AraGWAS HDF5 above.)

### Tooling note
All steps use **plink2** + **bcftools**. FASTA indexing uses **pyfaidx** (samtools
isn't in the `svar` env; `bcftools` has no `faidx`) — same as soy.

## Loading into UniqueWindowDataset

`crop_embed`'s loader requires **integer chromosome names** — both
`load_snps_from_vcf._parse_chrom` and the FASTA-key filter
(`{int(name): name … if name.isdigit()}`). The 1001G VCF is already `1..5` and the
Ensembl TAIR10 genome exposes bare `1..5` keys (Mt/Pt are non-digit and get
filtered out, matching the VCF's 5-chromosome restriction), so no rename is needed.

> **Scale:** the panel is 1,135 accessions — well under soy's 20,087. The
> fingerprint index is `(n_samples × n_windows)`; the whole panel is tractable, but
> subset samples (the `samples=` arg) for quick work.

## Caveats (correctness)

- **Everything is TAIR10.** The 1001G v3.1 variants were called against TAIR10 and
  Ensembl `Arabidopsis_thaliana.TAIR10` is the same assembly; `make check` (0
  mismatches expected) confirms the positions match the genome.
- **Indels dropped.** The source is `snp-short-indel`; we keep SNPs only, since the
  window-substitution model applies point changes. Revisit if indels are needed.
- **Two larger VCFs exist in the release** — `…_with_tair10_only_ACGTN.vcf.gz`
  (132 GB, whole-genome incl. invariant sites) is **not** what we want; the
  `snp-short-indel_only` file is the variants-only call set.
- **Phenotypes** download independently (`make phenotypes`); `make pheno` then
  aligns them onto the genotypes (see below).

## Phenotypes → aligned onto the genotypes

The AraPheno accession ids **are** the VCF sample IIDs (both are 1001-Genomes
accession numbers), so the join is an exact id match — no key translation. Of the
1,135 genotyped accessions, **1,041** have ≥1 phenotype (94 have none; 476
phenotyped accessions aren't in this VCF).

`make pheno` runs `scripts/build_arabidopsis_phenotypes.py`, restricting every
AraPheno table to genotyped accessions and emitting three views:

| Output | Shape | Use |
|--------|-------|-----|
| `arabidopsis_pheno_aligned.csv` | one row per genotyped accession, `.psam` order (1,135 rows) × trait columns named `p<phenotype_id>`, plus a `matched` flag | **What the training pipeline reads.** `training/common/features.py` does `read_csv(...).set_index("IID")` and reindexes onto the sample list, so it needs a *unique* IID key — which the `(IID, rep)` matrix below cannot provide. Replicates are therefore **averaged** here. Columns are ids, not names, because AraPheno names contain commas and sweep configs pass trait sets as comma-joined strings; look ids up in the coverage table. Restricted to traits covering ≥ `--aligned-min-genotyped` accessions (default 100) so `traits=all` stays meaningful. |
| `arabidopsis_pheno_matrix.csv` | rows keyed `(IID, rep)` in `.psam` order × cols = ~536 phenotype ids | the "one big matrix" — Y for modelling. Replicates kept on **separate rows** (never averaged): within each (accession, trait) the k values fill rows 0..k-1, so an accession spans as many rows as its most-replicated trait (~3,900 rows total). For a trait, select its column and `dropna`. **Sparse** — most cells NaN. |
| `arabidopsis_pheno_long.csv` | tidy: `IID, phenotype_id, phenotype_name, study, value` (replicates kept) | filter to one trait without carrying a mostly-empty matrix. |
| `arabidopsis_pheno_coverage.csv` | per phenotype: `phenotype_id, name, study, n_genotyped, n_values`, sorted by coverage | pick well-powered traits — `n_genotyped` is the real per-GWAS sample size (median ~a few hundred). |

The matrix is deliberately **not** filtered to complete cases (unlike soy's
`soy_pheno_complete.csv`): with ~536 heterogeneous traits there is no common
complete set, so you subset per analysis (use the coverage guide).

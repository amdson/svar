# Soy (Glycine max) — data sources & provenance

What this builds: a filtered, REF-aligned **VCF + reference genome** on
**Wm82.a2.v1**, the same way as rice — using the **USDA SoySNP50K** diversity panel
as the genotype source — **plus a phenotype table of 11 traits** joined on by USDA
accession id (`make pheno`). The genotypes are our own, PI-keyed and
coordinate-carrying; joining the SoyDNGP phenotype table onto them recovers the
same ~14,460 complete-phenotype accessions as the GP-WAITER `soybean14460` panel,
without needing that paper's anonymised (id-stripped, coordinate-less) matrix.

## Sources (both verified reachable)

| Piece | File / URL | Notes |
|------|------------|-------|
| Genotypes | `https://data.soybase.org/Glycine/max/diversity/Wm82.gnm2.div.Song_Hyten_2015/glyma.Wm82.gnm2.div.Song_Hyten_2015.vcf.gz` | USDA SoySNP50K, Song & Hyten 2015 (G3 5:1999-2006). 20,087 accessions × 42,509 biallelic SNPs, **already a VCF on Wm82.a2** with REF/ALT. CHROM `glyma.Wm82.gnm2.Gm01..Gm20` (+ scaffolds). |
| Reference genome | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/glycine_max/dna/Glycine_max.Glycine_max_v2.1.dna_sm.toplevel.fa.gz` | Ensembl Plants release-63, Glycine_max_v2.1 (= Wm82.a2.v1), soft-masked. Headers bare `1..20` (chrom 1 length 56,831,624 == SoySNP50K Gm01 — same coordinates). |
| Phenotypes | `https://raw.githubusercontent.com/IndigoFloyd/SoybeanWebsite/main/data.csv` | SoyDNGP project (Gao et al.; xtlab.hzau.edu.cn) `data.csv`: 20,506 accessions × 24 traits, keyed by USDA accession id (`acid`). This is the historical USDA-GRIN-derived phenotype corpus and the source GP-WAITER's `soybean14460` was built from. We use 11 traits (see below). |

Because the genotypes are already a VCF in the target assembly, soy needs **no
ped/map conversion and no CrossMap** (unlike rice) — just rename chroms → filter →
QC → `--ref-from-fa` → check.

> **Genome choice.** The SoyBase Wm82.gnm2.DTC4 *soft-masked* file is **truncated on
> the server** (133 MB; only 9 of 20 chromosomes — verified its server size equals
> the partial), so we use the Ensembl genome (same assembly, complete, and its bare
> `1..20` headers are exactly what we rename the VCF to). SoyBase `genome_main`
> (287 MB, unmasked) is intact but Gm-named.

## Build pipeline

`make` runs (output `$(DATA_ROOT)/soy/soysnp50k_a2_final.vcf` + `.pgen`):

0. **rename chroms** — `bcftools annotate --rename-chrs` maps
   `glyma.Wm82.gnm2.Gm01..Gm20 → 1..20` and keeps only those 20 (scaffolds
   dropped). This makes CHROM match the Ensembl genome AND gives the integer
   chromosome names `UniqueWindowDataset` requires (see "Loading" below).
1. **biallelic filter** — `bcftools view --min-alleles 2 --max-alleles 2` + sort
   (drops the multi-allelic sites the paper also removed).
2. **QC** (plink2) — paper-equivalent defaults, overridable:
   - `MAF=0.01`   (paper: MAF < 0.01 removed)
   - `GENO=0.2`   (paper: variant missing rate > 0.2 removed)
   - `MIND=`      (empty → no sample filter; paper filtered variants only). Set
     `make MIND=0.1` to also drop low-call-rate samples.
3. **LD pruning** (optional, off by default) — `make LD_PRUNE=1` runs
   `plink2 --indep-pairwise 50 5 0.4` (the paper's params, used there only for its
   soybean1861 set) and keeps the pruned SNPs. Tune with `LD_PARAMS="50 5 0.4"`.
4. **REF-from-fa** — `plink2 --ref-from-fa force` aligns each REF to the genome.
5. `make check` — `bcftools norm --check-ref w` confirms REF vs the genome.

### Imputation (Beagle) — skipped
The paper imputed missing genotypes with **Beagle 5.4 (22Jul22.46e)** after QC.
That step is intentionally **omitted** (Beagle isn't installed and we opted to use
plink2 only), so the final VCF still contains missing calls. The downstream
on-the-fly variant-application path tolerates these; revisit if a fully-imputed
matrix is needed.

### Tooling note
All steps use **plink2** (not the paper's PLINK v1.9). The QC thresholds are
identical; `--indep-pairwise 50 5 0.4` has the same window/step/r² meaning in both.
FASTA indexing uses **pyfaidx** (samtools isn't in the `svar` env; `bcftools` has no
`faidx`).

## Phenotypes (`make pheno`)

The SoyDNGP `data.csv` is a phenotype table keyed by USDA accession id (`acid`
column, PI/FC numbers). `make pheno` downloads it (as `soydngp_data_source.csv`)
and runs `scripts/build_soy_phenotypes.py`, which joins it onto our VCF sample
list **by exact accession id** and writes:

- `soy_pheno_aligned.csv` — one row per VCF sample, in `.psam` (IID) order; the 11
  trait columns plus `matched`/`complete` flags. `NaN` where a sample has no match
  or the source has no value. Align targets to the dataset by joining on `IID`.
- `soy_pheno_complete.csv` — the complete-case subset (all 11 traits present),
  ready to use directly as `Y`.

The 11 traits (as named in `data.csv`): `protein`, `oil`, `Linoleic`,
`Linolenic`, `R1` (flowering date), `R8` (maturity date), `Hgt` (plant height),
`Ldg` (lodging), `SQ` (seed quality), `SdWgt` (100-seed weight), `Yield`. These
are standard USDA soybean germplasm descriptors (also queryable, id-keyed, from
the SoyBase GRIN Data Explorer — `soybase.org/tools/grin/`).

**Join result (verified):** 16,960 of the 20,087 VCF samples match by exact id;
**14,460** have all 11 traits — this complete-case set *is* the GP-WAITER
`soybean14460` panel, reconstructed on our own coordinate-carrying genotypes. The
rest are genotype-only. Exact-id matching is deliberate (it reproduces the
published panel); a few hundred more could be recovered by normalising id suffixes
(e.g. `PI594471B` vs `PI594471`) — not done, to stay faithful to the panel.

> **Why not the anonymised `soybean14460` matrix?** GP-WAITER's Zenodo release is a
> bare genotype matrix (14,460 × 39,707, values `{-1,0,1}`) with **no accession ids
> and no SNP coordinates** — so it can neither be joined to our data nor fed to the
> coordinate-based embedding pipeline. Joining the *source* phenotypes onto our own
> PI-keyed, coordinate-carrying VCF sidesteps both problems. (The repo's
> `predict/snp.txt`, a 32,033-SNP `ChrNN_pos` list, is SoyDNGP's own model input —
> not the matrix's 39,707 columns — and is unneeded here.)

## Loading into UniqueWindowDataset

`crop_embed`'s loader requires **integer chromosome names** — both
`load_snps_from_vcf._parse_chrom` and the FASTA-key filter
(`{int(name): name … if name.isdigit()}`) — exactly like the rice Ensembl 1..12
build. That's why step 0 renames CHROM to `1..20` and we use the bare-`1..20`
Ensembl genome. Verified end-to-end:

```
final VCF: 39,556 SNPs, chroms 1..20, 20,087 samples (PI…); REF check 0 mismatches
UniqueWindowDataset(half_window=500, 50 samples): 75,058 unique windows,
  windows yield 1000 bp ACGT sequences across chroms 1, 11, 20.   → PASS
```

> **Scale:** the panel is 20,087 accessions. The fingerprint index is
> `(n_samples × n_windows)`, so building the dataset over *all* samples is heavy
> (~38k windows × 20k samples). Subset samples (the `samples=` arg) for quick work.

## Caveats (correctness)

- **Everything is Wm82.a2.v1.** Do not mix in SoyBase gnm5/gnm6 dirs (those are
  a4/a6). `div.Song_Hyten_2015` **gnm2** and Ensembl Glycine_max_v2.1 are both a2;
  REF concordance (0 mismatches) confirms the SoySNP50K positions match the genome.
- **CHROM naming:** SoySNP50K = `glyma.Wm82.gnm2.Gm01`; Ensembl = bare `1..20`; NCBI
  `GCF_000004515.4` (= Glycine_max_v2.0 = Wm82.a2.v1) = `NC_038xxx`. We rename the
  VCF to `1..20` to match Ensembl (and satisfy the dataset loader).
- The SoyBase datastore directory pages are JS file browsers, but the **direct file
  URLs** above are wget-able. NCBI `GCF_000004515.4` is a stable genome fallback.


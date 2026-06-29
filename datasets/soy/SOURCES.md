# Soy (Glycine max) — data sources & provenance

What this builds: a filtered, REF-aligned **VCF + reference genome** on
**Wm82.a2.v1**, the same way as rice — but using the **USDA SoySNP50K** diversity
panel as the genotype source. This is a usable soy dataset; it is **not** the
GP-WAITER paper's exact `soybean14460` panel (that build is parked — see the end).

## Sources (both verified reachable)

| Piece | File / URL | Notes |
|------|------------|-------|
| Genotypes | `https://data.soybase.org/Glycine/max/diversity/Wm82.gnm2.div.Song_Hyten_2015/glyma.Wm82.gnm2.div.Song_Hyten_2015.vcf.gz` | USDA SoySNP50K, Song & Hyten 2015 (G3 5:1999-2006). 20,087 accessions × 42,509 biallelic SNPs, **already a VCF on Wm82.a2** with REF/ALT. CHROM `glyma.Wm82.gnm2.Gm01..Gm20` (+ scaffolds). |
| Reference genome | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/glycine_max/dna/Glycine_max.Glycine_max_v2.1.dna_sm.toplevel.fa.gz` | Ensembl Plants release-63, Glycine_max_v2.1 (= Wm82.a2.v1), soft-masked. Headers bare `1..20` (chrom 1 length 56,831,624 == SoySNP50K Gm01 — same coordinates). |

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

---

## Parked: the GP-WAITER `soybean14460` paper panel

The original goal was the paper's exact panel — 14,460 accessions × 39,707 SNPs +
11 traits (Li et al., *Nat. Commun.* 2026, 17:4427). It is **blocked on a missing
column→SNP map**, recorded here so it can be resumed:

- **Genotype + phenotype:** Zenodo record **18779208** (CC-BY-4.0),
  `soybean14460.zip` → `soybean14460_gen.csv` + `_phe.csv`. (The DOI first cited,
  `10.5281/zenodo.18809685`, is the GP-WAITER *software* release; its README points
  to the data. Equivalent panels also appear in record 18476279.)
- **Verified format:** `…_gen.csv` is a **headerless** 14,460 × 39,707 matrix of
  **{-1, 0, 1}** — no accession IDs (positional, aligned to the phenotype by row)
  and **no SNP IDs or positions**. `…_phe.csv` is 11 trait columns
  (`protein,oil,Linoleic,Linolenic,R1,R8,Hgt,Ldg,SQ,SdWgt,Yield`), no ID column.
- **The blocker:** with unlabeled SNP columns, you cannot attach REF/ALT. The
  column→SNP map is in neither Zenodo record nor the GP-WAITER repo (only the demo
  "O" panel shipped a per-column site file) — it must come from the paper supplement
  or the authors. With it, the plan was: map columns → SNP → REF/ALT from the
  SoySNP50K VCF's 5 fixed columns, **check code polarity** (paper: 1=hom-ref,
  −1=hom-alt, 0=het — verify per-SNP mean vs VCF allele freq; a flip swaps every
  REF/ALT), decode to A/C/G/T, emit a VCF, then run the same QC tail as above.

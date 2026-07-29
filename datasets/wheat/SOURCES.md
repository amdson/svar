# Wheat (Triticum aestivum) — data sources & provenance

Panel: **CIMMYT "Seeds of Discovery" (SeeDs) Iranian bread-wheat landrace**
collection — hexaploid *T. aestivum* (AABBDD). ~2,440 Iranian landraces,
DArTSeq-genotyped and phenotyped under drought & heat at Ciudad Obregón. This is
the dataset behind Crossa/Sehgal et al., *"Genomic Prediction of Gene Bank Wheat
Landraces"* (G3, 2016; doi:10.1534/g3.116.029637) and *"Worldwide Selection
Footprints for Drought and Heat in Bread Wheat (T. aestivum L.)"*.

## Sources

| Piece | File / origin | Notes |
|------|---------------|-------|
| Genotypes | `MEXICO AND IRANIAN -- MARKERS.zip.gz` → `Iranian_Samples.csv` | DArTSeq SNP report, **ACCESS-GATED (CIMMYT); not re-downloadable**. 2,544 sample columns / 52,591 markers. Header carries BOTH CIMMYT `GID` (row 1) and DArT `SEEDDIV####` (row 2) per sample — this bridges genotype↔phenotype keys. 17 metadata cols (`AlleleID,CloneID,AlleleSequence,SNP,SnpPosition,...`), then samples. |
| Phenotypes | `dataverse_files.zip` → `PHENOTYPIC DATA IRANIAN/*.xlsx` | **ACCESS-GATED**. Keyed by CIMMYT `GID`. Files: DTH/DTM under Heat & Drought; PHT (drought); QUALITY (tkw, testw, kernel length/width, hardness, protein, SDS). |
| Reference genome | Ensembl Plants r60, `Triticum_aestivum.IWGSC.dna_sm.toplevel.fa.gz` | **IWGSC RefSeq v1.0** (Chinese Spring), soft-masked, ~14 Gb, 21 chromosomes `1A..7D` + `Un`. Auto-downloaded by the Makefile. |

> The two access-gated archives must be placed under `$(RAW)` (default
> `persian_wheat/`); the Makefile errors clearly if they're missing. They are NOT
> committed and NOT publicly re-downloadable.

## The core difference vs rice/soy: DArTSeq has no coordinates

DArTSeq markers are **69 bp sequence tags** with the SNP at a within-tag position
(`SnpPosition`) — there is **no chromosome/position**. So the build recovers
coordinates by **aligning the tags to the genome** (the wheat analog of rice's
CrossMap lift), rather than starting from a coordinate-bearing VCF.

## Build pipeline (`make`)

1. **extract tags** (`wheat_extract_tags.py`) — pull the 69 bp `AlleleSequence`
   per marker from `Iranian_Samples.csv` → `tags.fasta` (+ `tags_meta.tsv`).
2. **genome** — download IWGSC RefSeq v1.0; gunzip + faidx (`genome.fa`, headers
   `1A..7D,Un`); then rename headers → integers `1..22` (`wheat_genome.fa`) so the
   VCF/genome load through crop_embed's integer-chrom loader (like soy/rice).
3. **align** (`minimap2 -ax sr -I 20G`) tags → `genome.fa`; sort → `aln.bam`
   (CSI-indexed: wheat chromosomes exceed the 512 Mb BAI limit).
4. **validate + project** (`wheat_validate_project.py`) — keep markers that map
   uniquely (`MAPQ>=30`) with high flanking identity (`>=0.90`) and are biallelic
   concordant (genome base ∈ {ref,alt}); project each SNP to a genome coordinate.
   → `marker_coords.tsv`.
5. **build VCF** (`wheat_build_vcf.py`) — REF = genome allele, chroms → `1..22`,
   DArT calls (0=hom-ref, 1=hom-ALT, 2=het, -=missing) → VCF GT (flipping homs
   where the genome base is the DArT alt). Samples keyed by GID (duplicates
   collapsed). → `wheat_dartseq.vcf` (+ `chrom_map.tsv`).
6. **phenotypes** (`wheat_build_pheno.py`) — unpack the dataverse xlsx, join by GID
   onto VCF sample order → `wheat_pheno_aligned.csv` (+ `wheat_pheno_complete.csv`).
7. **check** — `bcftools norm --check-ref` vs `wheat_genome.fa`.

## Build result (verified 2026-07-28)

- Alignment: 52,591 tags → 73.0% mapped; **20,487 uniquely (MAPQ≥30)**; subgenome
  split A31/B33/D35 (balanced = real). 27% unmapped (adapter-read-through short
  anchors); ~34% multi-map (A/B/D homoeology).
- Validation: mean flanking identity **0.995**; SNP placeable 99.0%; **biallelic
  concordance 100%** → **20,280 markers** kept. `make check`: 0 REF mismatches.
- VCF: **20,280 markers × 2,442 GID-keyed samples**, chroms `1..22`.
- Phenotypes: 2,438/2,442 samples matched; 1,980 complete across all 12 traits
  (agronomic traits DTH/DTM/PHT cover ~2,400; quality traits ~2,008).

## Notes / caveats

- **Genotype encoding:** standard DArT 1-row SNP (`0` hom-ref, `1` hom-ALT, `2`
  het, `-` missing) — note `1`=hom-alt / `2`=het is a DArT quirk, mapped correctly.
- **Chromosome rename** `1A..7D,Un → 1..22` (see `chrom_map.tsv`); mirrors soy's
  `Gm01→1` so `crop_embed`'s integer-chrom loader works unchanged.
- **Coverage:** 20,280 of 52,591 markers survive unique mapping — expected for
  short DArTSeq tags on the hexaploid genome; adapter-trimming + a short-read
  aligner could recover more (unmapped 27%) if a larger panel is wanted.
- **Alt marker set:** `DArTSeq_..._102474markers.csv.zip` has 102,474 markers but
  no GID header; we use the MARKERS export (GID-keyed, 52,591) instead.

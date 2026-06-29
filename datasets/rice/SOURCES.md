# Rice (Oryza sativa) — data sources & provenance

Panel: **RiceDiversity 44K** — 413 *O. sativa* accessions genotyped on the 44K
SNP array (Zhao et al. 2011, *Nat. Commun.*). The genome assembly is **IRGSP-1.0**
(Ensembl Plants; equivalent to the MSU7 pseudomolecules). The array SNPs are
natively in **MSU6** coordinates and are remapped to IRGSP-1.0 during the build.

## Downloads

| File | URL | Notes |
|------|-----|-------|
| Reference genome (soft-masked FASTA) | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa.gz` | Ensembl Plants **release-62**, IRGSP-1.0 assembly. `dna_sm` = soft-masked. |
| Remap chain (MSU6 → IRGSP-1.0) | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/assembly_chain/oryza_sativa/MSU6_to_IRGSP-1.0.chain.gz` | Used by CrossMap to lift SNP coordinates. |
| SNP genotypes (PLINK ped/map) | `http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity.44K.MSU6.Genotypes_PLINK.zip` | RiceDiversity 44K GWAS set, MSU6 coords. Unpacks to a dotted folder name — the Makefile renames it to `RiceDiversity_44K_Genotypes_PLINK/`. Stem: `sativas413`. |
| Trait data (34 phenotypes) | `http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity_44K_Phenotypes_34traits_PLINK.txt` | Phenotype table keyed by accession. |
| SNP flanking sequence | `http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity.44K.MSU6.SNP_flanking_seq.txt` | MSU6 flanking seq per SNP; used only by the `check` sanity test. |

`http://www.ricediversity.org/...` is plain HTTP and occasionally slow; `wget -c`
(the Makefile default) resumes partial downloads.

## Coordinate systems

- SNP array + flanking table: **MSU6 / IRGSP build 4**.
- Reference genome + final VCF: **IRGSP-1.0 (MSU7)**.
- The CrossMap step (`MSU6_to_IRGSP-1.0.chain`) is what bridges them. Sites that
  fail to lift are written to `*_msu7.vcf.unmap`.

## Build → outputs

`make` runs: ped/map → VCF (MSU6) → CrossMap → biallelic filter → QC
(`--mind 0.1 --geno 0.2 --maf 0.05`) → `--ref-from-fa force`. Final artifacts in
`$(DATA_ROOT)/rice/`:

- `sativas413_msu7_final.vcf` — final biallelic, QC'd, REF-aligned VCF.
- `sativas413_msu7_final.{pgen,pvar,psam}` — PLINK2 binary equivalent.

## Sanity check

`make check` runs `scripts/test_dataset_flanking.py`, which confirms (a) each VCF
REF allele equals the reference-genome base at that position, and (b) the 16 bp
flanking each SNP match the MSU6 flanking-seq table. `make test-consensus`
additionally writes a few per-sample consensus genomes for manual inspection.

## Relationship to ~/rice_data

The original build lives in `~/rice_data` and predates this directory;
`crop_embed/data/coords.py` still points there. This Makefile reproduces that
pipeline into scratch — it does not read from or modify `~/rice_data`.

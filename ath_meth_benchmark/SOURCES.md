# Data sources & provenance

Every external URL the Makefile touches, in svar/datasets style. Retrieved
2026-09-01/02 unless noted. Nothing under `$(DATA_ROOT)` is committed.

## Shared Arabidopsis inputs (same as svar/datasets/arabidopsis)

| Piece | URL | Notes |
|---|---|---|
| TAIR10 genome | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa.gz` | Ensembl Plants r63, soft-masked, bare `1..5` headers |
| 1001G v3.1 VCF | `https://1001genomes.org/data/GMI-MPI/releases/v3.1/1001genomes_snp-short-indel_only_ACGTN.vcf.gz` | ~18 GB, 1,135 accessions, TAIR10, CHROM bare `1..5` |
| Gene GFF3 | `https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.63.gff3.gz` | genes/ncRNA_genes; carries NO TE features |
| TE annotation | `https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_transposable_elements/TAIR10_Transposable_Elements.txt` | 31,189 TEs w/ family+superfamily. The classic `/download_files/...` path 403s; this API path works. RefSeq/Ensembl carry no TAIR TE set. |
| Accession metadata | `https://tools.1001genomes.org/api/accessions.csv?query=SELECT * FROM tg_accessions;` | 1,135 rows; col 11 = admixture group (9 groups + admixed) |

## Methylation targets (GSE43857 only — v1 scope)

| Piece | URL | Notes |
|---|---|---|
| Series metadata | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE43nnn/GSE43857/soft/GSE43857_family.soft.gz` | per-GSM ecotype id, tissue, allc URL |
| allc files | per-GSM `https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1085nnn/GSM<id>/suppl/GSM<id>_mC_calls_<name>.tsv.gz` (manifest built from the SOFT) | 848 files ~110 GB. methylpy-style columns w/ header; chrom bare `1..5`; **no ChrC/ChrM rows**, so bisulfite non-conversion must come from Kawakatsu 2016 Table S1, not ChrC. NCBI intermittently drops connections — the downloader retries and validates gzip integrity. |
| GSE54292 (Dubin 10/16 °C) | not downloaded | deliberately excluded in v1: temperature×lab confound |

## Expression targets (GSE80744)

| Piece | URL | Notes |
|---|---|---|
| Normalized counts | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE80nnn/GSE80744/suppl/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz` | 24,175 genes × 727 accessions (`X<ecotype>` columns), Kawakatsu 2016 published pipeline; 665 intersect the VCF panel |
| Gene families | `https://ftp.psb.ugent.be/pub/plaza/plaza_public_dicots_05/GeneFamilies/genefamily_data.HOMFAM.csv.gz` | PLAZA dicots-05 HOMFAM; species code `ath`; all 22,611 coordinate-bearing genes matched |

## SIEVE (Brachypodium, standalone benchmark)

| Piece | URL | Notes |
|---|---|---|
| Everything public | `https://zenodo.org/records/18236856/files/<file>?download=1` | EMPRES/SIEVE record. Used: `peer.expression.csv` (796 lines × 27,914 genes, PEER-corrected — values are LINEAR-scale despite the README saying log10(1+TPM); build applies log2(1+x)), `peer.genes.csv`, `peer.samples.csv`, `snps.combined.M5.filtered.renamed.vcf.gz` (581,803 records; CHROM `Bd1..Bd5`; "renamed" = samples), `gene.npy`/`family.npy`/`group_for_cross_validation.npy` (EMPRES training arrays — Bd21 v3.0 `Bradi` ids, ZERO overlap with the Bd21-3 expression gene ids). Embeddings/models (~130 GB) not downloaded. |
| Bd21-3 v1.1 reference | Phytozome (login-gated): `https://phytozome-next.jgi.doe.gov/info/BdistachyonBd21_3_v1_1` | assembly FASTA + gene GFF3 + annotation_info via the JGI curl bundle (arrives as zips). VCF REF validated 3000/3000 against this FASTA — coordinates are Bd21-3. annotation_info has NO Bradi synonym column; family split uses best-hit-arabi/rice proxy (`sieve_family_split.py`). |

## Reference publications

- Kawakatsu et al. 2016 Cell (1001 Epigenomes; GSE43857, GSE80744)
- 1001 Genomes Consortium 2016 Cell (v3.1 panel, admixture groups)
- Vahedi Torghabeh et al. 2026 bioRxiv 10.64898/2026.02.27.708524 (EMPRES; SIEVE expression)
- Moslemi et al. 2026 bioRxiv 10.64898/2026.03.31.715642 (SIEVE population/fitness)

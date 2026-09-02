# ath_meth_benchmark/Makefile — three cross-accession benchmark datasets.
#
# Follows the svar/datasets conventions (common.mk style, self-contained since
# this repo lives outside datasets/): data lands on scratch under
# $(DATA_ROOT)/<dataset>, only code + docs are tracked, every URL lives in
# SOURCES.md, tools come from the svar conda env (see requirements.txt).
#
#   conda activate svar
#   make meth-meta      # methylation: metadata, intersection, annotations
#   make meth-download  # methylation: 848 GSE43857 allc files (~110 GB)
#   make meth-stages    # methylation: submit stage 1→2→3 Slurm chain
#   make meth-snv       # methylation: submit stage-4 SNV-array Slurm job
#   make expr           # expression: download + build + family split
#   make expr-t1        # expression: submit T1 baseline-sandwich Slurm job
#   make sieve          # SIEVE: download + build + family split
#                       # (reference FASTA/GFF3 need a one-time Phytozome
#                       #  login — `make sieve-reference` explains)
#
# Slurm targets SUBMIT and return; they do not wait. Everything is resumable.

SVAR_SCRATCH ?= /90daydata/small_grains/andrew.dickson
DATA_ROOT    ?= $(SVAR_SCRATCH)/datasets
CONDA_ENV    ?= svar
CONDA_RUN    := conda run -n $(CONDA_ENV)
PY           := $(CONDA_RUN) python

BENCH := $(CURDIR)
ARA   := $(DATA_ROOT)/arabidopsis
METH  := $(ARA)/methylation
EXPR  := $(ARA)/expression
SIEVE := $(DATA_ROOT)/brachypodium_sieve

.DEFAULT_GOAL := help
help:
	@grep -E '^#( |$$)' Makefile | head -22

# ══ Shared Arabidopsis inputs (also built by svar/datasets/arabidopsis) ═══════
$(ARA)/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa:
	wget -c -P "$(ARA)" https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa.gz
	gunzip -k "$@.gz" && $(PY) -c "import pyfaidx; pyfaidx.Fasta('$@')"

$(ARA)/1001genomes_snp-short-indel_only_ACGTN.vcf.gz:
	wget -c -P "$(ARA)" https://1001genomes.org/data/GMI-MPI/releases/v3.1/1001genomes_snp-short-indel_only_ACGTN.vcf.gz

$(ARA)/annotation/Arabidopsis_thaliana.TAIR10.63.gff3.gz:
	mkdir -p "$(ARA)/annotation"
	wget -c -P "$(ARA)/annotation" https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.63.gff3.gz

$(ARA)/annotation/TAIR10_Transposable_Elements.txt:
	mkdir -p "$(ARA)/annotation"
	curl -sL "https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_transposable_elements/TAIR10_Transposable_Elements.txt" -o "$@"
	head -1 "$@" | grep -q Transposon_Name   # TAIR gates some paths; verify

# ══ Methylation benchmark ═════════════════════════════════════════════════════
$(METH)/meta/GSE43857_family.soft.gz:
	mkdir -p "$(METH)/meta"
	wget -c -P "$(METH)/meta" https://ftp.ncbi.nlm.nih.gov/geo/series/GSE43nnn/GSE43857/soft/GSE43857_family.soft.gz

$(METH)/meta/accessions_1001g.csv:
	mkdir -p "$(METH)/meta"
	curl -sL "https://tools.1001genomes.org/api/accessions.csv?query=SELECT%20*%20FROM%20tg_accessions;" -o "$@"

.PHONY: meth-meta meth-download meth-stages meth-snv
meth-meta: $(METH)/meta/GSE43857_family.soft.gz $(METH)/meta/accessions_1001g.csv \
           $(ARA)/annotation/Arabidopsis_thaliana.TAIR10.63.gff3.gz \
           $(ARA)/annotation/TAIR10_Transposable_Elements.txt
	$(PY) build/stage1_intersect.py
	$(PY) build/stage1_accession_table.py

meth-download: meth-meta          # 848 files, ~110 GB, resumable
	bash build/download_allc.sh

meth-stages:                      # accumulator → site selection → extraction
	sbatch build/stage1_accumulate.sbatch

meth-snv: $(ARA)/1001genomes_snp-short-indel_only_ACGTN.vcf.gz
	sbatch build/stage4_snv_arrays.sbatch

# ══ Expression benchmark ══════════════════════════════════════════════════════
$(EXPR)/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz:
	mkdir -p "$(EXPR)"
	wget -c -P "$(EXPR)" "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE80nnn/GSE80744/suppl/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz"

$(EXPR)/genefamily_data.HOMFAM.csv.gz:
	mkdir -p "$(EXPR)"
	wget -c -O "$@" "https://ftp.psb.ugent.be/pub/plaza/plaza_public_dicots_05/GeneFamilies/genefamily_data.HOMFAM.csv.gz"

.PHONY: expr expr-t1
expr: $(EXPR)/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz \
      $(EXPR)/genefamily_data.HOMFAM.csv.gz \
      $(METH)/meta/accessions_1001g.csv \
      $(ARA)/annotation/Arabidopsis_thaliana.TAIR10.63.gff3.gz
	$(PY) build/expr_build.py \
	    --counts "$(EXPR)/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz" \
	    --psam "$(ARA)/arabidopsis_1001g_final.psam" \
	    --acc-meta "$(METH)/meta/accessions_1001g.csv" \
	    --gff3 "$(ARA)/annotation/Arabidopsis_thaliana.TAIR10.63.gff3.gz" \
	    --meth-accessions "$(METH)/meta/benchmark_accessions.tsv" \
	    --out-dir "$(EXPR)"
	$(PY) build/expr_family_split.py \
	    --h5 "$(EXPR)/expression_dataset.h5" \
	    --homfam "$(EXPR)/genefamily_data.HOMFAM.csv.gz"

expr-t1:                          # kinship BLUP + cis EN + HE cis-h2 (Slurm)
	sbatch baselines/t1_sandwich.sbatch

# ══ SIEVE benchmark (Brachypodium) ════════════════════════════════════════════
ZENODO := https://zenodo.org/records/18236856/files
SIEVE_DL := peer.expression.csv peer.genes.csv peer.samples.csv \
            snps.combined.M5.filtered.renamed.vcf.gz \
            gene.npy family.npy group_for_cross_validation.npy README.md

.PHONY: sieve-download sieve-reference sieve
sieve-download:
	mkdir -p "$(SIEVE)"
	for f in $(SIEVE_DL); do wget -c -O "$(SIEVE)/$$f" "$(ZENODO)/$$f?download=1"; done

sieve-reference:                  # Phytozome is login-gated; one-time manual step
	@test -s "$(SIEVE)/reference/BdistachyonBd21_3_460_v1.0.fa.gz" \
	  && test -s "$(SIEVE)/reference/BdistachyonBd21_3_460_v1.1.gene.gff3.gz" \
	  && echo "reference present" || { \
	  echo "Log in at https://phytozome-next.jgi.doe.gov (free), download the"; \
	  echo "Bd21-3 v1.1 assembly FASTA, gene GFF3, and annotation_info.txt via"; \
	  echo "the JGI curl command, then unzip into $(SIEVE)/reference/"; exit 1; }

sieve: sieve-reference
	$(PY) build/sieve_build.py \
	    --data-dir "$(SIEVE)" \
	    --out-dir "$(SIEVE)/dataset" \
	    --gff3 "$(SIEVE)/reference/BdistachyonBd21_3_460_v1.1.gene.gff3.gz"
	$(PY) build/sieve_family_split.py \
	    --h5 "$(SIEVE)/dataset/sieve_dataset.h5" \
	    --annotation-info "$(SIEVE)/reference/BdistachyonBd21_3_460_v1.1.P14.annotation_info.txt.gz"

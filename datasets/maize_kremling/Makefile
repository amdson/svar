# datasets/maize_kremling/Makefile — Zea mays, Kremling 2018 eQTL panel (AGPv3).
#
# Builds the Kremling et al. 2018 (Nature 555:520) expression panel: 3' RNA-seq
# FPMs for 7 tissues x ~300 Goodman-panel inbreds, plus the matching HapMap3.2.1
# genotypes ALREADY subset to the RNA-seq lines (MAF>0.05, KNN-imputed, TASSEL
# hapmap text, AGPv3). Everything stays on AGPv3 — the assembly both the counts
# (AGPv3.29 gene models) and hmp321 were made against — so there is no CrossMap
# and no gene-ID remap. See SOURCES.md for provenance and caveats.
#
#     conda activate svar   # tools: plink2, bcftools/bgzip, samtools
#     make            # download + build → $(STEM).pgen/.pvar/.psam (REF-aligned)
#     make check      # verify REF alleles against the AGPv3 genome
#     make dataset    # expression HDF5 (scripts/build_kremling_expression.py)
#
# Output goes to $(DATA) = $(DATA_ROOT)/maize_kremling (scratch; common.mk).

DATASET := maize_kremling
include ../common.mk

STEM   := $(DATA)/kremling_agpv3
CHROMS := 1 2 3 4 5 6 7 8 9 10

BASE := https://data.cyverse.org/dav-anon/iplant/projects/commons_repo/curated/Kremling_Nature3RNASeq282_March2018

# ── Reference genome + annotation — Ensembl Plants release-29, AGPv3 ──────────
FASTA_GZ := $(DATA)/Zea_mays.AGPv3.29.dna_sm.genome.fa.gz
FASTA    := $(DATA)/Zea_mays.AGPv3.29.dna_sm.genome.fa
GFF_GZ   := $(DATA)/Zea_mays.AGPv3.29.gff3.gz
URL_FA   := https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-29/fasta/zea_mays/dna/Zea_mays.AGPv3.29.dna_sm.genome.fa.gz
URL_GFF  := https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-29/gff3/zea_mays/Zea_mays.AGPv3.29.gff3.gz

# ── Genotypes — hmp321 RNAset subset, one tarball per chromosome ──────────────
HMP_TARS := $(foreach c,$(CHROMS),$(DATA)/merged_flt_c$(c).hmp321.onlyRNAset_MAFover005.KNNi.hmp.txt.tar.gz)

# ── Expression matrices ───────────────────────────────────────────────────────
EXPR_FPM := $(DATA)/df_STAR_HTSeq_counts_B73_match_based_on_genet_dist_DESeq2_normed_fpm_rounded.txt
EXPR_ALT := $(DATA)/df_STAR_HTSeq_counts_B73_match_based_on_genet_dist_DESeq2_normed_fpm_rounded_origNames_and_Altnames.txt

VCF_ALL  := $(DATA)/kremling_hmp321_rnaset.vcf.gz

all: $(STEM).pgen

# ── Downloads ─────────────────────────────────────────────────────────────────
$(DATA)/merged_flt_c%.hmp321.onlyRNAset_MAFover005.KNNi.hmp.txt.tar.gz: | $(DATA)
	$(WGET) "$(BASE)/RNAset_genos/$(notdir $@)"

$(EXPR_FPM): | $(DATA)
	$(WGET) "$(BASE)/Expression_matrix/$(notdir $@)"

$(EXPR_ALT): | $(DATA)
	$(WGET) "$(BASE)/Expression_matrix/$(notdir $@)"

$(FASTA_GZ): | $(DATA)
	$(WGET) "$(URL_FA)"

$(GFF_GZ): | $(DATA)
	$(WGET) "$(URL_GFF)"

$(FASTA): $(FASTA_GZ)
	gunzip -k "$<"
	$(SAMTOOLS) faidx "$@"

# ── hapmap text → single bgzipped VCF (provisional REF = hapmap allele1) ──────
# Streams each chromosome's tarball; het IUPAC codes become 0/1. REF/ALT are
# fixed against the genome by plink2 --ref-from-fa below.
$(VCF_ALL): $(HMP_TARS)
	$(CONDA_RUN) python ../../scripts/hmp_to_vcf.py \
	    --out "$@" $(HMP_TARS)

# ── plink2 import: force REF from the AGPv3 genome ────────────────────────────
$(STEM).pgen: $(VCF_ALL) $(FASTA)
	$(PLINK2) --vcf "$(VCF_ALL)" \
	    --fa "$(FASTA)" --ref-from-fa force \
	    --snps-only just-acgt --max-alleles 2 \
	    --set-missing-var-ids @_\# \
	    --sort-vars \
	    --make-pgen --out "$(STEM)"

# Also export a bgzipped, indexed VCF for tabix/pysam consumers.
vcf: $(STEM).vcf.gz
$(STEM).vcf.gz: $(STEM).pgen
	$(PLINK2) --pfile "$(STEM)" --export vcf bgz --out "$(STEM)"
	$(TABIX) -p vcf "$@"

# ── Checks ────────────────────────────────────────────────────────────────────
check: $(STEM).pgen $(FASTA)
	@echo "REF concordance was enforced by --ref-from-fa force; log:"
	@grep -A2 "ref-from-fa" "$(STEM).log" || true
	@echo "variants:" && wc -l < "$(STEM).pvar"
	@echo "samples:"  && grep -vc "^#" "$(STEM).psam"

# ── Expression HDF5 (needs $(EXPR_FPM), $(EXPR_ALT), $(GFF_GZ), $(STEM).psam) ─
dataset: $(EXPR_FPM) $(EXPR_ALT) $(GFF_GZ) $(STEM).pgen
	$(CONDA_RUN) python ../../scripts/build_kremling_expression.py

clean:
	rm -f "$(VCF_ALL)"
distclean:
	rm -rf "$(DATA)"

.PHONY: all check dataset vcf clean distclean

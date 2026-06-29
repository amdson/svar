# datasets/rice/Makefile — Oryza sativa (rice), 44K RiceDiversity panel.
#
# Reproduces the contents of ~/rice_data: the IRGSP-1.0 reference genome plus the
# 413-accession SNP genotypes remapped from MSU6 → IRGSP-1.0 (a.k.a. MSU7)
# coordinates, biallelic+QC filtered, with REF alleles forced to match the
# genome; plus the 34-trait phenotype table and the MSU6 flanking-seq table used
# for the sanity check.
#
# Quickstart (on any server):
#     conda activate svar        # tools: plink2, CrossMap, bcftools, samtools, tabix
#     make                       # download + build → $(VCF_OUT)
#     make check                 # verify REF alleles / flanking vs the genome
#
# Output goes to $(DATA) = $(DATA_ROOT)/rice (scratch by default; see common.mk).
# Every download URL + provenance note lives in SOURCES.md.

DATASET := rice
include ../common.mk

STEM     := sativas413
PLINK_IN := $(DATA)/RiceDiversity_44K_Genotypes_PLINK

# ── Raw download targets ──────────────────────────────────────────────────────
FASTA_GZ := $(DATA)/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa.gz
FASTA    := $(DATA)/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa
CHAIN_GZ := $(DATA)/MSU6_to_IRGSP-1.0.chain.gz
CHAIN    := $(DATA)/MSU6_to_IRGSP-1.0.chain
GENO_ZIP := $(DATA)/RiceDiversity.44K.MSU6.Genotypes_PLINK.zip
PHENO    := $(DATA)/RiceDiversity_44K_Phenotypes_34traits_PLINK.txt
FLANKING := $(DATA)/RiceDiversity.44K.MSU6.SNP_flanking_seq.txt

# Download URLs (kept here + in SOURCES.md; SOURCES.md carries the provenance).
URL_FASTA    := https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa.gz
URL_CHAIN    := https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/assembly_chain/oryza_sativa/MSU6_to_IRGSP-1.0.chain.gz
URL_GENO     := http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity.44K.MSU6.Genotypes_PLINK.zip
URL_PHENO    := http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity_44K_Phenotypes_34traits_PLINK.txt
URL_FLANKING := http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity.44K.MSU6.SNP_flanking_seq.txt

# ── Build intermediates / outputs ─────────────────────────────────────────────
PED      := $(PLINK_IN)/$(STEM).ped
MAP      := $(PLINK_IN)/$(STEM).map
VCF_MSU6 := $(PLINK_IN)/$(STEM).vcf
VCF_MSU7 := $(DATA)/$(STEM)_msu7.vcf
VCF_FILT := $(DATA)/$(STEM)_msu7_biallelic.vcf
VCF_QC   := $(DATA)/$(STEM)_msu7_qc.vcf
VCF_OUT  := $(DATA)/$(STEM)_msu7_final.vcf
PGEN     := $(DATA)/$(STEM)_msu7_final.pgen

REPO := $(abspath ../..)

.PHONY: all download build check test-consensus clean distclean
all: build

# ── Download ──────────────────────────────────────────────────────────────────
download: $(FASTA) $(FASTA).fai $(CHAIN) $(PED) $(PHENO) $(FLANKING)

$(FASTA_GZ): | $(DATA)
	$(WGET) "$(URL_FASTA)"
$(FASTA): $(FASTA_GZ)
	gunzip -kf "$<"
$(FASTA).fai: $(FASTA)
	$(SAMTOOLS) faidx "$<"

$(CHAIN_GZ): | $(DATA)
	$(WGET) "$(URL_CHAIN)"
$(CHAIN): $(CHAIN_GZ)
	gunzip -kf "$<"

$(GENO_ZIP): | $(DATA)
	$(WGET) "$(URL_GENO)"
# The zip unpacks to a folder with dots in the name; rename to underscores so the
# downstream paths (and crop_embed) find sativas413.ped / .map.
$(PED) $(MAP) &: $(GENO_ZIP)
	unzip -q -o "$<" -d "$(DATA)"
	@if [ -d "$(DATA)/RiceDiversity.44K.MSU6.Genotypes_PLINK" ] && [ ! -d "$(PLINK_IN)" ]; then \
		mv "$(DATA)/RiceDiversity.44K.MSU6.Genotypes_PLINK" "$(PLINK_IN)"; \
	fi

$(PHENO): | $(DATA)
	$(WGET) "$(URL_PHENO)"
$(FLANKING): | $(DATA)
	$(WGET) "$(URL_FLANKING)"

# ── Build ─────────────────────────────────────────────────────────────────────
build: $(VCF_OUT) $(PGEN)

# PLINK ped/map (MSU6 coords) → VCF
$(VCF_MSU6): $(PED) $(MAP)
	$(PLINK2) --ped $(PED) --map $(MAP) --export vcf --out $(PLINK_IN)/$(STEM)

# Remap MSU6 → IRGSP-1.0 (MSU7) with CrossMap (writes $@.unmap alongside)
$(VCF_MSU7): $(VCF_MSU6) $(CHAIN) $(FASTA)
	$(CROSSMAP) vcf $(CHAIN) $< $(FASTA) $@

# Keep only biallelic sites, then position-sort
$(VCF_FILT): $(VCF_MSU7)
	$(BCFTOOLS) view --min-alleles 2 --max-alleles 2 $< -o $@.tmp
	$(BCFTOOLS) sort $@.tmp -o $@
	rm $@.tmp

# Sample/variant QC: call-rate + MAF
$(VCF_QC): $(VCF_FILT)
	$(PLINK2) --vcf $< --mind 0.1 --geno 0.2 --maf 0.05 \
	          --export vcf --out $(DATA)/$(STEM)_msu7_qc

# Force REF allele to match the reference genome — the key correctness step.
$(PGEN) $(VCF_OUT) &: $(VCF_QC) $(FASTA)
	$(PLINK2) --vcf $< --fa $(FASTA) --ref-from-fa force \
	          --make-pgen --export vcf --out $(DATA)/$(STEM)_msu7_final

# ── Sanity checks ─────────────────────────────────────────────────────────────
# REF allele + 16bp flanking sequence agree with the reference genome.
check: $(VCF_OUT) $(FASTA) $(FLANKING)
	$(CONDA_RUN) python $(REPO)/scripts/test_dataset_flanking.py \
	    --vcf-path $(VCF_OUT) --fasta-path $(FASTA) --flanking-path $(FLANKING) \
	    --half-window 500 --buffer 0

# Spot-check: bcftools consensus genomes for a few samples (manual inspection).
TEST_SAMPLES := 081215-A05_1 081215-A06_3 081215-A07_4
TEST_DIR     := $(DATA)/test_consensus
VCF_BGZ      := $(VCF_OUT).gz
test-consensus: $(VCF_BGZ) $(FASTA)
	mkdir -p $(TEST_DIR)
	for s in $(TEST_SAMPLES); do \
	    $(BCFTOOLS) consensus -f $(FASTA) -s $$s -H 1 $(VCF_BGZ) > $(TEST_DIR)/$$s.fa; \
	done
$(VCF_BGZ): $(VCF_OUT)
	$(BCFTOOLS) view $< -O z -o $@
	$(TABIX) -p vcf $@

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:        # remove build intermediates; keep downloads + final outputs
	rm -f $(VCF_MSU6) $(VCF_MSU7) $(VCF_FILT) $(VCF_QC) $(DATA)/$(STEM)_msu7.unmap \
	      $(VCF_BGZ) $(VCF_BGZ).tbi
	rm -rf $(TEST_DIR)

distclean:    # remove EVERYTHING for rice (it's all re-downloadable)
	rm -rf $(DATA)

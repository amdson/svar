# datasets/soy/Makefile — Glycine max (soybean), Wm82.a2.v1.
#
# Builds a filtered, REF-aligned VCF + reference genome in the same fashion as
# rice, using the SoySNP50K diversity panel as the genotype source:
#   USDA SoySNP50K (Song & Hyten 2015) — 20,087 accessions × 42,509 SNPs, already
#   a multi-sample VCF on Wm82.a2 with REF/ALT, so no ped/map and no CrossMap are
#   needed (unlike rice). We just rename chroms → filter → QC → force REF → check.
#
# The VCF CHROM is renamed Gm01..Gm20 → 1..20 (scaffolds dropped) to match the
# Ensembl genome's bare 1..20 headers, so the outputs load directly into
# crop_embed's UniqueWindowDataset, which requires integer chromosome names
# (load_snps_from_vcf._parse_chrom + digit-only FASTA keys) — like the rice build.
#
# This is NOT the paper's exact soybean14460 panel (that needs a column→SNP map
# that isn't published — see SOURCES.md, "paper panel: parked"); it's a usable
# soy dataset built the rice way.
#
#     conda activate svar        # tools: plink2, bcftools  (samtools not required)
#     make            # download + build → $(VCF_OUT)
#     make check      # verify REF alleles against the genome
#
# Output goes to $(DATA) = $(DATA_ROOT)/soy (scratch by default; see common.mk).

DATASET := soy
include ../common.mk

STEM   := soysnp50k_a2
CHROMS := $(shell seq -s, 1 20)        # 1,2,...,20 — soybean has 20 chromosomes

# ── Reference genome — Ensembl Plants release-63, Glycine_max_v2.1, soft-masked ─
# Headers are bare 1..20 (verified: chrom 1 length 56,831,624 == SoySNP50K Gm01),
# so they already match the renamed VCF AND are the digit keys UniqueWindowDataset
# needs — no FASTA rename. Same assembly as the SoySNP50K positions (Wm82.a2.v1);
# `make check` confirms REF concordance.
# (Why not SoyBase's genome: its Wm82.gnm2.DTC4 *softmasked* file is truncated on
#  the server — only 9 of 20 chromosomes. genome_main would work but is Gm-named
#  and needs a rename. Ensembl is complete + already 1..20.)
ASSEMBLY   := Glycine_max.Glycine_max_v2.1
FASTA_GZ   := $(DATA)/$(ASSEMBLY).dna_sm.toplevel.fa.gz
FASTA      := $(DATA)/$(ASSEMBLY).dna_sm.toplevel.fa
URL_FASTA  := https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/glycine_max/dna/$(ASSEMBLY).dna_sm.toplevel.fa.gz

# ── Genotypes — SoySNP50K VCF (Song & Hyten 2015), Wm82.gnm2 ───────────────────
DIV_KEY := Wm82.gnm2.div.Song_Hyten_2015
VCF_SRC := $(DATA)/glyma.$(DIV_KEY).vcf.gz
URL_VCF := https://data.soybase.org/Glycine/max/diversity/$(DIV_KEY)/glyma.$(DIV_KEY).vcf.gz
CHR_MAP := $(DATA)/chrom_map.tsv        # "glyma.Wm82.gnm2.Gm01<TAB>1" ... per line

# ── Phenotypes — SoyDNGP data.csv, keyed by USDA accession id ──────────────────
# The 11 traits (protein, oil, Linoleic, Linolenic, R1, R8, Hgt, Ldg, SQ, SdWgt,
# Yield) come from the SoyDNGP project repo — the source GP-WAITER's anonymised
# "soybean14460" panel was built from. `make pheno` downloads it and joins it onto
# our VCF sample list BY ACCESSION ID, so the ~14,460 complete-phenotype accessions
# land on our own PI-keyed, coordinate-carrying genotypes. See SOURCES.md.
PHENO_SRC      := $(DATA)/soydngp_data_source.csv
URL_PHENO_SRC  := https://raw.githubusercontent.com/IndigoFloyd/SoybeanWebsite/main/data.csv
PHENO_ALIGNED  := $(DATA)/soy_pheno_aligned.csv
PHENO_COMPLETE := $(DATA)/soy_pheno_complete.csv
PSAM           := $(DATA)/$(STEM)_final.psam    # co-produced with $(PGEN) below

# ── QC + pruning options (override on the make command line) ──────────────────
# Defaults follow the GP-WAITER paper's QC (PLINK), done here with plink2:
#   MAF < 0.01, variant missing rate > 0.2, and multi-allelic sites filtered out
#   (the multi-allelic drop is the bcftools biallelic step below).
# Note: the paper also imputed missing genotypes with Beagle 5.4 — intentionally
# SKIPPED here (not installed); the final VCF therefore still carries missing
# calls, which the on-the-fly variant-application path tolerates.
MAF  ?= 0.01
GENO ?= 0.2
MIND ?=                       # sample call-rate filter; empty = skip (paper filtered variants only)
MIND_FLAG := $(if $(MIND),--mind $(MIND),)

# Optional LD pruning (the paper applied it only to its soybean1861 set), via
# plink2 --indep-pairwise <window> <step> <r^2>. Off by default; enable with:
#   make LD_PRUNE=1            (or tune LD_PARAMS=...)
LD_PRUNE  ?= 0
LD_PARAMS ?= 50 5 0.4

# ── Build intermediates / outputs (mirrors rice) ──────────────────────────────
VCF_NUM    := $(DATA)/$(STEM)_numchr.vcf      # chroms renamed to 1..20, scaffolds dropped
VCF_FILT   := $(DATA)/$(STEM)_biallelic.vcf
VCF_QC     := $(DATA)/$(STEM)_qc.vcf
VCF_PRUNED := $(DATA)/$(STEM)_pruned.vcf
VCF_OUT    := $(DATA)/$(STEM)_final.vcf
PGEN       := $(DATA)/$(STEM)_final.pgen

# What feeds the final REF-from-fa step: the pruned VCF iff LD_PRUNE=1, else QC.
ifeq ($(LD_PRUNE),1)
FINAL_IN := $(VCF_PRUNED)
else
FINAL_IN := $(VCF_QC)
endif

REPO := $(abspath ../..)

.PHONY: all download build pheno check clean distclean
all: build pheno

# ── Download + prep references ────────────────────────────────────────────────
download: $(FASTA) $(FASTA).fai $(VCF_SRC) $(PHENO_SRC)

$(FASTA_GZ): | $(DATA)
	$(WGET) "$(URL_FASTA)"
$(FASTA): $(FASTA_GZ)
	gunzip -kf "$<"
# Index with pyfaidx (samtools isn't in the env; bcftools has no `faidx`). pyfaidx
# is what UniqueWindowDataset uses anyway, and writes a standard .fai.
$(FASTA).fai: $(FASTA)
	$(CONDA_RUN) python -c "import sys,pyfaidx; pyfaidx.Faidx(sys.argv[1])" "$<"

$(VCF_SRC): | $(DATA)
	$(WGET) -O "$(VCF_SRC)" "$(URL_VCF)"

# SoyDNGP phenotype table (small; from GitHub). -O to a stable, provenance-clear name.
$(PHENO_SRC): | $(DATA)
	$(WGET) -O "$(PHENO_SRC)" "$(URL_PHENO_SRC)"

# ── Build ─────────────────────────────────────────────────────────────────────
build: $(VCF_OUT) $(PGEN)

# Rename CHROM Gm01..Gm20 → 1..20 and keep only those 20 (drop scaffolds).
$(CHR_MAP): | $(DATA)
	rm -f "$@"; for n in $$(seq 1 20); do printf 'glyma.Wm82.gnm2.Gm%02d\t%d\n' $$n $$n >> "$@"; done
# Two file-based steps (no pipe): a binary stream piped BETWEEN two separate
# `conda run` processes gets corrupted ("unknown file type").
$(VCF_NUM): $(VCF_SRC) $(CHR_MAP)
	$(BCFTOOLS) annotate --rename-chrs $(CHR_MAP) $(VCF_SRC) -Oz -o $@.renamed.vcf.gz
	$(BCFTOOLS) view -t $(CHROMS) $@.renamed.vcf.gz -o $@
	rm -f $@.renamed.vcf.gz

# Keep only biallelic sites, then position-sort.
$(VCF_FILT): $(VCF_NUM)
	$(BCFTOOLS) view --min-alleles 2 --max-alleles 2 $< -o $@.tmp
	$(BCFTOOLS) sort $@.tmp -o $@
	rm $@.tmp

# Variant QC: missing rate + MAF (paper values; +optional --mind). Multi-allelic
# already dropped by the biallelic step above.
$(VCF_QC): $(VCF_FILT)
	$(PLINK2) --vcf $< $(MIND_FLAG) --geno $(GENO) --maf $(MAF) \
	          --export vcf --out $(DATA)/$(STEM)_qc

# Optional LD pruning (only used when LD_PRUNE=1): compute a pruned SNP set, then
# keep it. plink2 prunes by variant ID, which the SoySNP50K VCF provides (ss…).
$(VCF_PRUNED): $(VCF_QC)
	$(PLINK2) --vcf $< --indep-pairwise $(LD_PARAMS) --out $(DATA)/$(STEM)_ldprune
	$(PLINK2) --vcf $< --extract $(DATA)/$(STEM)_ldprune.prune.in \
	          --export vcf --out $(DATA)/$(STEM)_pruned

# Force REF allele to match the reference genome — the key correctness step.
# Feeds from the QC VCF, or the pruned VCF when LD_PRUNE=1 (see FINAL_IN).
$(PGEN) $(VCF_OUT) &: $(FINAL_IN) $(FASTA) $(FASTA).fai
	$(PLINK2) --vcf $< --fa $(FASTA) --ref-from-fa force \
	          --make-pgen --export vcf --out $(DATA)/$(STEM)_final

# ── Phenotypes ────────────────────────────────────────────────────────────────
# Join the SoyDNGP phenotype table onto our VCF samples by accession id. Depends
# on $(PGEN) because the join needs $(PSAM) (VCF sample order), which is produced
# together with the .pgen by the grouped build rule above.
pheno: $(PHENO_ALIGNED)
$(PHENO_ALIGNED) $(PHENO_COMPLETE) &: $(PHENO_SRC) $(PGEN)
	$(CONDA_RUN) python $(REPO)/scripts/build_soy_phenotypes.py \
	    --data-csv $(PHENO_SRC) --psam $(PSAM) --out-dir $(DATA)

# ── Sanity check ──────────────────────────────────────────────────────────────
# Confirm every VCF REF allele agrees with the reference genome.
check: $(VCF_OUT) $(FASTA)
	$(BCFTOOLS) norm --check-ref w -f $(FASTA) $(VCF_OUT) -o /dev/null

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:        # remove build intermediates; keep downloads + final outputs
	rm -f $(VCF_NUM) $(VCF_FILT) $(VCF_QC) $(VCF_PRUNED) \
	      $(DATA)/$(STEM)_ldprune.prune.in $(DATA)/$(STEM)_ldprune.prune.out
distclean:    # remove EVERYTHING for soy (all re-downloadable)
	rm -rf $(DATA)

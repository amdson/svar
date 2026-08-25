# datasets/arabidopsis/Makefile — Arabidopsis thaliana, 1001 Genomes (TAIR10).
#
# Builds a filtered, REF-aligned VCF + reference genome the same way as soy, using
# the 1001 Genomes SNP/short-indel release as the genotype source:
#   1001 Genomes v3.1 (1,135 accessions) — a multi-sample VCF ALREADY on TAIR10
#   with REF/ALT, so (like soy, unlike rice) there is no ped/map and no CrossMap.
#   And unlike soy there is not even a chrom-rename: the VCF CHROM is already the
#   bare 1..5 that both the Ensembl TAIR10 genome and crop_embed's loader want.
#
# The source carries SNPs AND short indels; the build keeps only biallelic SNPs
# (bcftools view -v snps -m2 -M2) so the outputs load into crop_embed's
# UniqueWindowDataset, which applies point substitutions onto fixed-width windows.
#
#     conda activate svar        # tools: plink2, bcftools  (samtools not required)
#     make            # download + build → $(VCF_OUT)   (NB: source VCF is ~18 GB)
#     make check      # verify REF alleles against the genome
#     make phenotypes # (optional, independent) pull ALL AraPheno phenotype tables
#
# Output goes to $(DATA) = $(DATA_ROOT)/arabidopsis (scratch by default; common.mk).
# Every download URL + provenance note lives in SOURCES.md.

DATASET := arabidopsis
include ../common.mk

STEM   := arabidopsis_1001g
CHROMS := 1,2,3,4,5        # A. thaliana has 5 nuclear chromosomes (drop Mt/Pt)

# ── Reference genome — Ensembl Plants release-63, TAIR10, soft-masked ──────────
# Headers are bare 1..5 (+ Mt, Pt) — already match the 1001G VCF CHROM AND are the
# digit keys UniqueWindowDataset needs, so no FASTA rename. TAIR10 is the same
# assembly the 1001G variants were called against; `make check` confirms REF
# concordance. Mt/Pt are present in the FASTA but the VCF is restricted to 1..5.
ASSEMBLY   := Arabidopsis_thaliana.TAIR10
FASTA_GZ   := $(DATA)/$(ASSEMBLY).dna_sm.toplevel.fa.gz
FASTA      := $(DATA)/$(ASSEMBLY).dna_sm.toplevel.fa
URL_FASTA  := https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/arabidopsis_thaliana/dna/$(ASSEMBLY).dna_sm.toplevel.fa.gz

# ── Genotypes — 1001 Genomes v3.1 SNP/short-indel VCF (TAIR10) ─────────────────
# ~18 GB. CHROM is bare 1..5. Companion .vcfidx / .md5 live in the same dir; we
# don't need the index (the biallelic-SNP step below streams with -t, no index).
VCF_SRC := $(DATA)/1001genomes_snp-short-indel_only_ACGTN.vcf.gz
URL_VCF := https://1001genomes.org/data/GMI-MPI/releases/v3.1/1001genomes_snp-short-indel_only_ACGTN.vcf.gz

# ── QC options (override on the make command line) ────────────────────────────
# Same knobs as soy: MAF, variant missing rate, optional sample call-rate filter.
# A. thaliana accessions are inbred (near-fully homozygous), so these behave as
# for any biallelic SNP panel.
MAF  ?= 0.01
GENO ?= 0.2
MIND ?=                       # sample call-rate filter; empty = skip (filter variants only)
MIND_FLAG := $(if $(MIND),--mind $(MIND),)

# Optional LD pruning (off by default), via plink2 --indep-pairwise <win> <step> <r^2>.
LD_PRUNE  ?= 0
LD_PARAMS ?= 50 5 0.4

# ── Build intermediates / outputs (mirrors soy, minus the chrom-rename) ────────
# Every VCF here is bgzipped. Uncompressed, the biallelic-SNP intermediate alone
# is ~50 GB (≈11 M SNPs × 1,135 samples of "0/0<tab>" text) and the final export
# ~16 GB; bgzip cuts each roughly 10x, which is the difference between a build
# that fits comfortably in scratch and one that churns 100+ GB. Both consumers
# read it natively: plink2 takes a bgzipped --vcf, and crop_embed's loader is
# pysam-based ("plain or bgzipped", crop_embed/dataset.py).
VCF_FILT   := $(DATA)/$(STEM)_biallelic_snps.vcf.gz   # chroms 1..5, biallelic SNPs only
VCF_QC     := $(DATA)/$(STEM)_qc.vcf.gz
VCF_PRUNED := $(DATA)/$(STEM)_pruned.vcf.gz
VCF_OUT    := $(DATA)/$(STEM)_final.vcf.gz
VCF_OUT_TBI := $(VCF_OUT).tbi                         # random access for pysam
PGEN       := $(DATA)/$(STEM)_final.pgen

# Threads for bgzip compression in bcftools (the build's main CPU cost).
THREADS ?= 4

# What feeds the final REF-from-fa step: the pruned VCF iff LD_PRUNE=1, else QC.
ifeq ($(LD_PRUNE),1)
FINAL_IN := $(VCF_PRUNED)
else
FINAL_IN := $(VCF_QC)
endif

# ── Phenotypes — AraPheno REST ────────────────────────────────────────────────
# `make phenotypes` downloads the raw tables; `make pheno` aligns them onto the
# VCF sample set (needs $(PSAM), so it depends on the genotype build).
PHENO_DIR   := $(DATA)/phenotypes
PHENO_LIST  := $(PHENO_DIR)/phenotype_list.csv
# Touched when every values.csv is in. NB: keep the comment on its own line —
# `VAR := value   # comment` keeps the whitespace BEFORE the comment in the value,
# and this one is used quoted (`touch "$(PHENO_STAMP)"`), so a trailing-comment
# version creates a file literally named `.download_complete   ` while the Make
# target is the stripped `.download_complete`. The stamp then never matches its
# target and `make phenotypes` re-fetches all ~536 REST tables on every run.
PHENO_STAMP := $(PHENO_DIR)/.download_complete
URL_PHENO_LIST := https://arapheno.1001genomes.org/rest/phenotype/list.csv
# Per-phenotype values are pulled by looping the IDs in phenotype_list.csv:
#   https://arapheno.1001genomes.org/rest/phenotype/<id>/values.csv
PSAM         := $(DATA)/$(STEM)_final.psam       # co-produced with $(PGEN)
PHENO_LONG   := $(DATA)/arabidopsis_pheno_long.csv
PHENO_MATRIX := $(DATA)/arabidopsis_pheno_matrix.csv
PHENO_COVER  := $(DATA)/arabidopsis_pheno_coverage.csv

REPO := $(abspath ../..)

.PHONY: all download build check phenotypes pheno clean distclean
all: build pheno

# ── Download + prep references ────────────────────────────────────────────────
download: $(FASTA) $(FASTA).fai $(VCF_SRC)

$(FASTA_GZ): | $(DATA)
	$(WGET) "$(URL_FASTA)"
$(FASTA): $(FASTA_GZ)
	gunzip -kf "$<"
# Index with pyfaidx (samtools isn't in the env; bcftools has no `faidx`). pyfaidx
# is what UniqueWindowDataset uses anyway, and writes a standard .fai.
$(FASTA).fai: $(FASTA)
	$(CONDA_RUN) python -c "import sys,pyfaidx; pyfaidx.Faidx(sys.argv[1])" "$<"

# NB: no -O. The URL basename already equals $(VCF_SRC)'s, and `wget -c -O file`
# has undefined resume behaviour — on a 19 GB download that silently restarts
# from zero (or truncates) after a dropped connection. Plain `-c -P $(DATA)`
# resumes correctly, which is the whole point on a file this size.
$(VCF_SRC): | $(DATA)
	$(WGET) "$(URL_VCF)"

# ── Build ─────────────────────────────────────────────────────────────────────
build: $(VCF_OUT) $(VCF_OUT_TBI) $(PGEN)

# Keep chroms 1..5, biallelic SNPs only (drop indels + multi-allelic). One
# combined bcftools view streams the ~18 GB source once; -t needs no index.
#
# NB: we deliberately do NOT `bcftools sort` here. The 1001G VCF header defines no
# ##contig lines, so bcftools sort — which spills temp BCF, and BCF *requires*
# contigs in the header — dies with "[E::bcf_write] Unchecked error (1)". The
# release is already coordinate-sorted anyway, and plink2 re-orders variants in
# the QC step below via --sort-vars (which needs no contig headers), so ordering
# is still guaranteed downstream.
$(VCF_FILT): $(VCF_SRC)
	$(BCFTOOLS) view -t $(CHROMS) --types snps --min-alleles 2 --max-alleles 2 \
	          -Oz --threads $(THREADS) $(VCF_SRC) -o $@

# Variant QC: missing rate + MAF (+optional --mind). Multi-allelic / indels are
# already gone from the filter step. Order is preserved from the (already
# coordinate-sorted) source; the final step re-asserts it with --sort-vars, which
# plink2 only permits alongside --make-pgen (not with a bare --export vcf).
$(VCF_QC): $(VCF_FILT)
	$(PLINK2) --vcf $< $(MIND_FLAG) --geno $(GENO) --maf $(MAF) \
	          --export vcf bgz --out $(DATA)/$(STEM)_qc

# Optional LD pruning (only when LD_PRUNE=1). plink2 prunes by variant ID; if the
# source lacks IDs, plink2 assigns chr:pos-style IDs which --extract still matches.
$(VCF_PRUNED): $(VCF_QC)
	$(PLINK2) --vcf $< --set-all-var-ids '@:#' --indep-pairwise $(LD_PARAMS) \
	          --out $(DATA)/$(STEM)_ldprune
	$(PLINK2) --vcf $< --set-all-var-ids '@:#' --extract $(DATA)/$(STEM)_ldprune.prune.in \
	          --export vcf bgz --out $(DATA)/$(STEM)_pruned

# Force REF allele to match the reference genome — the key correctness step.
# Feeds from the QC VCF, or the pruned VCF when LD_PRUNE=1 (see FINAL_IN).
# Two plink2 calls: (1) --ref-from-fa + --sort-vars → sorted, REF-aligned pgen;
# (2) export that pgen to VCF. They're split because plink2 forbids --sort-vars in
# any command that also has --export vcf (even with --make-pgen present) — and
# --sort-vars is our stand-in for the bcftools sort the 1001G header won't allow.
$(PGEN) $(VCF_OUT) &: $(FINAL_IN) $(FASTA) $(FASTA).fai
	$(PLINK2) --vcf $< --fa $(FASTA) --ref-from-fa force --sort-vars \
	          --make-pgen --out $(DATA)/$(STEM)_final
	$(PLINK2) --pfile $(DATA)/$(STEM)_final --export vcf bgz \
	          --out $(DATA)/$(STEM)_final

# Tabix index over the final VCF — pysam (crop_embed's loader) needs it to fetch
# a region instead of streaming the whole file. bcftools, not tabix: the svar env
# ships bcftools but no standalone tabix binary.
$(VCF_OUT_TBI): $(VCF_OUT)
	$(BCFTOOLS) index -t --threads $(THREADS) $(VCF_OUT)

# ── Sanity check ──────────────────────────────────────────────────────────────
# Confirm every VCF REF allele agrees with the reference genome. (No per-SNP
# flanking table exists for 1001G, so this is the soy-style check, not rice's.)
check: $(VCF_OUT) $(FASTA)
	$(BCFTOOLS) norm --check-ref w -f $(FASTA) $(VCF_OUT) -o /dev/null

# ── Phenotypes ────────────────────────────────────────────────────────────────
# Two stages, mirroring soy: (1) `phenotypes` downloads the raw AraPheno tables;
# (2) `pheno` aligns them onto our VCF sample set by accession id.

# (1) Download the phenotype index, then every phenotype's values.csv (keyed by
# accession). A stamp file records completion so `pheno` can depend on it.
phenotypes: $(PHENO_STAMP)
$(PHENO_STAMP): $(PHENO_LIST)
	@echo "Downloading all AraPheno phenotype value tables into $(PHENO_DIR) ..."
	@tail -n +2 "$(PHENO_LIST)" | cut -d, -f2 | while read id; do \
	    case "$$id" in ''|*[!0-9]*) continue;; esac; \
	    wget -c -q -O "$(PHENO_DIR)/phenotype_$$id.csv" \
	        "https://arapheno.1001genomes.org/rest/phenotype/$$id/values.csv" \
	        && echo "  phenotype $$id" || echo "  phenotype $$id FAILED"; \
	done
	@touch "$(PHENO_STAMP)"
$(PHENO_LIST): | $(PHENO_DIR)
	wget -c -O "$(PHENO_LIST)" "$(URL_PHENO_LIST)"
$(PHENO_DIR):
	mkdir -p "$(PHENO_DIR)"

# (2) Align onto the VCF samples (exact accession-id match). Depends on $(PSAM),
# so `make pheno` builds the genotypes first if needed. Emits a tidy long table, a
# wide sample×trait matrix (VCF order), and a coverage guide — see the script.
PHENO_SCRIPT := $(REPO)/scripts/build_arabidopsis_phenotypes.py
pheno: $(PHENO_LONG)
$(PHENO_LONG) $(PHENO_MATRIX) $(PHENO_COVER) &: $(PHENO_STAMP) $(PSAM) $(PHENO_SCRIPT)
	$(CONDA_RUN) python $(PHENO_SCRIPT) \
	    --pheno-dir $(PHENO_DIR) --psam $(PSAM) --out-dir $(DATA)

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:        # remove build intermediates; keep downloads + final outputs
	rm -f $(VCF_FILT) $(VCF_QC) $(VCF_PRUNED) \
	      $(DATA)/$(STEM)_ldprune.prune.in $(DATA)/$(STEM)_ldprune.prune.out
distclean:    # remove EVERYTHING for arabidopsis (all re-downloadable)
	rm -rf $(DATA)

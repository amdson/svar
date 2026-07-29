# datasets/wheat/Makefile — Triticum aestivum (hexaploid bread wheat).
#
# CIMMYT "Seeds of Discovery" IRANIAN bread-wheat landrace panel: DArTSeq
# genotypes + agronomic/quality phenotypes, aligned to IWGSC RefSeq v1.0.
#
# UNLIKE rice/soy, DArTSeq markers ship with NO genomic coordinates — only 69bp
# sequence tags. This build recovers coordinates by aligning the tags to the wheat
# genome (minimap2), keeping only uniquely-mapped, concordant markers, then emits a
# VCF (chroms renamed 1A..7D,Un -> 1..22 for crop_embed) + a GID-joined phenotype
# table. Genotype (SEEDDIV) <-> phenotype (GID) keys are bridged by the MARKERS
# file's header, which carries both. See SOURCES.md for full provenance.
#
# The genome auto-downloads; the DArTSeq + phenotype archives are ACCESS-GATED
# (CIMMYT) and are NOT re-downloadable — place them under $(RAW) (default
# persian_wheat/). `make` errors with a clear message if they're missing.
#
#     conda activate svar        # minimap2, samtools, bcftools, pysam, openpyxl, pandas
#     make            # build genotypes+coords+VCF, then phenotypes
#     make check      # verify VCF REF alleles against the genome

DATASET := wheat
include ../common.mk

REPO := $(abspath ../..)
RAW  ?= $(REPO)/persian_wheat

# ── Access-gated raw inputs (must be present; not auto-downloadable) ───────────
MARKERS_SRC   := $(RAW)/MEXICO AND IRANIAN -- MARKERS.zip.gz
DATAVERSE_SRC := $(RAW)/dataverse_files.zip

# ── Reference genome — Ensembl Plants r60, IWGSC RefSeq v1.0 (Chinese Spring) ──
URL_FASTA := https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/triticum_aestivum/dna/Triticum_aestivum.IWGSC.dna_sm.toplevel.fa.gz
FASTA_GZ  := $(DATA)/genome.fa.gz
FASTA_RAW := $(DATA)/genome.fa
FASTA     := $(DATA)/wheat_genome.fa

# ── Intermediates / outputs ────────────────────────────────────────────────────
MARKERS_ZIP := $(DATA)/markers.zip
TAGS        := $(DATA)/tags.fasta
BAM         := $(DATA)/aln.bam
COORDS      := $(DATA)/marker_coords.tsv
VCF         := $(DATA)/wheat_dartseq.vcf
PGEN        := $(DATA)/wheat_dartseq.pgen   # plink2 fileset for the training pipeline
PHENO_STAMP := $(DATA)/.pheno_extracted
PHENO       := $(DATA)/wheat_pheno_aligned.csv

MINIMAP2 := $(CONDA_RUN) minimap2
PYTHON   := $(CONDA_RUN) python
MIN_MAPQ  ?= 30
MIN_IDENT ?= 0.90

.PHONY: all download build pheno check clean distclean
all: build pheno

# ── Genome: download -> gunzip -> faidx (original); then rename -> faidx (integer)
download: $(FASTA_RAW) $(FASTA)
$(FASTA_GZ): | $(DATA)
	$(WGET) -O "$(FASTA_GZ)" "$(URL_FASTA)"
$(FASTA_RAW): $(FASTA_GZ)
	gunzip -kf "$(FASTA_GZ)"
	$(SAMTOOLS) faidx "$(FASTA_RAW)"
# Rename headers 1A..7D,Un -> 1..22 so the VCF/genome load through crop_embed.
$(FASTA): $(FASTA_RAW)
	awk 'BEGIN{split("1A 1B 1D 2A 2B 2D 3A 3B 3D 4A 4B 4D 5A 5B 5D 6A 6B 6D 7A 7B 7D Un",a," "); \
	     for(i=1;i<=22;i++)m[a[i]]=i} \
	     /^>/{split(substr($$0,2),x," ");print ">"m[x[1]];next}{print}' "$(FASTA_RAW)" > "$(FASTA)"
	$(SAMTOOLS) faidx "$(FASTA)"

# ── Genotypes: gunzip MARKERS.zip.gz -> markers.zip -> tags.fasta ──────────────
$(MARKERS_ZIP): | $(DATA)
	@test -f "$(MARKERS_SRC)" || { echo "MISSING access-gated input: $(MARKERS_SRC)  (see SOURCES.md)"; exit 1; }
	gunzip -c "$(MARKERS_SRC)" > "$@"
$(TAGS): $(MARKERS_ZIP)
	$(PYTHON) $(REPO)/scripts/wheat_extract_tags.py --markers-zip "$(MARKERS_ZIP)" --out-dir "$(DATA)"

# ── Align tags -> genome (whole genome one index; MAPQ = genome-wide uniqueness) ─
$(BAM): $(TAGS) $(FASTA_RAW)
	$(MINIMAP2) -t $(shell nproc) -I 20G -ax sr "$(FASTA_RAW)" "$(TAGS)" 2> "$(DATA)/mm2.log" \
	  | $(SAMTOOLS) sort -@ 8 -o "$(BAM)" -
	$(SAMTOOLS) index -c "$(BAM)"    # CSI: wheat chromosomes exceed the 512Mb BAI limit

# ── Validate + project SNPs to genome coordinates ─────────────────────────────
$(COORDS): $(BAM) $(FASTA_RAW)
	$(PYTHON) $(REPO)/scripts/wheat_validate_project.py --wheat-dir "$(DATA)" \
	    --genome "$(FASTA_RAW)" --min-mapq $(MIN_MAPQ) --min-ident $(MIN_IDENT)

# ── Build VCF (chroms -> 1..22, REF = genome allele) ──────────────────────────
build: $(VCF) $(PGEN)
$(VCF): $(COORDS) $(MARKERS_ZIP)
	$(PYTHON) $(REPO)/scripts/wheat_build_vcf.py --wheat-dir "$(DATA)" --markers-zip "$(MARKERS_ZIP)"
# plink2 fileset (.pgen/.pvar/.psam) for training/common's SNP matrix loader.
$(PGEN): $(VCF)
	$(PLINK2) --vcf "$(VCF)" --make-pgen --out "$(DATA)/wheat_dartseq"

# ── Phenotypes: unpack dataverse (.zip -> .zip.gz -> .zip -> xlsx), join by GID ─
$(PHENO_STAMP): | $(DATA)
	@test -f "$(DATAVERSE_SRC)" || { echo "MISSING access-gated input: $(DATAVERSE_SRC)  (see SOURCES.md)"; exit 1; }
	cd "$(DATA)" && unzip -o "$(DATAVERSE_SRC)" "PHENOTYPIC DATA IRANIAN.zip.gz" >/dev/null
	cd "$(DATA)" && gunzip -kf "PHENOTYPIC DATA IRANIAN.zip.gz"
	cd "$(DATA)" && unzip -o "PHENOTYPIC DATA IRANIAN.zip" >/dev/null
	touch "$@"
pheno: $(PHENO)
$(PHENO): $(VCF) $(PHENO_STAMP)
	$(PYTHON) $(REPO)/scripts/wheat_build_pheno.py --wheat-dir "$(DATA)" \
	    --pheno-dir "$(DATA)/PHENOTYPIC DATA IRANIAN"

# ── Sanity check: VCF REF alleles agree with the (renamed) genome ─────────────
check: $(VCF) $(FASTA)
	$(BCFTOOLS) norm --check-ref w -f "$(FASTA)" "$(VCF)" -o /dev/null

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:        # drop big intermediates; keep genome, VCF, coords, phenotypes
	rm -f "$(MARKERS_ZIP)" "$(DATA)/mm2.log"
distclean:    # remove EVERYTHING for wheat (genome re-downloadable; raw archives are in $(RAW))
	rm -rf "$(DATA)"

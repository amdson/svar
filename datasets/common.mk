# common.mk — shared configuration for every per-dataset Makefile here.
#
# Each dataset's Makefile sets DATASET and then `include ../common.mk`. This file
# defines where data lands and the bioinformatics tool wrappers, so the dataset
# Makefiles only have to describe URLs and the build graph.
#
# WHAT'S TRACKED: only Makefiles, *.mk, and *.md (see .gitignore). The data is
# never committed — you rebuild it on any server with `make`.
#
# WHERE DATA LANDS: under cluster scratch by default, OFF the home quota. Each
# dataset goes in $(DATA_ROOT)/$(DATASET). Override per-invocation, e.g.:
#     make DATA_ROOT=/some/other/root
#     make SVAR_SCRATCH=/different/scratch
#
# TOOLS: provided by the conda env named by CONDA_ENV (default: svar) — plink2,
# CrossMap, bcftools, samtools, tabix. Activate it (or just rely on `conda run`)
# before building:  conda activate svar

# Scratch base — keep in sync with svar/env.sh. Override if your cluster differs.
SVAR_SCRATCH ?= /90daydata/small_grains/andrew.dickson
DATA_ROOT    ?= $(SVAR_SCRATCH)/datasets

ifndef DATASET
$(error DATASET is not set — define it in the dataset Makefile before `include ../common.mk`)
endif

# Per-dataset output directory (all downloads + build artifacts go here).
DATA := $(DATA_ROOT)/$(DATASET)

# common.mk is included near the top of each dataset Makefile, so the $(DATA)
# rule below would otherwise become the default goal. Pin the default to `all`.
.DEFAULT_GOAL := all

# Conda env carrying the tools, run non-interactively so Makefiles work in batch.
CONDA_ENV ?= svar
CONDA_RUN := conda run -n $(CONDA_ENV)

PLINK2   := $(CONDA_RUN) plink2
CROSSMAP := $(CONDA_RUN) CrossMap
BCFTOOLS := $(CONDA_RUN) bcftools
SAMTOOLS := $(CONDA_RUN) samtools
TABIX    := $(CONDA_RUN) tabix

# Resumable download into $(DATA). Use as:  $(WGET) "$(URL)"
WGET := wget -c -P "$(DATA)"

# Order-only prerequisite so every recipe can depend on the dir existing: `| $(DATA)`
$(DATA):
	mkdir -p "$(DATA)"

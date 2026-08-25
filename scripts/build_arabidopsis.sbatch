#!/bin/bash
#
# scripts/build_arabidopsis.sbatch
# --------------------------------
# Rebuild the arabidopsis dataset (1001 Genomes v3.1 → TAIR10) from source on
# Atlas, so nothing has to be copied from another cluster. Runs the recipe in
# datasets/arabidopsis/Makefile; every stage is a Make target, so re-submitting
# after a wall-clock timeout resumes at the first unfinished stage.
#
# Two stages, submitted separately so the slow download isn't repeated when a
# build step needs re-running:
#     sbatch --export=STAGE=download scripts/build_arabidopsis.sbatch
#     sbatch --export=STAGE=build --dependency=afterok:<jobid> scripts/build_arabidopsis.sbatch
# STAGE defaults to `all` (download + build + check + phenotypes) when unset.
#
# The 19.2 GB source VCF is fetched with `wget -c`, so a killed download picks up
# where it left off rather than starting over.
#
#SBATCH --job-name=arab_build
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=small_grains
#SBATCH --output=logs/arab_%x_%j.out
#SBATCH --error=logs/arab_%x_%j.err

set -uo pipefail

REPO=/home/andrew.dickson/svar
cd "$REPO" || exit 1
mkdir -p logs

STAGE="${STAGE:-all}"

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_ROOT=/apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate svar

# `conda activate` only defines `conda` as a *shell function*, which Make's
# /bin/sh recipes do not inherit — every $(CONDA_RUN) line in the dataset
# Makefiles then dies with "conda: command not found". Put the real executable on
# PATH so sub-shells can find it.
export PATH="$CONDA_ROOT/bin:$PATH"

# Sets SVAR_SCRATCH (→ DATA_ROOT=$SVAR_SCRATCH/datasets, see datasets/common.mk).
source "$REPO/env.sh"

echo "===== $(date) :: arabidopsis STAGE=$STAGE START ====="
echo "Host:    $(hostname)"
echo "Scratch: $SVAR_SCRATCH"
echo "Target:  $SVAR_SCRATCH/datasets/arabidopsis"
echo "Tools:   $(bcftools --version | head -1) / $(plink2 --version)"
df -h "$SVAR_SCRATCH" | tail -1

cd "$REPO/datasets/arabidopsis" || exit 1

case "$STAGE" in
  download)    make download ;;
  # AraPheno REST pulls (~550 small CSVs) touch none of the genotype targets, so
  # this can run alongside the 19 GB download instead of after it.
  phenotypes)  make phenotypes ;;
  build)       make build && make check && make phenotypes && make pheno ;;
  all)         make download && make build && make check && make phenotypes && make pheno ;;
  *)           echo "unknown STAGE=$STAGE (want: download | phenotypes | build | all)"; exit 2 ;;
esac
rc=$?

echo "----- disk after -----"
du -sh "$SVAR_SCRATCH/datasets/arabidopsis" 2>/dev/null
echo "===== $(date) :: arabidopsis STAGE=$STAGE EXIT=$rc ====="
exit $rc

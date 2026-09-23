#!/bin/bash
#
# Stage 3 on its own: extract (mc, total) counts at the stage-2 retained sites
# into dataset.h5. Stage 2 returned MARGINAL (CG retained 371k, between the
# 50k and 500k thresholds), so the chained stage1 sbatch stopped for review;
# Andrew chose to proceed (2026-09-23). Resumable via stage3_tmp/acc*.npy.
#
#   sbatch ath_meth_benchmark/build/stage3_extract.sbatch
#
#SBATCH --job-name=ath_meth_stage3
#SBATCH --partition=atlas
#SBATCH --cpus-per-task=18
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --account=small_grains
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu/etc/profile.d/conda.sh
conda activate svar
cd "$HOME/svar"
export PYTHONUNBUFFERED=1
DATA=/90daydata/small_grains/andrew.dickson/datasets/arabidopsis
echo "Host: $(hostname) CPUs: $SLURM_CPUS_PER_TASK $(date)"

/home/andrew.dickson/.conda/envs/svar/bin/python ath_meth_benchmark/build/stage3_extract.py \
    --stage1-dir "$DATA/methylation/stage1" \
    --stage2-dir "$DATA/methylation/stage2" \
    --allc-dir "$DATA/methylation/allc" \
    --accessions "$DATA/methylation/meta/benchmark_accessions.tsv" \
    --out "$DATA/methylation/dataset.h5" \
    --workers 16
echo "DONE $(date)"

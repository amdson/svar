#!/bin/bash
#
# training_ge/run_sieve_full.sbatch
# ---------------------------------
# The full SIEVE family-holdout fine-tune: every scoreable train-family gene
# at hw=4000 (~150k pairs), val on 1,500 held-out-family genes per epoch.
# Batches are built once up front (~15 min CPU) and cached in RAM; the model
# checkpoint + JSONL metrics land next to --output under $SVAR_SCRATCH/runs/.
#
# Submit when ready (e.g. tonight):
#   sbatch training_ge/run_sieve_full.sbatch
#   sbatch --begin=22:00 training_ge/run_sieve_full.sbatch   # queue for 10pm
#
# Add hyperparameter overrides by editing RUN_ARGS below — expect iteration.
#
#SBATCH --job-name=ge_sieve_full
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=small_grains
#SBATCH --output=logs/ge_sieve_full_%j.out
#SBATCH --error=logs/ge_sieve_full_%j.err

set -euo pipefail

source /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu/etc/profile.d/conda.sh
conda activate svar

REPO="$HOME/svar"
cd "$REPO"
source "$REPO/env.sh"

RUN_ARGS=(
    --holdout family
    --n-genes 20000
    --epochs 4
    --output "$SVAR_SCRATCH/runs/ge_sieve_family_full.pt"
)

python -m training_ge.run "${RUN_ARGS[@]}"

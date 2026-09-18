#!/bin/bash
#
# training_ge/run_ge.sbatch — generic Slurm wrapper for training_ge.run.
# Every argument after the script is passed straight through, so:
#
#   sbatch -J ge_exact  training_ge/run_ge.sbatch --dataset ath --exact ... --output $SVAR_SCRATCH/runs/x.pt
#   sbatch -J ge_head   training_ge/run_ge.sbatch --dataset ath --head-only ...
#
# Console log: logs/<jobname>_<jobid>.out ; metrics sidecar + checkpoint land
# next to --output as always. Partition list lets Slurm take whichever GPU
# frees first (exact-forward at hw=4kb with --hap-chunk 16 fits a 48 GB L40S).
#
#SBATCH --partition=gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=small_grains
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu/etc/profile.d/conda.sh
conda activate svar  # PATH only; the interpreter below is pinned because the default env is no longer svar
cd "$HOME/svar"
source "$HOME/svar/env.sh"
export PYTHONUNBUFFERED=1
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) args: $*"
/home/andrew.dickson/.conda/envs/svar/bin/python -m training_ge.run "$@"

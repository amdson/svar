#!/bin/bash
#
# training_meth/run_meth.sbatch — Slurm wrapper. First arg training_meth.<module>
# picks the entry point (default extract_features); everything else passes through.
#
#   sbatch -J meth_carbon training_meth/run_meth.sbatch --backend carbon --n-sites 1000 --out $SVAR_SCRATCH/runs/meth_feat_carbon.npz
#   sbatch -J meth_fit -p atlas --gres=none training_meth/run_meth.sbatch training_meth.fit_sites --features ...
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
conda activate svar
cd "$HOME/svar"
source "$HOME/svar/env.sh"
export PYTHONUNBUFFERED=1
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none) args: $*"
MOD=training_meth.extract_features
case "${1:-}" in training_meth.*) MOD=$1; shift;; esac
/home/andrew.dickson/.conda/envs/svar/bin/python -m "$MOD" "$@"

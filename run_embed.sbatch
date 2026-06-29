#!/bin/bash
#
# run_embed.sbatch
# ----------------
# Generate ONLY the embedding caches (no head training): runs embed_all_windows.sh,
# which builds all 16 Carbon window-embedding caches —
#   {variant-cache, full-forward} × {250,500,1000,2000} nt × {mean, snp} pooling
# into checkpoints/carbon/.
#
# Resumable: each embed skips already-finished windows via its per-cache
# .embtable.ckpt.pt, so if the job hits the wall clock just resubmit and it continues.
#
# Submit with:   sbatch run_embed.sbatch
#
#SBATCH --job-name=svar_embed
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --account=small_grains            # ← or 'sandbox'; set to your project account
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

set -uo pipefail

REPO=/home/andrew.dickson/svar
cd "$REPO"
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
source /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu/etc/profile.d/conda.sh
conda activate svar

# Route model weights + caches to cluster scratch (not the home quota).
source "$REPO/env.sh"
echo "Scratch: $SVAR_SCRATCH  (HF_HOME=$HF_HOME)"

echo "Python: $(which python)"
echo "Host:   $(hostname)   GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Embeddings ────────────────────────────────────────────────────────────────
echo "===== $(date) :: embeddings START ====="
bash embed_all_windows.sh
echo "===== $(date) :: embeddings DONE ====="

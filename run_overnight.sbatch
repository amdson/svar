#!/bin/bash
#
# run_overnight.sbatch
# --------------------
# One overnight job on Atlas:
#   STEP 1  embed_all_windows.sh    — build all 16 Carbon window-embedding caches
#                                     ({variant-cache, full-forward} × {250,500,1000,2000} nt
#                                      × {mean, snp} within-window pooling)
#   STEP 2  run_head_sweep.sh       — train heads over the knob sweep on every cache
#
# Both steps are resumable: embeds skip already-finished windows via their per-cache
# .embtable.ckpt.pt, and the head sweep skips heads whose model.pt already exists. So
# if the job hits the wall clock, just resubmit and it continues where it stopped.
#
#   sbatch run_overnight.sbatch
#
#SBATCH --job-name=svar_embed_heads
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=small_grains            # ← or 'sandbox'; set to your project account
#SBATCH --output=logs/overnight_%j.out
#SBATCH --error=logs/overnight_%j.err

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

# ── Step 1: embeddings ────────────────────────────────────────────────────────
# Don't abort the whole job if a single embed fails — the head sweep skips any
# cache that didn't materialize, so we still train on everything that succeeded.
echo "===== $(date) :: STEP 1/2  embeddings ====="
bash embed_all_windows.sh || echo "WARNING: embed_all_windows.sh exited nonzero; continuing to head sweep"

# ── Step 2: head sweep ────────────────────────────────────────────────────────
echo "===== $(date) :: STEP 2/2  head sweep ====="
bash train_pipeline/run_head_sweep.sh

echo "===== $(date) :: DONE ====="

#!/bin/bash
#SBATCH --job-name=crop_embed
#SBATCH --partition=gpu                  # ← set to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                        # model weights + full chromosome cache
#SBATCH --time=12:00:00                  # checkpointing means resubmit is safe if this is too short
#SBATCH --output=logs/crop_embed_%j.out
#SBATCH --error=logs/crop_embed_%j.err
# #SBATCH --account=small_grains         # ← uncomment and set if your cluster requires it
# #SBATCH --nodelist=YOUR_GPU_NODE       # ← uncomment to pin to a specific node

set -euo pipefail

mkdir -p logs

# ── Activate environment ──────────────────────────────────────────────────────
# Adjust ENV_NAME if your torch/transformers install lives in a different env.
ENV_NAME="svar"
source /home/adickson/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "Python: $(which python)"
echo "Host:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Run ───────────────────────────────────────────────────────────────────────
# Pass --snp-only to pool only SNP-containing tokens instead of the full window.
# Each mode writes to a separate output file, but delete the matching .ckpt.pt
# before switching modes to avoid resuming from an incompatible checkpoint.
cd /home/adickson/svar
python demo_embed.py --snp-only

echo "Done."

#!/usr/bin/env bash
# pretrained_eval/sanity_check.sh — quick smoke of the AR log-likelihood eval on a
# handful of rice windows. Use to confirm a checkpoint loads + produces sane
# whole-sequence / per-SNP numbers before committing to a full ~30k-window run.
#
# Usage:
#   bash pretrained_eval/sanity_check.sh [model] [n_windows] [batch_size]
#     model      : alias (500m | 3b | 8b) or a HF repo / local dir   [default: 8b]
#     n_windows  : how many variant windows to score                 [default: 8]
#     batch_size : per-forward batch                                  [default: 4]
#
# Needs a GPU. The 8B weights are already cached on scratch (no re-download).
# Carbon-8B peaks ~21 GB; 3B ~10 GB; 500M ~3 GB. Drop batch_size if you OOM.
#
# This writes NO per-window CSV (--per-window-out none) so it never clobbers the
# real full-run outputs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/env.sh"            # routes HF cache etc. to /90daydata scratch

MODEL="${1:-8b}"
N="${2:-8}"
BATCH="${3:-4}"

echo ">> sanity check: model=$MODEL  windows=$N  batch=$BATCH"
python "$HERE/pretrained_eval/eval.py" \
    --model-path "$MODEL" \
    --limit "$N" \
    --batch-size "$BATCH" \
    --per-window-out none

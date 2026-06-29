#!/usr/bin/env bash
# Regenerate the fig-02 head-training data: the three window-aggregation diffs
# (absolute / centered / refdelta) on the variant-cache hw500 cache, matched
# otherwise (mlp / mean / warm-start standardizer / 15 epochs), then plot.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew.dickson/.conda/envs/svar/bin/python
CACHE=checkpoints/manual/sativas413_carbon500m_vc_hw500.ckpt.pt

run () {  # $1 = variant name, $2 = variant flag
  echo "=== head train: $1 ==="
  $PY train_pipeline/train_head.py \
    --cache "$CACHE" --half-window 500 --head mlp --pool mean \
    --warm-start-standardizer --epochs 15 --lr 1e-3 --weight-decay 1e-4 \
    $2 --output "trained_heads/repro/$1/model.pt"
}

run absolute ""
run centered "--center-windows"
run refdelta "--subtract-reference"

echo "=== plot fig 02 ==="
$PY presentation/plot_diff_curves.py \
  --absolute trained_heads/repro/absolute/model.metrics.jsonl \
  --centered trained_heads/repro/centered/model.metrics.jsonl \
  --refdelta trained_heads/repro/refdelta/model.metrics.jsonl
echo "=== head-diff repro done ==="

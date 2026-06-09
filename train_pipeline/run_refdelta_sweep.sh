#!/usr/bin/env bash
#
# run_refdelta_sweep.sh
# ---------------------
# Train FPRefDeltaSumHeadModel heads (reference-subtracted, see
# crop_embed.models.fp_head_model) on every cache from run_window_sweep.sh:
#   half-window ∈ {250, 500, 1000}  ×  snp-only ∈ {off, on}   = 6 caches
# and, per cache, a linear and an mlp delta-head                = 12 head runs.
#
# These reuse the EXISTING caches in checkpoints/sweep/ — the reference delta is
# computed from the cache, so no re-embedding is needed. Run run_window_sweep.sh
# first to generate the caches.
#
# The 6 caches are spread round-robin across the 3 GPUs (2 per GPU); each GPU
# trains its caches sequentially and the three lanes run concurrently.
#
#   bash train_pipeline/run_refdelta_sweep.sh
#
set -uo pipefail

cd /home/andrew/svar
PY=/home/andrew/anaconda3/envs/svar/bin/python

CACHE_DIR=checkpoints/sweep
HEAD_DIR=trained_heads/sweep_refdelta_mean
LOG_DIR=logs/sweep_refdelta
mkdir -p "$HEAD_DIR" "$LOG_DIR"

# ── Sweep knobs ───────────────────────────────────────────────────────────────
HALF_WINDOWS=(250 500 1000)
HEADS=(linear mlp)
EPOCHS=100
LR=1e-3

# ── One cache: train every head as a reference-delta model ────────────────────
# Pinned to whatever GPU the calling lane exported as CUDA_VISIBLE_DEVICES.
run_config() {
  local hw=$1 snp_only=$2
  local suffix=""
  if [[ $snp_only -eq 1 ]]; then suffix="_snponly"; fi
  local tag="hw${hw}${suffix}"
  local cache="$CACHE_DIR/sativas413_${tag}.ckpt.pt"
  local log="$LOG_DIR/${tag}.log"

  if [[ ! -f "$cache" ]]; then
    echo "[gpu ${CUDA_VISIBLE_DEVICES}] SKIP $tag — cache not found: $cache (run run_window_sweep.sh first)"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] [gpu ${CUDA_VISIBLE_DEVICES}] START $tag"
  {
    for head in "${HEADS[@]}"; do
      echo "######## $(date) :: train refdelta $head $tag ########"
      $PY train_pipeline/train_refdelta_head.py \
          --cache "$cache" --head "$head" \
          --half-window "$hw" \
          --epochs "$EPOCHS" --lr "$LR" --warm-start-standardizer \
          --output "$HEAD_DIR/${head}_${tag}/model.pt" \
        || echo "!!! TRAIN FAILED: $head $tag"
    done
  } >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] [gpu ${CUDA_VISIBLE_DEVICES}] DONE  $tag"
}

# ── Build the 6 configs and assign them round-robin to GPUs 0/1/2 ─────────────
configs=()
for hw in "${HALF_WINDOWS[@]}"; do
  for snp in 0 1; do
    configs+=("$hw $snp")
  done
done

gpu0=(); gpu1=(); gpu2=()
i=0
for cfg in "${configs[@]}"; do
  case $((i % 3)) in
    0) gpu0+=("$cfg");;
    1) gpu1+=("$cfg");;
    2) gpu2+=("$cfg");;
  esac
  i=$((i + 1))
done

lane() {
  export CUDA_VISIBLE_DEVICES=$1; shift
  for cfg in "$@"; do
    run_config $cfg   # unquoted: splits "hw snp" into two args
  done
}

echo "Refdelta sweep: ${#configs[@]} caches × ${#HEADS[@]} heads on 3 GPUs"
echo "  gpu0: ${gpu0[*]}"
echo "  gpu1: ${gpu1[*]}"
echo "  gpu2: ${gpu2[*]}"
echo "Caches: $CACHE_DIR/  |  heads: $HEAD_DIR/  |  logs: $LOG_DIR/"

lane 0 "${gpu0[@]}" &
lane 1 "${gpu1[@]}" &
lane 2 "${gpu2[@]}" &
wait

echo "[$(date +%H:%M:%S)] Refdelta sweep complete. Trained heads under $HEAD_DIR/"

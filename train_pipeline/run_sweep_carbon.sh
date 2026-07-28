#!/usr/bin/env bash
#
# run_sweep_carbon.sh
# -------------------
# Carbon-500M analog of run_sweep.sh. Generate embedding caches with the
# HuggingFaceBio/Carbon-500M backend and train heads for every combination of
#   half-window ∈ {250, 500, 1000}  ×  snp-only ∈ {off, on}   = 6 caches
# and, for each cache, train both head VARIANTS (absolute + reference-delta),
# each as a linear and an mlp head:
#   6 caches × 2 variants × 2 heads                            = 24 head runs.
#
# Same structure as run_sweep.sh, with three Carbon-specific changes:
#   * embeds with --backend carbon (no --attn-impl; that's a dnabert2-only knob).
#   * smaller embed --batch-size (Carbon-500M is ~4× DNABERT-2's param count).
#   * heads land FLAT under one root, named ${variant}_${head}_${tag}, so the
#     track_training notebook can pick up every run with a single os.listdir
#     (the nested variant/ layout would collide run labels). See the snippet at
#     the bottom of this file for the exact MODEL_PATHS to paste.
#
# The 6 cache configs are spread round-robin across the 3 GPUs (2 per GPU). Each
# GPU processes its configs sequentially (embed → train all variants/heads), and
# the three GPU lanes run concurrently.
#
# Per-config stdout/stderr is captured under logs/sweep_carbon/<tag>.log; high-level
# START/DONE markers go to this script's stdout so you can watch progress.
#
#   bash train_pipeline/run_sweep_carbon.sh
#
set -uo pipefail

cd /home/andrew/svar
PY=/home/andrew/anaconda3/envs/svar/bin/python

CACHE_DIR=checkpoints/sweep_carbon
HEAD_DIR=trained_heads/sweep_carbon
LOG_DIR=logs/sweep_carbon
mkdir -p "$CACHE_DIR" "$HEAD_DIR" "$LOG_DIR"

# ── Sweep knobs ───────────────────────────────────────────────────────────────
HALF_WINDOWS=(2000)
HEADS=(linear mlp)
VARIANTS=(absolute refdelta centered)   # absolute=FPSumHeadModel, refdelta=ref-subtracted, centered=per-window-mean-subtracted
POOL=mean               # window pooling mode (sum|mean); passed to train_head.py
MODEL_PATH=HuggingFaceBio/Carbon-500M   # encoder repo (swap to Carbon-3B to scale up)
BATCH=16               # embed batch size (500M backbone — smaller than dnabert2's 64)
MAXLEN=2048            # tokenizer truncation (>= longest 2*half-window in tokens)
EPOCHS=15
LR=1e-3

# Extra train_head.py flags per head variant ('' for absolute).
variant_flags() {
  case $1 in
    absolute) echo "" ;;
    refdelta) echo "--subtract-reference" ;;
    centered) echo "--center-windows" ;;
    *) echo "!!! unknown variant: $1" >&2; return 1 ;;
  esac
}

# ── One full config: embed the cache, then train every variant×head on it ─────
# Pinned to whatever GPU the calling lane exported as CUDA_VISIBLE_DEVICES.
run_config() {
  local hw=$1 snp_only=$2
  local suffix="" snp_arg=""
  if [[ $snp_only -eq 1 ]]; then suffix="_snponly"; snp_arg="--snp-only"; fi
  local tag="hw${hw}${suffix}"
  local cache="$CACHE_DIR/sativas413_carbon_${tag}.ckpt.pt"
  local log="$LOG_DIR/${tag}.log"

  echo "[$(date +%H:%M:%S)] [gpu ${CUDA_VISIBLE_DEVICES}] START $tag"
  {
    echo "######## $(date) :: embed $tag ########"
    $PY train_pipeline/embed_windows.py \
        --backend carbon --model-path "$MODEL_PATH" \
        --half-window "$hw" $snp_arg \
        --max-length "$MAXLEN" --batch-size "$BATCH" \
        --output "$cache" \
      || { echo "!!! EMBED FAILED: $tag"; return 1; }

    for variant in "${VARIANTS[@]}"; do
      local vflags; vflags=$(variant_flags "$variant") || return 1
      for head in "${HEADS[@]}"; do
        echo "######## $(date) :: train $variant $head $tag ########"
        $PY -m training.emb_nn.run \
            --cache "$cache" --head "$head" $vflags --pool "$POOL" \
            --half-window "$hw" \
            --epochs "$EPOCHS" --lr "$LR" --warm-start-standardizer \
            --output "$HEAD_DIR/${variant}_${head}_${tag}/model.pt" \
          || echo "!!! TRAIN FAILED: $variant $head $tag"
      done
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

# ── One sequential lane per GPU; lanes run concurrently ───────────────────────
lane() {
  export CUDA_VISIBLE_DEVICES=$1; shift
  for cfg in "$@"; do
    run_config $cfg   # unquoted: splits "hw snp" into two args
  done
}

echo "Carbon sweep: ${#configs[@]} caches × ${#VARIANTS[@]} variants × ${#HEADS[@]} heads on 3 GPUs (backend=carbon, model=$MODEL_PATH, pool=$POOL)"
echo "  variants: ${VARIANTS[*]}"
echo "  gpu0: ${gpu0[*]}"
echo "  gpu1: ${gpu1[*]}"
echo "  gpu2: ${gpu2[*]}"
echo "Logs: $LOG_DIR/  |  caches: $CACHE_DIR/  |  heads: $HEAD_DIR/"

lane 0 "${gpu0[@]}" &
lane 1 "${gpu1[@]}" &
lane 2 "${gpu2[@]}" &
wait

echo "[$(date +%H:%M:%S)] Carbon sweep complete. Trained heads under $HEAD_DIR/ (flat ${variant}_${head}_${tag} dirs)"

# ── View the same table in train_pipeline/track_training.ipynb ────────────────
# Replace the MODEL_PATHS cell with:
#
#   import os
#   from pathlib import Path
#   base_dir = "../trained_heads/sweep_carbon"
#   MODEL_PATHS = [Path(f"{base_dir}/{s}/model.metrics.jsonl") for s in sorted(os.listdir(base_dir))]
#
# Every run dir is named <variant>_<head>_<tag>, so load_all labels them uniquely.

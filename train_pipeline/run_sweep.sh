#!/usr/bin/env bash
#
# run_sweep.sh
# ------------
# Generate embedding caches and train heads for every combination of
#   half-window ∈ {250, 500, 1000}  ×  snp-only ∈ {off, on}   = 6 caches
# and, for each cache, train both head VARIANTS (absolute + reference-delta),
# each as a linear and an mlp head:
#   6 caches × 2 variants × 2 heads                            = 24 head runs.
#
# Consolidates the old run_window_sweep.sh (absolute heads) and
# run_refdelta_sweep.sh (reference-delta heads) — the reference delta is computed
# from the same cache, so both variants reuse the one embed per config (no
# re-embedding). The variant is just a train_head.py flag (--subtract-reference).
#
# The 6 cache configs are spread round-robin across the 3 GPUs (2 per GPU). Each
# GPU processes its configs sequentially (embed → train all variants/heads), and
# the three GPU lanes run concurrently. Caches use the pure-PyTorch ("torch")
# attention backend.
#
# Per-config stdout/stderr is captured under logs/sweep/<tag>.log; high-level
# START/DONE markers go to this script's stdout so you can watch progress.
#
#   bash train_pipeline/run_sweep.sh
#
set -uo pipefail

cd /home/andrew/svar
PY=/home/andrew/anaconda3/envs/svar/bin/python

CACHE_DIR=checkpoints/sweep
HEAD_DIR=trained_heads/sweep
LOG_DIR=logs/sweep
mkdir -p "$CACHE_DIR" "$HEAD_DIR" "$LOG_DIR"

# ── Sweep knobs ───────────────────────────────────────────────────────────────
HALF_WINDOWS=(250 500 1000)
HEADS=(linear mlp)
VARIANTS=(absolute refdelta)   # absolute = FPSumHeadModel, refdelta = ref-subtracted
POOL=sum               # window pooling mode (sum|mean); passed to train_head.py
BACKEND=torch          # attention backend for embedding generation
BATCH=64               # embed batch size
MAXLEN=2048            # tokenizer truncation (>= longest 2*half-window in tokens)
EPOCHS=100
LR=1e-3

# Extra train_head.py flags per head variant ('' for absolute).
variant_flags() {
  case $1 in
    absolute) echo "" ;;
    refdelta) echo "--subtract-reference" ;;
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
  local cache="$CACHE_DIR/sativas413_${tag}.ckpt.pt"
  local log="$LOG_DIR/${tag}.log"

  echo "[$(date +%H:%M:%S)] [gpu ${CUDA_VISIBLE_DEVICES}] START $tag"
  {
    echo "######## $(date) :: embed $tag ########"
    $PY train_pipeline/embed_windows.py \
        --half-window "$hw" $snp_arg \
        --attn-impl "$BACKEND" \
        --max-length "$MAXLEN" --batch-size "$BATCH" \
        --output "$cache" \
      || { echo "!!! EMBED FAILED: $tag"; return 1; }

    for variant in "${VARIANTS[@]}"; do
      local vflags; vflags=$(variant_flags "$variant") || return 1
      for head in "${HEADS[@]}"; do
        echo "######## $(date) :: train $variant $head $tag ########"
        $PY train_pipeline/train_head.py \
            --cache "$cache" --head "$head" $vflags --pool "$POOL" \
            --half-window "$hw" \
            --epochs "$EPOCHS" --lr "$LR" --warm-start-standardizer \
            --output "$HEAD_DIR/${variant}/${head}_${tag}/model.pt" \
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

echo "Sweep: ${#configs[@]} caches × ${#VARIANTS[@]} variants × ${#HEADS[@]} heads on 3 GPUs (backend=$BACKEND, pool=$POOL)"
echo "  variants: ${VARIANTS[*]}"
echo "  gpu0: ${gpu0[*]}"
echo "  gpu1: ${gpu1[*]}"
echo "  gpu2: ${gpu2[*]}"
echo "Logs: $LOG_DIR/  |  caches: $CACHE_DIR/  |  heads: $HEAD_DIR/"

lane 0 "${gpu0[@]}" &
lane 1 "${gpu1[@]}" &
lane 2 "${gpu2[@]}" &
wait

echo "[$(date +%H:%M:%S)] Sweep complete. Trained heads under $HEAD_DIR/ (per variant subdir)"

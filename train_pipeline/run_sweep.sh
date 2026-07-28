#!/usr/bin/env bash
#
# run_sweep.sh
# ------------
# Generate embedding caches and train heads for every combination of
#   half-window ∈ HALF_WINDOWS  ×  snp-only ∈ {off, on}   = N caches
# and, for each cache, train every head VARIANT (absolute + reference-delta + centered),
# each as a linear and an mlp head, over the POOL × WARM_START_STANDARDIZER product
# (sum-pool + no-standardizer skipped):
#   N caches × |VARIANTS| × |HEADS| × (|POOL|·|WARM_START| − skips)  head runs.
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
HALF_WINDOWS=(1000)
HEADS=(linear mlp)
VARIANTS=(absolute centered refdelta)   # absolute=FPSumHeadModel, refdelta=ref-subtracted, centered=per-window-mean-subtracted
POOL=(mean sum)        # window pooling modes to sweep (sum|mean); passed to train_head.py
BACKEND=torch          # attention backend for embedding generation
BATCH=64               # embed batch size
MAXLEN=2048            # tokenizer truncation (>= longest 2*half-window in tokens)
EPOCHS=15
LR=1e-3
WARM_START_STANDARDIZER=(0 1)   # 0 = train WITHOUT --warm-start-standardizer, 1 = with it

# Heads train over the full POOL × WARM_START_STANDARDIZER product, EXCEPT sum-pool + no-standardizer
# (unbounded pooled magnitude with no normalization), which is skipped in the loop below. That leaves:
#   mean + warm-start, mean + no-warm-start, sum + warm-start
# Delete the `continue` guard in run_config to also run sum + no-warm-start (the full 4-way product).

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
        for pool in "${POOL[@]}"; do
          for ws in "${WARM_START_STANDARDIZER[@]}"; do
            # Skip sum-pool without the standardizer (unbounded magnitude). Delete to run all 4.
            if [[ "$pool" == "sum" && $ws -eq 0 ]]; then continue; fi

            local ws_flag="" ws_tag="ws0"
            if [[ $ws -eq 1 ]]; then ws_flag="--warm-start-standardizer"; ws_tag="ws1"; fi
            local run="${head}_${tag}_${pool}_${ws_tag}"

            echo "######## $(date) :: train $variant $run ########"
            $PY -m training.emb_nn.run \
                --cache "$cache" --head "$head" $vflags --pool "$pool" \
                --half-window "$hw" \
                --epochs "$EPOCHS" --lr "$LR" $ws_flag \
                --output "$HEAD_DIR/${variant}/${run}/model.pt" \
              || echo "!!! TRAIN FAILED: $variant $run"
          done
        done
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

echo "Sweep: ${#configs[@]} caches × ${#VARIANTS[@]} variants × ${#HEADS[@]} heads × {pool×warmstart} on 3 GPUs (backend=$BACKEND)"
echo "  variants: ${VARIANTS[*]}  |  pool: ${POOL[*]}  |  warm-start: ${WARM_START_STANDARDIZER[*]} (sum+nostd skipped)"
echo "  gpu0: ${gpu0[*]}"
echo "  gpu1: ${gpu1[*]}"
echo "  gpu2: ${gpu2[*]}"
echo "Logs: $LOG_DIR/  |  caches: $CACHE_DIR/  |  heads: $HEAD_DIR/"

lane 0 "${gpu0[@]}" &
lane 1 "${gpu1[@]}" &
lane 2 "${gpu2[@]}" &
wait

echo "[$(date +%H:%M:%S)] Sweep complete. Trained heads under $HEAD_DIR/ (per variant subdir)"

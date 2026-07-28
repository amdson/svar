#!/usr/bin/env bash
#
# run_head_sweep.sh
# -----------------
# Train heads over the full Cartesian product of a set of sweep knobs, on a set
# of *pre-built* embedding caches (e.g. the ones produced by embed_all_windows.sh).
# Unlike run_sweep.sh this does NOT embed anything — you hand it cache paths.
#
# Every knob below is a bash array (a "tuple"); the script iterates over all
# combinations. To sweep a new axis, add values to its array. To pin an axis,
# give its array a single value. The per-cache --half-window is read from each
# cache's own metadata, so caches of different window sizes can be mixed freely.
#
#   total runs = |CACHES|·|VARIANTS|·|HEADS|·|POOL|·|WARM_START|·|LR|·|WEIGHT_DECAY|·|EPOCHS|
#                (minus the sum-pool + no-standardizer combos, skipped by default)
#
# Per-run stdout/stderr -> logs/head_sweep/<run>.log ; START/DONE markers + a
# preflight combo count go to this script's stdout. Set DRYRUN=1 to print the
# train_head.py commands without running them.
#
#   bash train_pipeline/run_head_sweep.sh
#   DRYRUN=1 bash train_pipeline/run_head_sweep.sh      # preview only
# 
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."      # repo root (script lives in train_pipeline/)
PY=${PY:-python}

HEAD_DIR=trained_heads/vc_comparison
LOG_DIR=logs/vc_comparison
mkdir -p "$HEAD_DIR" "$LOG_DIR"

# ── Sweep knobs (edit these tuples) ───────────────────────────────────────────
# Embedding caches to train on (tuple of ckpt paths) — all 16 produced by
# embed_all_windows.sh: {variant-cache, full-forward} × {250,500,1000,2000} nt ×
# {mean, snp} within-window pooling.
CACHES=(
  checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt
  checkpoints/manual/sativas413_carbon500m_hw500_snponly.ckpt.pt
  checkpoints/manual/sativas413_carbon500m_vc_hw500.ckpt.pt
  checkpoints/manual/sativas413_carbon500m_vc_hw500_snponly.ckpt.pt
)
HEADS=(linear mlp)                       # head architecture
VARIANTS=(centered)    # absolute / per-window-mean-subtracted / reference-subtracted
POOL=(sum)                          # window pooling
STANDARDIZER=(perdim rms)                # de-mean/rescale layer: perdim (per-dim std) vs rms (single scalar)
WARM_START_STANDARDIZER=(1)            # 0 = no --warm-start-standardizer, 1 = with it
LR=(1e-3)                                # learning rate(s)
WEIGHT_DECAY=(1e-4)                      # weight decay(s)
EPOCHS=(15)                              # epoch count(s)

# Fixed (non-swept) knobs. Promote any of these to an array + nested loop the
# same way if you want to sweep it.
BATCH=64
HIDDEN_DIM=""        # MLP hidden width ("" = default emb_dim); ignored by linear
N_LAYERS=4           # MLP residual blocks; ignored by linear
DROPOUT=0.2          # MLP dropout; ignored by linear

# SKIP_SUM_NOSTD=1     # 1 = skip sum-pool + no-standardizer (unbounded magnitude); 0 = run it

# ── train_head.py flags per head variant ──────────────────────────────────────
variant_flags() {
  case $1 in
    absolute) echo "" ;;
    refdelta) echo "--subtract-reference" ;;
    centered) echo "--center-windows" ;;
    *) echo "!!! unknown variant: $1" >&2; return 1 ;;
  esac
}

# ── Preflight: count the combos ───────────────────────────────────────────────
total=$(( ${#CACHES[@]} * ${#VARIANTS[@]} * ${#HEADS[@]} * ${#POOL[@]} * ${#STANDARDIZER[@]} \
         * ${#WARM_START_STANDARDIZER[@]} * ${#LR[@]} * ${#WEIGHT_DECAY[@]} * ${#EPOCHS[@]} ))
echo "Head sweep: up to $total runs)"
echo "  caches:${#CACHES[@]}  variants:${VARIANTS[*]}  heads:${HEADS[*]}  pool:${POOL[*]}  standardizer:${STANDARDIZER[*]}"
echo "  warm-start:${WARM_START_STANDARDIZER[*]}  lr:${LR[*]}  wd:${WEIGHT_DECAY[*]}  epochs:${EPOCHS[*]}"
echo "  logs:$LOG_DIR/  heads:$HEAD_DIR/"

# ── Sweep ─────────────────────────────────────────────────────────────────────
for cache in "${CACHES[@]}"; do
  if [[ ! -f "$cache" ]]; then echo "!!! MISSING CACHE (skipping): $cache"; continue; fi
  # Per-cache half-window straight from the cache metadata (no manual bookkeeping).
  hw=$($PY -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['metadata']['half_window'])" "$cache") \
    || { echo "!!! could not read half_window from $cache"; continue; }
  ctag=$(basename "$cache"); ctag=${ctag%.ckpt.pt}

  for variant in "${VARIANTS[@]}"; do
    vflags=$(variant_flags "$variant") || continue
    for head in "${HEADS[@]}"; do
      hdr_flags=""
      if [[ "$head" == "mlp" ]]; then
        hdr_flags="--n-layers $N_LAYERS --dropout $DROPOUT"
        [[ -n "$HIDDEN_DIM" ]] && hdr_flags="$hdr_flags --hidden-dim $HIDDEN_DIM"
      fi
      for pool in "${POOL[@]}"; do
        for std in "${STANDARDIZER[@]}"; do
          for ws in "${WARM_START_STANDARDIZER[@]}"; do
            if [[ "$pool" == "sum" && $ws -eq 0 && ${SKIP_SUM_NOSTD:-1} -eq 1 ]]; then continue; fi
            ws_flag=""; ws_tag="ws0"
            if [[ $ws -eq 1 ]]; then ws_flag="--warm-start-standardizer"; ws_tag="ws1"; fi
            for lr in "${LR[@]}"; do
              for wd in "${WEIGHT_DECAY[@]}"; do
                for ep in "${EPOCHS[@]}"; do
                  run="${ctag}__${variant}_${head}_${pool}_${std}_${ws_tag}_lr${lr}_wd${wd}_ep${ep}"
                  out="$HEAD_DIR/${variant}/${run}/model.pt"
                  # Resumable: skip a run whose head is already on disk (so a
                  # re-submitted job picks up where the last one left off).
                  if [[ "${DRYRUN:-0}" -ne 1 && -s "$out" ]]; then
                    echo "[$(date +%H:%M:%S)] SKIP  $run (exists)"; continue
                  fi
                  cmd="$PY -m training.emb_nn.run \
--cache $cache --half-window $hw --head $head $vflags $hdr_flags \
--pool $pool --standardizer $std $ws_flag --lr $lr --weight-decay $wd --epochs $ep --batch-size $BATCH \
--output $out"

                  if [[ "${DRYRUN:-0}" -eq 1 ]]; then echo "$cmd"; continue; fi
                  echo "[$(date +%H:%M:%S)] START $run"
                  $cmd > "$LOG_DIR/${run}.log" 2>&1 \
                    && echo "[$(date +%H:%M:%S)] DONE  $run" \
                    || echo "[$(date +%H:%M:%S)] FAIL  $run (see $LOG_DIR/${run}.log)"
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "[$(date +%H:%M:%S)] Head sweep complete. Heads under $HEAD_DIR/ (per-variant subdir)."

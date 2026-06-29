#!/usr/bin/env bash
# Finisher: wait for the background AR training to complete, then run the
# remaining GPU steps serially (so they don't contend for the V100), and wait
# on the head-diff job. One log to watch.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew.dickson/.conda/envs/svar/bin/python
AR_PID="${1:?need AR pid}"
HEAD_PID="${2:?need head-diff pid}"

echo "[finish] waiting for AR training (pid $AR_PID) …"
while kill -0 "$AR_PID" 2>/dev/null; do sleep 15; done
echo "[finish] AR training finished."

if [[ -f checkpoints/variant_ar/carbon_vc.metrics.jsonl ]]; then
  echo "[finish] plotting fig 05 (AR curves) …"
  $PY presentation/plot_ar_curves.py \
    --metrics checkpoints/variant_ar/carbon_vc.metrics.jsonl || echo "[finish] AR plot FAILED"
else
  echo "[finish] ERROR: AR metrics sidecar missing; skipping fig 05."
fi

echo "[finish] generating correctness table (Table 1) …"
$PY presentation/make_correctness_table.py || echo "[finish] correctness table FAILED"

echo "[finish] waiting for head-diff job (pid $HEAD_PID) …"
while kill -0 "$HEAD_PID" 2>/dev/null; do sleep 10; done
echo "[finish] head-diff job finished."

echo "[finish] final figure check:"
$PY presentation/paper/check_figures.py || true
echo "[finish] ALL DONE."

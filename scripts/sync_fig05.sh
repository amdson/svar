#!/usr/bin/env bash
# Wait for the AR run + finisher to regenerate presentation/figs/05_ar_curves.png,
# then copy it into the self-contained paper/figs/ bundle.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew.dickson/.conda/envs/svar/bin/python
AR_PID="${1:?need AR pid}"
# 1) wait for AR training to exit
while kill -0 "$AR_PID" 2>/dev/null; do sleep 20; done
# 2) wait until the finisher has replotted fig 05 fresh (mtime within last 30 min)
for _ in $(seq 1 60); do
  if [ "$(find presentation/figs/05_ar_curves.png -mmin -30 2>/dev/null)" ]; then break; fi
  sleep 15
done
cp presentation/figs/05_ar_curves.png presentation/paper/figs/05_ar_curves.png
echo "[sync] copied fresh fig 05 into paper/figs/ at $(date +%H:%M)"
$PY presentation/paper/check_figures.py || true
echo "[sync] done"

#!/bin/bash
#
# Finish/verify the GSE43857 allc bulk download. Resumable: existing
# gzip-valid files are skipped, so this is cheap if the interactive-session
# download already completed. Exits nonzero if any file is still missing.
#
#SBATCH --job-name=ath_meth_dl
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --account=small_grains
#SBATCH --output=/home/andrew.dickson/ath_meth_benchmark/logs/download_%j.out
#SBATCH --error=/home/andrew.dickson/ath_meth_benchmark/logs/download_%j.err

set -uo pipefail
BASE=/90daydata/small_grains/andrew.dickson/datasets/arabidopsis/methylation

bash /home/andrew.dickson/ath_meth_benchmark/build/download_allc.sh

n_expected=$(wc -l < "$BASE/meta/allc_download_manifest.txt")
n_present=$(ls "$BASE/allc" | wc -l)
echo "present: $n_present / $n_expected"
[ "$n_present" -ge "$n_expected" ]

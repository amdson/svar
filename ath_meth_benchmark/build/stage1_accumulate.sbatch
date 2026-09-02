#!/bin/bash
#
# Stage 1 streaming accumulator pass over 811 GSE43857 allc files.
# Resumable: finished worker chunks are skipped via partials/chunkNNN.npz.
#
#   sbatch stage1_accumulate.sbatch
#
#SBATCH --job-name=ath_meth_stage1
#SBATCH --partition=atlas
#SBATCH --cpus-per-task=34
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --account=small_grains
#SBATCH --output=/home/andrew.dickson/ath_meth_benchmark/logs/stage1_%j.out
#SBATCH --error=/home/andrew.dickson/ath_meth_benchmark/logs/stage1_%j.err

set -uo pipefail

DATA=/90daydata/small_grains/andrew.dickson/datasets/arabidopsis
BENCH=/home/andrew.dickson/ath_meth_benchmark

source /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu/etc/profile.d/conda.sh
conda activate svar

echo "Host: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK  $(date)"

python "$BENCH/build/stage1_accumulate.py" \
    --allc-dir "$DATA/methylation/allc" \
    --accessions "$DATA/methylation/meta/benchmark_accessions.tsv" \
    --fai "$DATA/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa.fai" \
    --out-dir "$DATA/methylation/stage1" \
    --min-cov 5 \
    --workers 16 \
&& python "$BENCH/build/stage2_select.py" \
    --stage1-dir "$DATA/methylation/stage1" \
    --gff3 "$DATA/annotation/Arabidopsis_thaliana.TAIR10.63.gff3.gz" \
    --te-txt "$DATA/annotation/TAIR10_Transposable_Elements.txt" \
    --out-dir "$DATA/methylation/stage2" \
    --kappa 3.0

# Stage 3 only proceeds on the spec's GO criterion (CG retained > 500k);
# MARGINAL/NO-GO stops here for user review of report_stage2.md.
if grep -q "VERDICT: GO" "$DATA/methylation/stage2/report_stage2.md"; then
    echo "GO — running stage 3 extraction"
    python "$BENCH/build/stage3_extract.py" \
        --stage1-dir "$DATA/methylation/stage1" \
        --stage2-dir "$DATA/methylation/stage2" \
        --allc-dir "$DATA/methylation/allc" \
        --accessions "$DATA/methylation/meta/benchmark_accessions.tsv" \
        --out "$DATA/methylation/dataset.h5" \
        --workers 16
else
    echo "Stage 2 verdict not GO — stopping for user review"
fi

echo "DONE $(date)"

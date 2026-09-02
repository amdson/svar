#!/bin/bash
#
# T1 baseline sandwich: LD-pruned GRM + kinship BLUP + cis elastic net +
# cis-h2 (HE) for all 22.6k genes on the 665-accession expression dataset.
#
#SBATCH --job-name=ath_expr_t1
#SBATCH --partition=atlas
#SBATCH --cpus-per-task=34
#SBATCH --mem=340G
#SBATCH --time=24:00:00
#SBATCH --account=small_grains
#SBATCH --output=/home/andrew.dickson/ath_meth_benchmark/logs/t1_%j.out
#SBATCH --error=/home/andrew.dickson/ath_meth_benchmark/logs/t1_%j.err

set -uo pipefail

DATA=/90daydata/small_grains/andrew.dickson/datasets/arabidopsis
BENCH=/home/andrew.dickson/ath_meth_benchmark
PFX=$DATA/methylation/arabidopsis_1001g_biallelic_unfiltered
OUT=$DATA/expression/baselines

source /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu/etc/profile.d/conda.sh
conda activate svar
mkdir -p "$OUT"

echo "Host: $(hostname)  $(date)"

# keep-list: the 665 expression accessions
python - "$DATA/expression/expression_dataset.h5" "$OUT/keep665.txt" <<'EOF'
import sys, h5py
with h5py.File(sys.argv[1]) as h5, open(sys.argv[2], "w") as out:
    for x in h5["accessions/ecotype_id"][:]:
        out.write(x.decode() + "\n")
EOF

if [ ! -s "$OUT/grm_prune.prune.in" ]; then
    plink2 --pfile "$PFX" \
        --keep "$OUT/keep665.txt" \
        --maf 0.05 --geno 0.1 \
        --indep-pairwise 50 5 0.2 \
        --threads "$SLURM_CPUS_PER_TASK" --memory 300000 \
        --out "$OUT/grm_prune"
fi

python "$BENCH/baselines/t1_sandwich.py" \
    --h5 "$DATA/expression/expression_dataset.h5" \
    --pfile "$PFX" \
    --prune-in "$OUT/grm_prune.prune.in" \
    --out-dir "$OUT" \
    --cis-window 100000 \
    --maf-min 0.01 \
    --workers 30

echo "DONE $(date)"

#!/bin/bash
#
# Stage 4 (spec §5): unfiltered-biallelic pgen + per-accession SNV arrays.
# Uses arabidopsis_1001g_biallelic_snps.vcf.gz (NO MAF/missingness filters —
# rare alleles are the §3b context-changing signal), not the `_final` set.
#
#SBATCH --job-name=ath_meth_stage4
#SBATCH --partition=atlas
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --account=small_grains
#SBATCH --output=/home/andrew.dickson/ath_meth_benchmark/logs/stage4_%j.out
#SBATCH --error=/home/andrew.dickson/ath_meth_benchmark/logs/stage4_%j.err

set -uo pipefail

DATA=/90daydata/small_grains/andrew.dickson/datasets/arabidopsis
BENCH=/home/andrew.dickson/ath_meth_benchmark
PFX=$DATA/methylation/arabidopsis_1001g_biallelic_unfiltered

source /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-25.11.1-tjcxbz5i5qh6q7of6cwnp2bdstdptxxu/etc/profile.d/conda.sh
conda activate svar

echo "Host: $(hostname)  $(date)"

if [ ! -s "$PFX.pgen" ]; then
    # sort+ref-align first (plink2 forbids --sort-vars with --export), pgen only
    plink2 --vcf "$DATA/arabidopsis_1001g_biallelic_snps.vcf.gz" \
        --set-all-var-ids '@:#:$r:$a' \
        --ref-from-fa force --fa "$DATA/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa" \
        --sort-vars \
        --make-pgen \
        --threads "$SLURM_CPUS_PER_TASK" --memory 100000 \
        --out "$PFX"
fi

python "$BENCH/build/stage4_snv_arrays.py" \
    --pfile "$PFX" \
    --out "$DATA/methylation/snv_arrays.h5"

echo "DONE $(date)"

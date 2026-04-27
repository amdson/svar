#!/usr/bin/env bash
# Downloads and decompresses all inputs required by Makefile.
# Run once from variant_cache/:  bash get_data.sh
set -euo pipefail

DATA=../rice_data
mkdir -p "$DATA"

# ── Download ──────────────────────────────────────────────────────────────────

wget -c -P "$DATA" \
    https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa.gz

wget -c -P "$DATA" \
    https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/assembly_chain/oryza_sativa/MSU6_to_IRGSP-1.0.chain.gz

wget -c -P "$DATA" \
    http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity.44K.MSU6.Genotypes_PLINK.zip

wget -c -P "$DATA" \
    http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity_44K_Phenotypes_34traits_PLINK.txt

wget -c -P "$DATA" \
    http://www.ricediversity.org/data/sets/44kgwas/RiceDiversity.44K.MSU6.SNP_flanking_seq.txt

# ── Decompress ────────────────────────────────────────────────────────────────

# -k keeps the .gz so re-running is idempotent; remove manually to save space
gunzip -kf "$DATA/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa.gz"
gunzip -kf "$DATA/MSU6_to_IRGSP-1.0.chain.gz"

# ZIP extracts to a folder; Makefile expects RiceDiversity_44K_Genotypes_PLINK/
unzip -q -o "$DATA/RiceDiversity.44K.MSU6.Genotypes_PLINK.zip" -d "$DATA"

# The ZIP may unpack with dots in the folder name — rename if needed
PLINK_DOT="$DATA/RiceDiversity.44K.MSU6.Genotypes_PLINK"
PLINK_DIR="$DATA/RiceDiversity_44K_Genotypes_PLINK"
if [[ -d "$PLINK_DOT" && ! -d "$PLINK_DIR" ]]; then
    mv "$PLINK_DOT" "$PLINK_DIR"
fi

# ── Index FASTA ───────────────────────────────────────────────────────────────
# Required by pyfaidx, pysam, and CrossMap before make can run.

conda run -n svar samtools faidx "$DATA/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa"

echo "All data ready. Run: make"
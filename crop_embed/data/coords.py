"""
crop_embed/data/coords.py
--------------------------
Coordinate mapping and VCF remapping for the Rice Diversity 44K dataset.

The reference FASTA (GCA_rice.fasta) uses MSU7/IRGSP1 coordinates, while the
original VCF and flanking-seq table use MSU6/IRGSP.v4 coordinates.
``build_msu6_to_msu7_map`` builds a per-SNP lookup table bridging the two.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from pyfaidx import Fasta

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

_DATA_DIR    = Path(__file__).resolve().parents[3] / "rice_data"

FASTA_PATH   = str(_DATA_DIR / "Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa")
VCF_PATH     = str(_DATA_DIR / "RiceDiversity_44K_Genotypes_PLINK" / "sativas413.vcf")
FLANKING_PATH = str(_DATA_DIR / "RiceDiversity.44K.MSU6.SNP_flanking_seq.txt")
SNP_INFO_PATH = str(_DATA_DIR / "RiceDiversity.44K.MSU6.SNP_Information.MSU7.txt")
DEFAULT_VCF_PATH = str(_DATA_DIR / "sativas413_msu7_final.vcf")

def chrom_name_map(fasta_path: str = FASTA_PATH) -> dict[int, str]:
    """Return {chrom_int: fasta_key} for numeric chromosomes in the FASTA."""
    ref = Fasta(fasta_path)
    return {int(name): name for name in ref.keys() if name.isdigit()}

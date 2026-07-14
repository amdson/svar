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
# Rice data is migrated to cluster scratch (see svar/env.sh + datasets/common.mk):
# ``$DATA_ROOT/rice``, with ``DATA_ROOT`` defaulting to ``$SVAR_SCRATCH/datasets``.
# Prefer the scratch copy; fall back to the legacy ~/rice_data build if scratch
# isn't populated yet, so this keeps working mid-migration.

def _rice_data_dir() -> Path:
    scratch = os.environ.get("SVAR_SCRATCH", "/90daydata/small_grains/andrew.dickson")
    data_root = os.environ.get("DATA_ROOT", str(Path(scratch) / "datasets"))
    scratch_rice = Path(data_root) / "rice"
    legacy_rice = Path(__file__).resolve().parents[3] / "rice_data"
    return scratch_rice if scratch_rice.exists() else legacy_rice


_DATA_DIR    = _rice_data_dir()

FASTA_PATH   = str(_DATA_DIR / "Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa")
VCF_PATH     = str(_DATA_DIR / "RiceDiversity_44K_Genotypes_PLINK" / "sativas413.vcf")
FLANKING_PATH = str(_DATA_DIR / "RiceDiversity.44K.MSU6.SNP_flanking_seq.txt")
SNP_INFO_PATH = str(_DATA_DIR / "RiceDiversity.44K.MSU6.SNP_Information.MSU7.txt")
DEFAULT_VCF_PATH = str(_DATA_DIR / "sativas413_msu7_final.vcf")
DEFAULT_PHENO_PATH = str(_DATA_DIR / "RiceDiversity_44K_Phenotypes_34traits_PLINK.txt")


# ---------------------------------------------------------------------------
# Coordinate map
# ---------------------------------------------------------------------------

def build_msu6_to_msu7_map() -> pd.Series:
    """
    Build a lookup table from MSU6 SNP positions to MSU7 FASTA positions.

    The SNP_Information file maps ~36 k of the 44 k SNPs from MSU6 to MSU7.
    For the remaining ~7 k SNPs the offset is estimated from the nearest
    mapped neighbour on the same chromosome (piecewise-constant interpolation).

    Returns
    -------
    pd.Series indexed by (chr, pos_msu6) → pos_msu7 (int)
    Both index levels use plain Python ints.

    Notes
    -----
    The SNP_Information file has a header/data column-count mismatch (7 header
    fields, 8 data fields), requiring explicit column names on read.
    """
    flanking = pd.read_csv(FLANKING_PATH, sep="\t")

    msu7 = pd.read_csv(
        SNP_INFO_PATH, sep="\t", header=0,
        names=["CHR", "SNPID", "cM",
               "pos_IRGSP_v4", "pos_MSU6",
               "pos_IRGSP1_MSU7", "MAF", "callrates"],
    )
    msu7["pos_IRGSP1_MSU7"] = pd.to_numeric(msu7["pos_IRGSP1_MSU7"], errors="coerce")

    df = flanking.rename(columns={"pos": "pos_msu6"}).merge(
        msu7[["SNPID", "pos_IRGSP1_MSU7"]],
        left_on="snp_id", right_on="SNPID", how="left",
    )
    df["_offset"] = df["pos_IRGSP1_MSU7"] - df["pos_msu6"]

    # Interpolate offset for SNPs absent from the MSU7 map
    interpolated = df["_offset"].copy()
    for _, grp in df.groupby("chr"):
        mapped   = grp["_offset"].notna()
        unmapped = ~mapped
        if not mapped.any() or not unmapped.any():
            continue
        mapped_pos = grp.loc[mapped,   "pos_msu6"].values
        mapped_off = grp.loc[mapped,   "_offset"].values
        query_pos  = grp.loc[unmapped, "pos_msu6"].values
        nearest    = np.abs(query_pos[:, None] - mapped_pos[None, :]).argmin(axis=1)
        interpolated.loc[grp.index[unmapped]] = mapped_off[nearest]

    df["pos_msu7"] = (df["pos_msu6"] + interpolated).round().astype(int)
    return df.set_index(["chr", "pos_msu6"])["pos_msu7"]


# ---------------------------------------------------------------------------
# VCF coordinate remapping
# ---------------------------------------------------------------------------

def remap_vcf_coordinates(
    input_vcf_path: str,
    output_vcf_path: str,
    coord_map: pd.Series,
    drop_unmapped: bool = True,
) -> dict:
    """
    Write a new VCF with positions remapped from MSU6 to MSU7 coordinates.

    Parameters
    ----------
    input_vcf_path  : path to the source VCF (MSU6 coordinates)
    output_vcf_path : path for the remapped output VCF
    coord_map       : Series returned by build_msu6_to_msu7_map(),
                      indexed by (chr_int, pos_msu6_1based) → pos_msu7_1based
    drop_unmapped   : if True (default), records with no coord_map entry are
                      omitted; if False they are written with REMAP_MISSING=1

    Returns
    -------
    dict with keys "written", "dropped", "total"
    """
    in_vcf = pysam.VariantFile(input_vcf_path)
    header = in_vcf.header.copy()

    if not drop_unmapped:
        header.add_meta("INFO", items=[
            ("ID",          "REMAP_MISSING"),
            ("Number",      0),
            ("Type",        "Flag"),
            ("Description", "Position not found in MSU6-to-MSU7 coordinate map"),
        ])

    counts = {"written": 0, "dropped": 0, "total": 0}

    with pysam.VariantFile(output_vcf_path, "w", header=header) as out_vcf:
        for rec in in_vcf.fetch():
            counts["total"] += 1

            chrom_str       = rec.chrom
            chrom_int       = int(chrom_str) if chrom_str.isdigit() else int(chrom_str.lstrip("chr"))
            pos_msu6_1based = rec.pos  # pysam rec.pos == 1-based POS for this VCF

            pos_msu7 = coord_map.get((chrom_int, pos_msu6_1based))

            if pos_msu7 is None:
                if drop_unmapped:
                    counts["dropped"] += 1
                    continue
                rec.info["REMAP_MISSING"] = True
            else:
                rec.pos = int(pos_msu7) - 1  # pysam expects 0-based

            out_vcf.write(rec)
            counts["written"] += 1

    in_vcf.close()
    print(
        f"Remapped {counts['written']:,} / {counts['total']:,} records "
        f"({counts['dropped']:,} dropped) → {output_vcf_path}"
    )
    return counts


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def chrom_name_map(fasta_path: str = FASTA_PATH) -> dict[int, str]:
    """Return {chrom_int: fasta_key} for numeric chromosomes in the FASTA."""
    ref = Fasta(fasta_path)
    return {int(name): name for name in ref.keys() if name.isdigit()}

"""
training/common/datasets.py
---------------------------
Dataset registry. Each crop is described by a ``DatasetSpec`` that resolves the
paths to its canonical artifacts (plink2 pgen fileset + reference genome +
IID-keyed phenotype CSV). The rest of the pipeline is dataset-agnostic: it asks
``get_dataset(name)`` for a spec and never hardcodes a path.

Paths resolve under ``$DATA_ROOT/<crop>`` (default ``$SVAR_SCRATCH/datasets``),
the same convention as the dataset Makefiles. Rice additionally falls back to the
legacy ``~/rice_data`` build when scratch isn't populated (mirrors coords.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Columns in a phenotype CSV that are keys/flags, never traits.
NON_TRAIT_COLS = frozenset({"IID", "matched", "complete"})


def _default_scratch() -> str:
    """Cluster scratch when present, else the local ~/svar_scratch fallback
    (mirrors coords.py). An explicit $SVAR_SCRATCH always wins."""
    cluster = "/90daydata/small_grains/andrew.dickson"
    return cluster if Path(cluster).exists() else str(Path.home() / "svar_scratch")


def data_root() -> Path:
    scratch = os.environ.get("SVAR_SCRATCH") or _default_scratch()
    return Path(os.environ.get("DATA_ROOT", str(Path(scratch) / "datasets")))


def _rice_dir() -> Path:
    scratch_rice = data_root() / "rice"
    legacy_rice = Path.home() / "rice_data"
    return scratch_rice if scratch_rice.exists() else legacy_rice


def _vcf_for(stem: Path | str) -> str:
    """VCF path for a build stem, accepting either compression.

    The dataset Makefiles emit bgzipped VCFs where the uncompressed text would be
    unreasonably large (arabidopsis' final export is ~16 GB plain, ~1 GB bgzipped),
    while the older rice/soy builds left theirs plain. pysam — the only reader of
    this path (crop_embed/data/vcf.py) — takes both, so resolve whichever is on
    disk. Falls back to the plain name when neither exists, so the error message
    from ``samples()`` names the file the Makefile would build.
    """
    plain, gz = Path(f"{stem}.vcf"), Path(f"{stem}.vcf.gz")
    if plain.exists():
        return str(plain)
    return str(gz) if gz.exists() else str(plain)


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's canonical artifacts live. Paths are strings (may not all
    exist for every crop yet); accessors raise clearly if a needed file is absent."""
    name: str
    pgen_prefix: str          # plink2 fileset stem → .pgen/.pvar/.psam
    fasta_path: str           # reference genome (for window reconstruction)
    pheno_csv: str            # IID-keyed phenotype table
    vcf_path: str | None = None
    trait_cols: tuple[str, ...] | None = None   # None → infer from pheno_csv header

    # ── accessors ────────────────────────────────────────────────────────────
    @property
    def psam(self) -> str:
        return self.pgen_prefix + ".psam"

    def samples(self) -> list[str]:
        """Canonical sample order — the .psam IID column, or the VCF sample list
        when no pgen fileset exists (e.g. rice, which ships only a VCF)."""
        if Path(self.psam).exists():
            from crop_embed.data.pgen import _read_psam
            return _read_psam(self.psam)
        if self.vcf_path and Path(self.vcf_path).exists():
            import pysam
            with pysam.VariantFile(self.vcf_path) as vf:
                return list(vf.header.samples)
        raise FileNotFoundError(
            f"[{self.name}] no sample source: neither {self.psam} nor a VCF "
            f"({self.vcf_path}) exists — build the fileset (see datasets/{self.name}/Makefile)."
        )

    def resolved_trait_cols(self) -> list[str]:
        """Trait columns, from the spec if given else inferred from the pheno CSV."""
        if self.trait_cols is not None:
            return list(self.trait_cols)
        import pandas as pd
        if not Path(self.pheno_csv).exists():
            raise FileNotFoundError(
                f"[{self.name}] phenotype CSV not found: {self.pheno_csv}"
            )
        cols = pd.read_csv(self.pheno_csv, nrows=0).columns
        return [c for c in cols if c not in NON_TRAIT_COLS]

    def cache_path(self, backbone: str, half_window: int, *, snp_only: bool = False) -> str:
        """Convention for a fixed-embedding cache: address by (crop, backbone, window)."""
        scratch = os.environ.get("SVAR_SCRATCH", str(Path.home() / "svar_scratch"))
        suffix = "_snponly" if snp_only else ""
        return str(Path(scratch) / "caches" / self.name /
                   f"{backbone}_hw{half_window}{suffix}.ckpt.pt")


@lru_cache(maxsize=None)
def _registry() -> dict[str, DatasetSpec]:
    root = data_root()
    rice = _rice_dir()
    soy = root / "soy"
    arab = root / "arabidopsis"

    SOY_TRAITS = ("protein", "oil", "Linoleic", "Linolenic", "R1", "R8",
                  "Hgt", "Ldg", "SQ", "SdWgt", "Yield")

    return {
        "soy": DatasetSpec(
            name="soy",
            pgen_prefix=str(soy / "soysnp50k_a2_final"),
            fasta_path=str(soy / "Glycine_max.Glycine_max_v2.1.dna_sm.toplevel.fa"),
            pheno_csv=str(soy / "soy_pheno_aligned.csv"),
            vcf_path=str(soy / "soysnp50k_a2_final.vcf"),
            trait_cols=SOY_TRAITS,
        ),
        "rice": DatasetSpec(
            name="rice",
            pgen_prefix=str(rice / "sativas413_msu7_final"),
            fasta_path=str(rice / "Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa"),
            # IID-keyed rice pheno is a Phase-1 follow-up (source table is NSFTVID-keyed).
            pheno_csv=str(rice / "rice_pheno_aligned.csv"),
            vcf_path=str(rice / "sativas413_msu7_final.vcf"),
        ),
        "arabidopsis": DatasetSpec(
            name="arabidopsis",
            pgen_prefix=str(arab / "arabidopsis_1001g_final"),
            fasta_path=str(arab / "Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa"),
            pheno_csv=str(arab / "arabidopsis_pheno_aligned.csv"),
            vcf_path=_vcf_for(arab / "arabidopsis_1001g_final"),
        ),
    }


def get_dataset(name: str) -> DatasetSpec:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(reg)}")
    return reg[name]


def list_datasets() -> list[str]:
    return sorted(_registry())

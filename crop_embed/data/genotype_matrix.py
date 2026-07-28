"""
crop_embed/data/genotype_matrix.py
----------------------------------
Extract a dense **additive-dosage SNP matrix** (samples × variants, values 0/1/2)
from a plink2 ``.pgen/.pvar/.psam`` fileset, for SNP-matrix models (RR-BLUP,
sklearn baselines). This is the "Path A" companion to the window/embedding path:
the loaders in ``vcf.py``/``pgen.py`` collapse genotypes to a per-SNP binary
alt-flag for sequence reconstruction; here we want the true additive count.

Missing genotypes are filled by ``impute``:
  "ref"  (default) → 0, i.e. homozygous reference. Deterministic, no leakage.
  "mean"           → per-variant mean over non-missing calls (a float).

Reads the binary ``.pgen`` via pgenlib (seconds on the 20k-sample soy set), the
same dependency ``pgen.py`` uses. Sample/variant order follow the fileset unless
an explicit ``samples`` list is given (then rows follow that order).
"""
from __future__ import annotations

import numpy as np

from crop_embed.data.pgen import _read_psam
from crop_embed.data.vcf import _parse_chrom


def _read_pvar_ids(pvar_path: str) -> list[str]:
    """Variant IDs from a .pvar, in file (=.pgen) order.

    Uses the ID column when present and not '.', else falls back to 'chrom:pos'.
    """
    ids: list[str] = []
    with open(pvar_path) as f:
        for ln in f:
            if ln.startswith("#"):        # ## meta lines and the #CHROM header
                continue
            parts = ln.split()
            chrom, pos, vid = parts[0], parts[1], parts[2]
            if vid and vid != ".":
                ids.append(vid)
            else:
                ids.append(f"{_parse_chrom(chrom)}:{pos}")
    return ids


def load_dosage_matrix(
    pgen_prefix: str,
    samples: list[str] | None = None,
    *,
    impute: str = "ref",
    dtype: type = np.float32,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Parameters
    ----------
    pgen_prefix : path to the fileset, with or without a trailing ``.pgen``; the
                  companion ``.pvar`` and ``.psam`` must sit alongside it.
    samples     : sample IDs (rows), in this order; None = all, in .psam order.
    impute      : missing-genotype fill, "ref" (→0) or "mean".
    dtype       : output dtype (default float32; use int8 with impute="ref" to
                  quarter the memory when a float matrix isn't needed).

    Returns
    -------
    X           : (n_samples, n_variants) additive dosages 0/1/2 (missing filled).
    sample_ids  : row order (== ``samples`` if given).
    variant_ids : column order (.pvar ID column, or 'chrom:pos').
    """
    if impute not in ("ref", "mean"):
        raise ValueError(f"impute must be 'ref' or 'mean', got {impute!r}")
    try:
        import pgenlib
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "load_dosage_matrix needs the 'Pgenlib' package (pip install Pgenlib)."
        ) from e

    prefix = str(pgen_prefix)
    if prefix.endswith(".pgen"):
        prefix = prefix[:-5]

    all_samples = _read_psam(prefix + ".psam")
    if samples is None:
        samples = list(all_samples)
    take = np.fromiter((all_samples.index(s) for s in samples),
                       dtype=np.intp, count=len(samples))

    variant_ids = _read_pvar_ids(prefix + ".pvar")
    n, V = len(samples), len(variant_ids)

    reader = pgenlib.PgenReader(str.encode(prefix + ".pgen"))
    if reader.get_variant_ct() != V:
        reader.close()
        raise ValueError(
            f".pgen has {reader.get_variant_ct()} variants but .pvar has {V}"
        )

    X = np.empty((n, V), dtype=dtype)
    buf = np.empty(reader.get_raw_sample_ct(), dtype=np.int8)  # 0/1/2, -9 = missing
    for v in range(V):
        reader.read(v, buf)
        col = buf[take].astype(dtype, copy=True)
        miss = col < 0
        if miss.any():
            if impute == "ref":
                col[miss] = 0
            else:  # mean
                obs = col[~miss]
                col[miss] = obs.mean() if obs.size else 0.0
        X[:, v] = col
    reader.close()

    return X, list(samples), variant_ids

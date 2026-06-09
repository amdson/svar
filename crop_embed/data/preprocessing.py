"""
crop_embed/data/preprocessing.py
---------------------------------
Phenotype and SNP preprocessing pipeline for the Rice Diversity 44K dataset.

Pipeline order
--------------
1. load_vcf_sparse       : VCF → scipy sparse binary matrix (samples × SNPs)
2. load_phenotypes       : TSV → DataFrame + trait-column list
3. align_samples         : match VCF samples to phenotype rows via NSFTVID
4. scale_phenotypes      : per-trait StandardScaler that skips NaNs
5. impute_phenotypes     : KNN imputation of remaining NaNs
6. reduce_snp_features   : TruncatedSVD dimensionality reduction
7. train_test_split_data : stratified-random 80/20 split (default)

Or call load_data() to run steps 1–6 in one shot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

_DATA_DIR = Path(__file__).resolve().parents[3] / "rice_data"

from crop_embed.data.coords import DEFAULT_VCF_PATH, DEFAULT_PHENO_PATH

_PHENO_ID_COLS = {"HybID", "NSFTVID"}

DEFAULT_N_SVD_COMPONENTS = 120
DEFAULT_TEST_SIZE        = 0.2
DEFAULT_RANDOM_STATE     = 42
DEFAULT_KNN_NEIGHBORS    = 10

def load_vcf_sparse(
    vcf_path: Path = DEFAULT_VCF_PATH,
) -> tuple[sp.csr_matrix, list[str], list[str], int]:
    """
    Parse a biallelic, homozygous VCF into a sparse binary matrix.

    Encoding
    --------
    ``0/0`` → 0  (reference)
    ``1/1`` → 1  (alt allele)
    ``./.`` → 0  (missing, treated as reference; counted separately)

    Returns
    -------
    X         : csr_matrix, shape (n_samples, n_snps), dtype uint8
    samples   : list[str]  — VCF sample IDs
    snp_ids   : list[str]  — SNP IDs from the ID column
    n_missing : int        — total ./. calls encoded as 0
    """
    _gt_map = {"0/0": 0, "1/1": 1, "./.": 0}
    samples: list[str] = []
    snp_ids: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    n_missing = 0

    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = fields[9:]
                continue

            snp_idx = len(snp_ids)
            snp_ids.append(fields[2])

            for samp_idx, gt in enumerate(fields[9:]):
                if gt == "./.":
                    n_missing += 1
                if _gt_map.get(gt, 0) == 1:
                    rows.append(samp_idx)
                    cols.append(snp_idx)

    X = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(samples), len(snp_ids)),
        dtype=np.uint8,
    )
    return X, samples, snp_ids, n_missing

def load_phenotypes(
    pheno_path: Path = DEFAULT_PHENO_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Read the phenotype TSV.

    Returns
    -------
    pheno_df   : raw DataFrame (all columns, NSFTVID cast to int)
    trait_cols : list of trait column names (everything except ID columns)
    """
    pheno_df = pd.read_csv(pheno_path, sep="\t")
    pheno_df["NSFTVID"] = pheno_df["NSFTVID"].astype(int)
    trait_cols = [c for c in pheno_df.columns if c not in _PHENO_ID_COLS]
    return pheno_df, trait_cols

def align_samples(
    X_vcf: sp.csr_matrix,
    vcf_samples: list[str],
    pheno_df: pd.DataFrame,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """
    Match VCF samples to phenotype rows using the NSFTVID embedded in sample
    names (suffix after the last ``_``, e.g. ``081215-A05_1`` → NSFTVID 1).

    Returns
    -------
    X_aligned    : sparse matrix reordered to match phenotype row order
    pheno_matched: phenotype DataFrame filtered & reordered to matched samples
    """
    vcf_nsftvid   = [int(s.rsplit("_", 1)[-1]) for s in vcf_samples]
    vcf_id_to_idx = {nid: i for i, nid in enumerate(vcf_nsftvid)}

    pheno_matched = (
        pheno_df[pheno_df["NSFTVID"].isin(vcf_id_to_idx)]
        .copy()
        .reset_index(drop=True)
    )

    row_order = [vcf_id_to_idx[nid] for nid in pheno_matched["NSFTVID"]]
    X_aligned = X_vcf[row_order, :]
    return X_aligned, pheno_matched

def align_targets_to_dataset(
    dataset,
    pheno_df: pd.DataFrame,
    trait_cols: list[str],
) -> "np.ndarray":
    """
    Build a (n_samples, n_traits) target array aligned to ``dataset.samples``
    order (the order the training loop sees), without reordering the dataset.

    Samples in the VCF that have no matching phenotype row get NaN — the
    training loss (masked_mse) skips those entries.

    Sample IDs are matched by NSFTVID — the integer suffix after the last
    underscore in dataset.samples[i] (e.g. ``081215-A05_1`` → NSFTVID 1).
    """
    nsftvids = [int(s.rsplit("_", 1)[-1]) for s in dataset.samples]
    Y_df = pheno_df.set_index("NSFTVID").reindex(nsftvids)[trait_cols]
    return Y_df.values.astype(float)

def scale_phenotypes(Y_raw: np.ndarray) -> np.ndarray:
    """
    Standardise each trait column independently, ignoring NaN entries.
    NaN positions are preserved as NaN in the output.
    """
    Y_scaled = np.full_like(Y_raw, np.nan, dtype=float)
    for j in range(Y_raw.shape[1]):
        col  = Y_raw[:, j].astype(float)
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            continue
        mu  = col[mask].mean()
        std = col[mask].std(ddof=0)
        Y_scaled[mask, j] = (col[mask] - mu) / (std if std > 0 else 1.0)
    return Y_scaled

def impute_phenotypes(
    Y_scaled: np.ndarray,
    n_neighbors: int = DEFAULT_KNN_NEIGHBORS,
) -> np.ndarray:
    """Fill NaN values in the scaled phenotype matrix using KNN imputation."""
    return KNNImputer(n_neighbors=n_neighbors).fit_transform(Y_scaled)

def reduce_snp_features(
    X: sp.csr_matrix,
    n_components: int = DEFAULT_N_SVD_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Reduce the sparse SNP matrix with TruncatedSVD."""
    svd       = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd


# ---------------------------------------------------------------------------
# Step 7 — Train / test split
# ---------------------------------------------------------------------------

def train_test_split_data(
    X_feat: np.ndarray,
    Y_scaled: np.ndarray,
    Y_imputed: np.ndarray,
    test_size: float    = DEFAULT_TEST_SIZE,
    random_state: int   = DEFAULT_RANDOM_STATE,
) -> tuple:
    """
    Split features and both label arrays into train / test sets.

    Returns
    -------
    X_train, X_test,
    Y_train, Y_test,             (scaled, NaNs present — for evaluation masking)
    Y_train_imputed, Y_test_imputed  (no NaNs — for model fitting)
    """
    return train_test_split(
        X_feat, Y_scaled, Y_imputed,
        test_size=test_size,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Convenience: run the full pipeline in one call
# ---------------------------------------------------------------------------

def load_data(
    vcf_path: Path          = DEFAULT_VCF_PATH,
    pheno_path: Path        = DEFAULT_PHENO_PATH,
    n_svd_components: int   = DEFAULT_N_SVD_COMPONENTS,
    knn_neighbors: int      = DEFAULT_KNN_NEIGHBORS,
    random_state: int       = DEFAULT_RANDOM_STATE,
) -> dict:
    """
    Run the full preprocessing pipeline and return a results dict.

    Keys
    ----
    X_sparse, X_svd, svd, Y_scaled, Y_imputed,
    trait_cols, pheno_matched, snp_ids, vcf_samples, n_missing_gt
    """
    X_vcf, vcf_samples, snp_ids, n_missing = load_vcf_sparse(vcf_path)
    pheno_df, trait_cols                   = load_phenotypes(pheno_path)
    X_sparse, pheno_matched                = align_samples(X_vcf, vcf_samples, pheno_df)

    Y_raw     = pheno_matched[trait_cols].values.astype(float)
    Y_scaled  = scale_phenotypes(Y_raw)
    Y_imputed = impute_phenotypes(Y_scaled, n_neighbors=knn_neighbors)
    X_svd, svd = reduce_snp_features(
        X_sparse, n_components=n_svd_components, random_state=random_state
    )

    return {
        "X_sparse":     X_sparse,
        "X_svd":        X_svd,
        "svd":          svd,
        "Y_scaled":     Y_scaled,
        "Y_imputed":    Y_imputed,
        "trait_cols":   trait_cols,
        "pheno_matched": pheno_matched,
        "snp_ids":      snp_ids,
        "vcf_samples":  vcf_samples,
        "n_missing_gt": n_missing,
    }

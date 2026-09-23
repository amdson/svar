"""training_meth — frozen-feature ("embedding style") learning on the cross-accession
arabidopsis methylation benchmark (ath_meth_benchmark).

Pipeline: extract_features (GPU, one pass per backbone) -> fit_sites (CPU, per-site
ridge on held-out accessions, genotype bar, permutation null, cross-site transfer).
See STATE.md.
"""

"""
training/sweeps/arabidopsis_baselines.py
----------------------------------------
Baseline sweep on arabidopsis (1001 Genomes / TAIR10): all sklearn baseline
models over the 20 best-covered AraPheno traits.

Unlike soy, arabidopsis has ~2.28M variants — a dense SNP matrix is infeasible —
so EVERY model runs on the raw sparse matrix reduced by TruncatedSVD-500 (ridge
and pls included). Sample count is small (~1135 accessions), so the kernel/tree
models are cheap here; the up-front cost is the sparse matrix build.

Traits are the top-20 by post-collapse coverage (see build_arabidopsis_aligned.py,
which writes the one-row-per-IID arabidopsis_pheno_aligned.csv this reads through).

    python -m training.sweep --config training/sweeps/arabidopsis_baselines.py --dry-run
    python -m training.sweep --config training/sweeps/arabidopsis_baselines.py
"""

# Top-20 AraPheno phenotype ids by coverage (accessions with a value: 512–1003).
TRAITS = "261,262,703,701,702,705,704,700,706,707,535,533,532,523,526,525,528,538,527,529"

baselines = {
    "runner": "snp_sklearn",
    "name": "svd500",
    "fixed": {"dataset": "arabidopsis", "traits": TRAITS,
              "seed": 42, "sparse": True, "svd": 500},
    "grid": {"model": ["ridge", "pls", "krr", "svr", "rf", "gbm"]},
}

SWEEP = [baselines]

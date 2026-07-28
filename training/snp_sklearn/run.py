"""
training/snp_sklearn/run.py
---------------------------
sklearn suite on the additive SNP-dosage matrix.

    python -m training.snp_sklearn.run --dataset soy --model ridge --traits protein,oil
    python -m training.snp_sklearn.run --dataset soy --model rf --svd 500
"""
from __future__ import annotations

import argparse

from training.common import harness
from training.snp_sklearn.estimators import add_sklearn_args, make_estimator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    harness.add_common_args(p)
    add_sklearn_args(p)
    args = p.parse_args()
    harness.run_sklearn("snp", lambda: make_estimator(args.model, args), args)


if __name__ == "__main__":
    main()

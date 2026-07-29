"""
training/emb_sklearn/run.py
---------------------------
Same sklearn suite as snp_sklearn (ridge / svr-RBF / krr-RBF / rf / gbm / pls),
but on per-sample POOLED fixed embeddings.
Requires a window-embedding cache (see train_pipeline/embed_windows.py); address
it by --backbone/--half-window or pass --cache directly.

    python -m training.emb_sklearn.run --dataset soy --model ridge \
        --backbone carbon500m --half-window 500
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
    harness.run_sklearn("emb", lambda: make_estimator(args.model, args), args)


if __name__ == "__main__":
    main()

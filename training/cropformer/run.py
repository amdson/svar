"""
training/cropformer/run.py
--------------------------
Cropformer (CNN + self-attention) genomic-prediction baseline on the additive SNP
matrix, wired into the shared harness so it is directly comparable to snp_sklearn:
same 70/15/15 split, one trait at a time, val + test Pearson, run record.

    python -m training.cropformer.run --dataset wheat --traits protein
    python -m training.cropformer.run --dataset soy   --mic-k 10000 --max-epochs 100
    python -m training.cropformer.run --dataset rice  --mic-k 0      # use all SNPs

The model label is fixed to 'cropformer' (single architecture); --mic-k, --lr,
--num-heads, --hidden-dim, --max-epochs, etc. tune it (see estimator.py).
"""
from __future__ import annotations

import argparse

from training.common import harness
from training.cropformer.estimator import add_cropformer_args, make_estimator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    harness.add_common_args(p)
    add_cropformer_args(p)
    # single architecture: default --model to 'cropformer' so it needn't be passed.
    for a in p._actions:
        if a.dest == "model":
            a.required = False
            a.default = "cropformer"
    args = p.parse_args()
    harness.run_sklearn("snp", lambda: make_estimator(args), args)


if __name__ == "__main__":
    main()

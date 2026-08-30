"""
variant_ll/run.py
-----------------
Runner for the allele log-likelihood benchmark: build windows, score B0/B1/B2,
evaluate Carbon through the variant cache, fine-tune, and re-evaluate — all in
**bits per SNP**, so the model and the baselines are directly comparable.

All three experiments BENCHMARK.md calls for are this one runner with different
flags; they share a loop (``variant_ll/harness.py``) so their numbers stay
comparable by construction rather than by luck.

    # 1. The benchmark — held-out accessions, the question the module exists to answer.
    python -m variant_ll.run --dataset arabidopsis --chrom 4 --half-window 500 \
        --windows 200 --epochs 80 --eval-accessions heldout --permute

    # 2. Capacity probe — same accessions the model trained on. Pure fitting, no
    #    generalisation; the baselines are refit on those accessions too.
    python -m variant_ll.run --windows 20 --epochs 75 --eval-accessions insample

    # 3. Approximation cost — (2) again through the exact full forward. The
    #    cache-vs-exact gap is BENCHMARK.md §7's control.
    python -m variant_ll.run --windows 20 --epochs 75 --eval-accessions insample \
        --backend exact --hap-chunk 8

Sweeps over these axes live in ``training/sweeps/variant_ll.py``:

    python -m training.sweep --config training/sweeps/variant_ll.py --gpus 0,2

See BENCHMARK.md for the objective and the decision matrix, PLAN.md for the run
sequence.
"""
from __future__ import annotations

import argparse
import sys

from variant_ll import harness


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    harness.add_common_args(p)
    return p


def main() -> int:
    args = build_parser().parse_args()
    out = harness.run(args)
    return 1 if out.get("status") == "empty" else 0


if __name__ == "__main__":
    sys.exit(main())

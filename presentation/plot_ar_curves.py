"""
presentation/plot_ar_curves.py
------------------------------
Goal 5 (plot). Train + val log-likelihood on SNP tokens vs. step/epoch for the
Carbon AR model fine-tuned through the variant cache, read from the JSONL sidecar
that variant_ar/train.py now writes next to its --output.

Generate the run (needs GPU — full Carbon-500M fwd+bwd through the variant cache;
--val-frac 0.1 = the "90% subset", --backend cache = the variant-cache method):

    /home/andrew.dickson/.conda/envs/svar/bin/python variant_ar/train.py \
        --backend cache --half-window 500 --buffer 0 --max-length 2048 \
        --val-frac 0.1 --epochs 5 --batch-size 4 --lr 1e-5 --precision fp32 \
        --output checkpoints/variant_ar/carbon_vc.pt

Then:

    /home/andrew.dickson/.conda/envs/svar/bin/python presentation/plot_ar_curves.py \
        --metrics checkpoints/variant_ar/carbon_vc.metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def load_metrics(path):
    train, val = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (val if any(k.startswith("val/") for k in r) else train).append(r)
    return train, val


def series(rows, key, x="step"):
    xs = [r[x] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    return xs, ys


def main() -> int:
    p = argparse.ArgumentParser(description="Plot Carbon AR variant-cache fine-tune curves.")
    p.add_argument("--metrics", required=True,
                   help="JSONL sidecar from variant_ar/train.py (…/model.metrics.jsonl).")
    p.add_argument("--out", default=str(HERE / "figs" / "05_ar_curves.png"))
    args = p.parse_args()

    if not Path(args.metrics).exists():
        print(f"{args.metrics} not found — run variant_ar/train.py first (needs GPU).")
        return 1

    tr, va = load_metrics(args.metrics)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_ll, ax_ppl) = plt.subplots(1, 2, figsize=(13, 4.6))

    # Log-likelihood on SNP tokens (higher = better); LL = -NLL.
    ax_ll.plot(*series(tr, "train/ll"), label="train LL", alpha=.75)
    ax_ll.plot(*series(va, "val/ll"), "o-", label="val LL")
    ax_ll.set_xlabel("step")
    ax_ll.set_ylabel("log-likelihood on SNP tokens (−NLL)")
    ax_ll.set_title("Carbon AR fine-tune via variant cache (90% train)")
    ax_ll.legend(); ax_ll.grid(alpha=.3)

    # Validation perplexity per epoch (lower = better).
    xs, ys = series(va, "val/perplexity")
    ax_ppl.plot(xs, ys, "s-", color="tab:red")
    ax_ppl.set_xlabel("step")
    ax_ppl.set_ylabel("val perplexity (SNP tokens)")
    ax_ppl.set_title("Validation perplexity")
    ax_ppl.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

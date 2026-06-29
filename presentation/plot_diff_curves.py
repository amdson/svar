"""
presentation/plot_diff_curves.py
--------------------------------
Goal 2 (plot). Regenerate figs/02_diff_curves.png and figs/02_per_trait_pcc.png
from the head-training metrics JSONL sidecars of the three window-aggregation
diffs (absolute / centered / refdelta). Mirrors the recipe in PLOTS.md §2.

    python presentation/plot_diff_curves.py \\
        --absolute trained_heads/repro/absolute/model.metrics.jsonl \\
        --centered trained_heads/repro/centered/model.metrics.jsonl \\
        --refdelta trained_heads/repro/refdelta/model.metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TRAITS = ["Seed length", "Flowering time at Arkansas", "Amylose content"]


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
    p = argparse.ArgumentParser(description="Plot head-training aggregation-diff curves.")
    p.add_argument("--absolute", required=True)
    p.add_argument("--centered", required=True)
    p.add_argument("--refdelta", required=True)
    p.add_argument("--out-curves", default=str(HERE / "figs" / "02_diff_curves.png"))
    p.add_argument("--out-traits", default=str(HERE / "figs" / "02_per_trait_pcc.png"))
    args = p.parse_args()

    runs = {"absolute": args.absolute, "centered": args.centered,
            "refdelta": args.refdelta}
    for label, path in runs.items():
        if not Path(path).exists():
            print(f"{path} not found — run train_head.py for '{label}' first.")
            return 1

    # ── Mean MSE/PCC, train + val ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("train/mean_mse", axes[0, 0], "train MSE"),
              ("val/mean_mse",   axes[0, 1], "val MSE"),
              ("train/mean_pcc", axes[1, 0], "train PCC"),
              ("val/mean_pcc",   axes[1, 1], "val PCC")]
    for label, path in runs.items():
        tr, va = load_metrics(path)
        for key, ax, title in panels:
            rows = va if key.startswith("val/") else tr
            ax.plot(*series(rows, key), label=label)
            ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=.3)
    axes[0, 0].legend()
    fig.suptitle("Effect of window-aggregation diff (mlp / mean / ws1, vc hw500)")
    fig.tight_layout()
    fig.savefig(args.out_curves, dpi=150)
    print(f"Wrote {args.out_curves}")

    # ── Per-trait val PCC ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(5 * len(TRAITS), 4),
                             squeeze=False)
    for j, trait in enumerate(TRAITS):
        ax = axes[0][j]
        for label, path in runs.items():
            _, va = load_metrics(path)
            ax.plot(*series(va, f"val/pcc/{trait}"), label=label)
        ax.set_title(f"val PCC — {trait}"); ax.set_xlabel("step"); ax.grid(alpha=.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(args.out_traits, dpi=150)
    print(f"Wrote {args.out_traits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

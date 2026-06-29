"""
presentation/plot_window_sweep.py
---------------------------------
Goal 1 (plot). Draw the three curves from count_windows.py's CSV:
unique windows, mean tokens per window, and unique variants per window, all vs
window length on a log x-axis.

    python presentation/count_windows.py        # produces window_sweep.csv
    python presentation/plot_window_sweep.py     # -> figs/01_window_sweep.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent


def main() -> int:
    p = argparse.ArgumentParser(description="Plot the window-length sweep.")
    p.add_argument("--csv", default=str(HERE / "window_sweep.csv"))
    p.add_argument("--out", default=str(HERE / "figs" / "01_window_sweep.png"))
    args = p.parse_args()

    df = pd.read_csv(args.csv).sort_values("length")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("n_unique_windows", "Unique windows in dataset", "# unique windows"),
        ("mean_tokens", "Mean tokens per window", "mean tokens (Carbon 6-mer)"),
        ("variants_per_window", "Unique variants per window", "unique fps / window"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, title, ylab) in zip(axes, panels):
        sub = df.dropna(subset=[col])
        ax.plot(sub["length"], sub[col], "o-")
        ax.set_xscale("log")
        ax.set_xlabel("window length (bp)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=.3, which="both")
    fig.suptitle("Window-length sweep (sativas413)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

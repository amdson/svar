"""Build presentation/forward_calls.ipynb — full forward passes across the three
stages full -> dedup -> variant cache. No nbformat / data inputs."""
import json
from pathlib import Path

def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in text.strip("\n").split("\n")]}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in text.strip("\n").split("\n")]}

cells = []

cells.append(md(r"""
# Full forward passes: full → dedup → variant cache

Each stage strictly reduces the number of **full** language-model forward passes
needed to embed the dataset (counts from the `carbon500m_hw500` cache):

| stage | what gets a full forward | count |
|---|---|---|
| **full** | every `(sample, window)` pair | `n_samples × n_windows` |
| **dedup** | every unique fingerprint (reference + variant) | `n_unique` |
| **variant cache** | the reference window per window only | `n_windows` |

Dedup drops the repeats (exact); the variant cache then replaces each variant
fingerprint's full forward with a cheap **partial recompute** of just its SNP
tokens, leaving only one reference full-forward per window.
"""))

cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
%matplotlib inline

FIGS = Path("/home/andrew.dickson/svar/presentation/figs"); FIGS.mkdir(parents=True, exist_ok=True)

# Real counts from the hw500 cache metadata.
N_SAMPLES, N_WINDOWS, N_UNIQUE = 383, 23138, 53398
full  = N_SAMPLES * N_WINDOWS      # one forward per (sample, window)
dedup = N_UNIQUE                   # one forward per unique fingerprint
vc    = N_WINDOWS                  # one full forward (reference) per window
var_fp = N_UNIQUE - N_WINDOWS      # variant fingerprints -> partial recomputes

stages = ["full\n(every sample × window)",
          "dedup\n(unique fingerprints)",
          "variant cache\n(reference per window)"]
vals   = [full, dedup, vc]
colors = ["#c0392b", "#e8833a", "#2e7d4f"]

fig, ax = plt.subplots(figsize=(12, 5.2))
ypos = [2, 1, 0]                                   # full on top
bars = ax.barh(ypos, vals, color=colors, height=0.6)
ax.set_yticks(ypos); ax.set_yticklabels(stages, fontsize=11)
ax.set_xscale("log")
ax.set_xlim(vc / 3, full * 6)
ax.set_xlabel("number of FULL forward passes (log scale)")

for y, v in zip(ypos, vals):
    ax.text(v * 1.18, y, f"{v:,}", va="center", fontsize=12, fontweight="bold")

def reduction(y_hi, y_lo, v_hi, v_lo, text):
    xm = np.sqrt(v_hi * v_lo)                      # geometric midpoint on log axis
    ax.annotate("", xy=(xm, y_lo + 0.30), xytext=(xm, y_hi - 0.30),
                arrowprops=dict(arrowstyle="-|>", color="#33414f", lw=2))
    ax.text(xm * 1.25, (y_hi + y_lo) / 2, text, va="center", fontsize=11,
            fontweight="bold", color="#33414f")

reduction(2, 1, full, dedup, f"÷ {full/dedup:.0f}\n(drop repeats, exact)")
reduction(1, 0, dedup, vc, f"÷ {dedup/vc:.1f}\n(variants → partial)")

ax.text(full * 0.9, -0.62,
        f"{full/vc:.0f}× fewer full forwards than the naïve pass",
        ha="right", fontsize=12, color="#2e7d4f", fontweight="bold")
ax.text(vc / 2.8, -0.95,
        f"variant cache turns {var_fp:,} variant-fingerprint forwards into partial "
        f"recomputes (only the SNP tokens, cost ∝ T + B·V)",
        ha="left", va="top", fontsize=9, color="#566573")

ax.grid(axis="x", alpha=.3, which="both")
ax.set_ylim(-1.3, 2.7)
ax.set_title("Full forward passes per stage:  full → dedup → variant cache", fontsize=13)
fig.tight_layout()
fig.savefig(FIGS / "10_forward_calls_stages.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(md(r"""
**Reading it.** Going `full → dedup` is an *exact* speedup — the deduplicated
sequences are identical, so the embeddings are unchanged. Going `dedup → variant
cache` is an *approximate* speedup on the multi-SNP windows: instead of a full
forward per variant fingerprint, only its SNP tokens are recomputed against the
cached reference (see `vc_explainer.ipynb`), leaving one reference full-forward
per window.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (svar)", "language": "python",
                        "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).parent / "forward_calls.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")

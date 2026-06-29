"""Build presentation/fingerprint_explainer.ipynb — how unique fingerprints are
derived from windowed genomes, and why it's fast. No nbformat / data inputs."""
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
# Deriving unique fingerprints from windowed genomes

Every sample shares the same reference genome and differs only at SNPs, so a
window's DNA sequence for a sample is **fully determined** by which SNPs in that
window it carries the alt allele at. We encode that as a **fingerprint**:

```
fingerprint = (chrom, window_start, window_end, alt_positions)
```

Two `(sample, window)` pairs with the same fingerprint have the *identical*
sequence — so they get embedded **once**. The expensive step is the language-model
forward pass per sequence; deduplicating before embedding turns
`n_samples × n_windows` forwards into just `n_unique_fingerprints` forwards, then
per-sample vectors are reassembled by cheap table lookups. The result is exact
(not an approximation).

The numbers below are from the `carbon500m_hw500` cache.
"""))

cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path
%matplotlib inline

FIGS = Path("/home/andrew.dickson/svar/presentation/figs"); FIGS.mkdir(parents=True, exist_ok=True)

# Real counts from the hw500 cache metadata (refresh via:
#   torch.load(cache, weights_only=False)["metadata"] -> n_samples/n_windows/n_unique_windows)
N_SAMPLES, N_WINDOWS, N_UNIQUE = 383, 23138, 53398
N_PAIRS = N_SAMPLES * N_WINDOWS

C_SHARED = "#e7edf3"; C_EDGE = "#9aa7b4"
# one colour per distinct fingerprint group
GROUP = {"ref": "#9dc3e6", "A": "#e8833a", "B": "#7fb285", "C": "#b58fd0"}

def cell(ax, x, y, w=1.0, h=1.0, color=C_SHARED, lw=0.8, ec=C_EDGE):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=ec, lw=lw))

def chip(ax, x, y, w, h, text, fc, fs=11, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                 facecolor=fc, edgecolor=ec or "#5b6b7b", lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

def arrow(ax, a, b, color="#33414f", lw=1.8, style="-|>"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color,
                                                    lw=lw, shrinkA=2, shrinkB=2))
print("ready — {:,} samples × {:,} windows = {:.2f}M pairs  ->  {:,} unique  ({:.0f}x)".format(
    N_SAMPLES, N_WINDOWS, N_PAIRS/1e6, N_UNIQUE, N_PAIRS/N_UNIQUE))
"""))

# ── Figure A ──
cells.append(md(r"""
## 1. One window: samples → fingerprints

Within a single window with three SNP sites, each sample is a pattern of
reference (grey) / alt (orange) calls. The fingerprint is just the **set of alt
positions**, so samples with the same pattern collapse to the same fingerprint —
here 6 samples → 3 unique sequences to embed.
"""))
cells.append(code(r"""
# sample x SNP-site alt mask (1 = alt). Site indices shown to the reader are 1-based.
samples = ["sample 1", "sample 2", "sample 3", "sample 4", "sample 5", "sample 6"]
mask = np.array([
    [0, 0, 0],   # {}
    [0, 1, 0],   # {2}
    [0, 1, 0],   # {2}
    [1, 0, 1],   # {1,3}
    [1, 0, 1],   # {1,3}
    [0, 0, 0],   # {}
])
def fp_label(row):
    pos = [str(j+1) for j, v in enumerate(row) if v]
    return "{" + ",".join(pos) + "}" if pos else "{ }  (reference)"
def fp_key(row):
    return tuple(np.nonzero(row)[0].tolist())
GROUP_OF = {(): "ref", (1,): "A", (0, 2): "B"}  # 0-based alt position tuples

fig, ax = plt.subplots(figsize=(12, 5))
n = len(samples)
for i, name in enumerate(samples):
    y = n - 1 - i
    for j in range(3):
        cell(ax, j, y, color=(GROUP["A"] if mask[i, j] else C_SHARED))
    ax.text(-0.3, y + 0.5, name, ha="right", va="center", fontsize=10)
    # fingerprint chip per sample, coloured by its group
    g = GROUP_OF[fp_key(mask[i])]
    chip(ax, 4.4, y + 0.06, 3.4, 0.88, fp_label(mask[i]), GROUP[g], fs=11)
    arrow(ax, (3.15, y + 0.5), (4.35, y + 0.5))
ax.text(1.5, n + 0.35, "SNP sites in window  (1   2   3)", ha="center", fontsize=10)
ax.text(6.1, n + 0.35, "fingerprint = set of alt positions", ha="center", fontsize=10)

# collapse to the unique set
ux = 9.2
ax.text(ux + 1.4, n + 0.35, "unique fingerprints", ha="center", fontsize=10, color="#c0392b")
uniq = [("ref", "{ }"), ("A", "{2}"), ("B", "{1,3}")]
for k, (g, lab) in enumerate(uniq):
    chip(ax, ux, (n - 1.5) - 1.6*k, 2.8, 0.9, lab, GROUP[g], fs=12)
ax.text(ux + 1.4, (n - 1.5) - 1.6*len(uniq) + 0.2,
        "3 sequences embedded\n(not 6)", ha="center", va="top", fontsize=10,
        color="#c0392b", fontweight="bold")
arrow(ax, (7.95, n/2.0), (ux - 0.15, n/2.0), color="#c0392b")

ax.set_xlim(-2.4, 12.6); ax.set_ylim(-1.2, n + 0.9)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Fingerprinting one window: identical patterns collapse", fontsize=13)
fig.tight_layout(); fig.savefig(FIGS / "08_fingerprint_window.png", dpi=150,
                                bbox_inches="tight"); plt.show()
"""))

# ── Figure B ──
cells.append(md(r"""
## 2. Why it's fast: dedup across the whole dataset

Across all windows, almost every `(sample, window)` cell is the **reference
fingerprint** (no alts) — shared by every sample. So the number of *distinct*
sequences is tiny next to the number of instances, and we embed each distinct
sequence only once.
"""))
cells.append(code(r"""
fig, (axm, axb) = plt.subplots(1, 2, figsize=(15, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.1]})

# ---- left: the fingerprint matrix, mostly the shared reference fingerprint ----
rng = np.random.default_rng(3)
R, Cc = 12, 18
axm.set_title("(sample × window) fingerprint matrix", fontsize=12)
for i in range(R):
    for j in range(Cc):
        # ~12% of cells carry a variant fingerprint (coloured); rest are reference (grey)
        if rng.random() < 0.12:
            col = rng.choice([GROUP["A"], GROUP["B"], GROUP["C"]])
        else:
            col = C_SHARED
        cell(axm, j, R - 1 - i, color=col)
axm.text(Cc/2, R + 0.3, "← n_windows →", ha="center", fontsize=10)
axm.text(-0.6, R/2, "← n_samples →", ha="center", va="center", rotation=90, fontsize=10)
axm.text(Cc/2, -0.8, "most cells = reference fingerprint (grey), shared across all samples",
         ha="center", va="top", fontsize=9.5, color="#566573")
axm.set_xlim(-1.6, Cc + 0.4); axm.set_ylim(-1.4, R + 0.9)
axm.set_aspect("equal"); axm.axis("off")

# ---- right: forward-pass count, naive vs dedup (log scale) ----
labels = ["naïve\n(every sample × window)", "dedup\n(unique fingerprints)"]
vals = [N_PAIRS, N_UNIQUE]
colors = ["#c0392b", "#2e7d4f"]
bars = axb.barh(labels, vals, color=colors, height=0.55)
axb.set_xscale("log")
axb.set_xlabel("language-model forward passes (log scale)")
axb.set_title("Forwards to embed the dataset", fontsize=12)
for b, v in zip(bars, vals):
    axb.text(v*1.15, b.get_y() + b.get_height()/2,
             f"{v:,}", va="center", fontsize=11, fontweight="bold")
axb.set_xlim(N_UNIQUE/3, N_PAIRS*6)
axb.annotate(f"{N_PAIRS/N_UNIQUE:.0f}× fewer forwards\n(exact — same sequences, embedded once)",
             xy=(N_UNIQUE, 1), xytext=(N_UNIQUE*1.1, 0.55), fontsize=11, color="#2e7d4f",
             fontweight="bold", va="center")
axb.grid(axis="x", alpha=.3, which="both")

fig.suptitle("Deduplication: embed each unique window once, look up the rest", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "09_fingerprint_dedup_scale.png", dpi=150, bbox_inches="tight"); plt.show()
"""))

cells.append(md(r"""
**The contrast worth stating:** this dedup is *exact* — the embedded sequences are
literally identical, so pooling per-sample vectors from the table gives the same
answer as embedding every `(sample, window)` pair, just ~166× cheaper. (That is
distinct from the *variant cache*, which approximates the forward pass for windows
that do carry several SNPs — see `vc_explainer.ipynb`.)
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
out = Path(__file__).parent / "fingerprint_explainer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")

"""Build presentation/vc_explainer.ipynb — a matplotlib explainer for the
variant-cache forward pass (no nbformat dependency, no data inputs)."""
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
# How the variant-cache forward pass works

The samples in a cohort differ from the reference genome only at SNPs. After
tokenization, a window's token sequence is therefore **almost identical across
samples** — only the few tokens whose 6-mer span contains a SNP change. The
variant cache exploits this:

- **Standard forward** runs every sample's full sequence through every layer →
  work ∝ `B · T · L`  (B = distinct haplotypes, T = tokens/window, L = layers).
- **Variant-cache forward** runs the reference window **once**, caches its
  per-layer hidden states, then recomputes **only the differing tokens** for each
  sample — each variant query attends to *{cached reference context} ∪ {variant
  tokens}* (a single causal softmax, done as two attentions merged with the
  log-sum-exp trick). Work ∝ `(T + B · V) · L`, with `V ≪ T`.

The two figures below show (1) how a batch decomposes into shared vs differing
tokens, and (2) the two forward passes side by side.
"""))

# ── shared drawing helpers ──
cells.append(code(r"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
%matplotlib inline

REPO = Path("/home/andrew.dickson/svar")
FIGS = REPO / "presentation" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# palette
C_SHARED = "#e7edf3"   # token identical to reference across all samples
C_REF    = "#9dc3e6"   # SNP position, sample carries the reference allele
C_ALT    = "#e8833a"   # SNP position, sample carries the alt allele
C_EDGE   = "#9aa7b4"

def cell(ax, x, y, w=1.0, h=1.0, color=C_SHARED, lw=0.8, ec=C_EDGE):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=ec, lw=lw))

def box(ax, x, y, w, h, text, fc="#f4f6f8", ec="#5b6b7b", fs=11, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=fc, edgecolor=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

def arrow(ax, xy_from, xy_to, color="#33414f", lw=2.0, style="-|>"):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=2, shrinkB=2))
print("helpers ready")
"""))

# ── Figure 1: batch decomposition ──
cells.append(md(r"""
## 1. Decomposing a batch into shared vs differing tokens

Rows are the reference plus a few sample haplotypes; columns are token positions
in one window. Almost every column is **shared** (grey — identical to reference
for every sample). Only the **variant positions** differ, and even there a sample
is either reference (blue) or alt (orange). Those orange/blue columns are the only
tokens the cache has to recompute.
"""))
cells.append(code(r"""
T = 16
rows = ["reference", "sample 1", "sample 2", "sample 3", "sample 4", "sample 5"]
snp_cols = [3, 8, 12]
rng = np.random.default_rng(1)
# per-sample alt/ref at each SNP column (reference row is always ref)
alt = {c: rng.integers(0, 2, len(rows) - 1) for c in snp_cols}

fig, ax = plt.subplots(figsize=(12, 4.6))
n = len(rows)
for i, name in enumerate(rows):
    y = n - 1 - i
    for t in range(T):
        if t in snp_cols and name != "reference":
            color = C_ALT if alt[t][i - 1] == 1 else C_REF
        elif t in snp_cols:           # reference row at a SNP column -> ref allele
            color = C_REF
        else:
            color = C_SHARED
        cell(ax, t, y, color=color)
    ax.text(-0.4, y + 0.5, name, ha="right", va="center", fontsize=10)

# bracket + label the variant columns
for t in snp_cols:
    ax.add_patch(Rectangle((t, -0.15), 1, n + 0.15, fill=False,
                           edgecolor="#c0392b", lw=1.6, ls=(0, (4, 2))))
ax.text(np.mean(snp_cols) + 0.5, n + 0.45,
        "variant token positions — differ across samples (recomputed)",
        ha="center", va="bottom", fontsize=10, color="#c0392b")
ax.text((T - len(snp_cols)) / 2 - 1.5, -0.7,
        "all other tokens are shared (identical to reference) — computed once",
        ha="center", va="top", fontsize=10, color="#566573")

# legend
handles = [Rectangle((0, 0), 1, 1, fc=c, ec=C_EDGE) for c in (C_SHARED, C_REF, C_ALT)]
ax.legend(handles, ["shared (= reference)", "SNP token, ref allele", "SNP token, alt allele"],
          loc="upper left", bbox_to_anchor=(0.0, -0.04), ncol=3, frameon=False, fontsize=9)

ax.set_xlim(-3, T + 0.5); ax.set_ylim(-1.4, n + 1.0)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Anatomy of a window batch: shared tokens + a few differing tokens", fontsize=13)
fig.tight_layout(); fig.savefig(FIGS / "06_vc_batch_decomposition.png", dpi=150,
                                bbox_inches="tight"); plt.show()
"""))

# ── Figure 2: standard vs variant-cache forward ──
cells.append(md(r"""
## 2. Standard vs variant-cache forward pass

**Standard** pushes all `B · T` tokens through every layer. **Variant-cache**
forwards the reference once into a per-layer cache, then recomputes only the
differing tokens, which attend back into the cached reference context.
"""))
cells.append(code(r"""
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))

def token_grid(ax, x, y, r, c, colors, cw=0.32, ch=0.32, gap=0.04):
    for i in range(r):
        for j in range(c):
            col = colors(i, j)
            ax.add_patch(Rectangle((x + j*(cw+gap), y + i*(ch+gap)), cw, ch,
                                   facecolor=col, edgecolor=C_EDGE, lw=0.5))

# ---- Standard ----
axL.set_title("Standard forward — recompute everything", fontsize=12)
def std_color(i, j):
    return C_ALT if j in (2, 5) and i in (1, 3) else (C_REF if j in (2, 5) else C_SHARED)
GX = 1.46  # center the 8-col grids (step 0.36) under the layer box at x=2.9
token_grid(axL, GX, 5.5, 5, 8, std_color)
axL.text(2.9, 7.5, "batch:  B haplotypes × T tokens", ha="center", fontsize=10)
box(axL, 1.2, 3.4, 3.4, 1.6, "Transformer × L layers\n(every token, every layer)",
    fc="#eaf0f6")
arrow(axL, (2.9, 5.45), (2.9, 5.05))
token_grid(axL, GX, 1.2, 5, 8, std_color)
arrow(axL, (2.9, 3.35), (2.9, 3.05))
axL.text(2.9, 0.55, "work  ∝  B · T · L", ha="center", fontsize=13, color="#c0392b",
         fontweight="bold")
axL.set_xlim(0, 6); axL.set_ylim(0, 8); axL.axis("off")

# ---- Variant cache ----
axR.set_title("Variant-cache forward — reference once + variant deltas", fontsize=12)
# reference lane
token_grid(axR, 0.4, 6.7, 1, 8, lambda i, j: C_SHARED if j not in (2, 5) else C_REF)
axR.text(0.4 + 4*0.36, 6.7 + 0.55, "reference window (1 × T)", ha="center", fontsize=9)
box(axR, 4.0, 6.55, 2.0, 0.9, "Transformer × L", fc="#eaf0f6", fs=9)
arrow(axR, (3.35, 6.86), (4.0, 6.95))
box(axR, 6.6, 6.2, 2.9, 1.6,
    "cached reference\nhidden states\n(all positions, every layer)", fc="#e7f3ea",
    ec="#3b8f5a", fs=9)
arrow(axR, (6.0, 7.0), (6.6, 7.0))
# variant lane
token_grid(axR, 0.4, 3.1, 5, 2, lambda i, j: C_ALT, cw=0.34, ch=0.30)
axR.text(0.4 + 0.36, 3.1 + 5*0.34 + 0.15, "differing tokens only\n(B × V,  V ≪ T)",
         ha="center", fontsize=9)
box(axR, 3.2, 3.3, 2.4, 1.4, "Transformer × L\n(variant tokens)", fc="#fdeee2",
    ec="#cf7a3a", fs=9)
arrow(axR, (1.4, 3.6), (3.2, 4.0))
# cache feeds the variant attention
arrow(axR, (8.0, 6.15), (5.0, 4.75), color="#3b8f5a", lw=1.8, style="-|>")
axR.text(7.2, 5.4, "attend to cached\nref context\n(log-sum-exp merge)",
         ha="center", va="center", fontsize=8.5, color="#2e7d4f")
# output
box(axR, 6.7, 3.4, 2.8, 1.2, "variant hidden\nstates / logits", fc="#f4f6f8", fs=9)
arrow(axR, (5.6, 4.0), (6.7, 4.0))
axR.text(4.8, 1.4, "work  ∝  (T  +  B · V) · L", ha="center", fontsize=13,
         color="#2e7d4f", fontweight="bold")
axR.text(4.8, 0.6, "reference computed once · only the few SNP tokens redone per sample",
         ha="center", fontsize=9, color="#566573")
axR.set_xlim(0, 9.8); axR.set_ylim(0, 8); axR.axis("off")

fig.suptitle("Standard vs variant-cache forward pass", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGS / "07_vc_forward_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(md(r"""
**Why the merge, not a concatenation.** A variant query at position `p` needs a
single softmax over *all* real keys ≤ p. Those split cleanly into the cached
reference keys (non-variant positions) and the variant keys, so the cache runs
two attentions and combines them with log-sum-exp — avoiding ever expanding the
reference activations to the batch size (which is exactly the cost it's trying to
save). Implementation: `CARBON_modules/variant_cache_layers.py`.
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

out = Path(__file__).parent / "vc_explainer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")

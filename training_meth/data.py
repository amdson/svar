"""Site-level data source for the arabidopsis methylation benchmark.

Reads ``dataset.h5`` (built by ath_meth_benchmark/build/stage3_extract.py) and
patches accession-specific windows from ``snv_arrays.h5`` + TAIR10 via
ath_meth_benchmark.build.window_loader.WindowLoader.

Per site i the source yields
  * ``windows(i)``   unique window strings across accessions, the inverse index
                     (n_acc,) and the index of the pure-reference window
  * ``targets(i)``   p = mc/total (nan below min_cov) and total, both (n_acc,)
  * ``genotypes(i)`` hom-ALT dosage of every SNV any accession carries inside the
                     window, (n_snv, n_acc) int8, plus their 1-based positions

Coordinates. The benchmark's ``sites/pos`` is 0-based (the allc files are
0-based; stage 1 indexed them as if 1-based), so the cytosine sits at TAIR10
1-based ``pos + 1`` on both strands. ``pos_offset`` (default 1) applies that
shift and the constructor verifies it against the reference (C on '+', G on
'-') for >= 99 percent of sites. Minus-strand sites are reverse-complemented
so the presented centre base is always the cytosine itself (window anchored
one base further right, so after RC the C sits at index window//2 exactly as
for plus-strand sites).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "ath_meth_benchmark" / "build") not in sys.path:
    sys.path.insert(0, str(_REPO / "ath_meth_benchmark" / "build"))
from window_loader import WindowLoader  # noqa: E402

from training_meth.backbones import revcomp  # noqa: E402

DATA = "/90daydata/small_grains/andrew.dickson/datasets/arabidopsis"
DEFAULT_H5 = f"{DATA}/methylation/dataset.h5"
DEFAULT_SNV = f"{DATA}/methylation/snv_arrays.h5"
DEFAULT_FASTA = f"{DATA}/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa"

SITE_COLS = ["chrom", "pos", "strand", "context", "subtask", "split_role", "pos_split",
             "annotation_class", "mean_p", "var_obs", "noise", "n_obs", "cg_density", "site_idx"]


def _str(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


class MethSiteSource:
    def __init__(self, h5_path=DEFAULT_H5, snv_h5=DEFAULT_SNV, fasta=DEFAULT_FASTA,
                 window: int = 512, min_cov: int = 5, pos_offset: int = 1):
        self.window, self.min_cov, self.pos_offset = window, min_cov, pos_offset
        self.h5 = h5py.File(h5_path, "r")
        g = self.h5["sites"]
        self.sites = pd.DataFrame({c: (_str(g[c][:]) if g[c].dtype.kind in "OS" else g[c][:])
                                   for c in SITE_COLS if c in g})
        ga = self.h5["accessions"]
        self.acc = pd.DataFrame({c: _str(ga[c][:]) for c in ["ecotype_id", "admixture_group",
                                                            "tissue", "acc_split"]})
        self.n_acc = len(self.acc)
        self.counts = self.h5["counts"]  # (n_sites, n_acc, 2) int16 [mc, total]
        self.wl = WindowLoader(fasta, snv_h5, window=window)
        missing = [e for e in self.acc.ecotype_id if e not in self.wl.h5["acc"]]
        if missing:
            raise RuntimeError(f"{len(missing)} accessions lack SNV arrays, e.g. {missing[:5]}")
        self._check_coordinates()

    def _check_coordinates(self, n: int = 2000):
        s = self.sites if len(self.sites) <= n else self.sites.sample(n, random_state=0)
        want = np.where(s.strand.to_numpy() == "+", ord("C"), ord("G"))
        got = np.array([self.wl.ref[str(c)][int(p) + self.pos_offset - 1]
                        for c, p in zip(s.chrom, s.pos)])
        frac = float((got == want).mean())
        if frac < 0.99:
            raise RuntimeError(f"only {frac:.3f} of sites have a cytosine at pos+{self.pos_offset}; "
                               "check pos_offset against the benchmark build")
        self.coord_check = frac

    # ----------------------------------------------------------- selection
    def select_sites(self, n: int | None, *, contexts=("CG",), subtasks=None, pos_splits=("train",),
                     split_role="eval", annotation=None, seed=0, min_var_ratio: float | None = None):
        s = self.sites
        m = s.context.isin(contexts) & s.pos_split.isin(pos_splits) & (s.split_role == split_role)
        if subtasks:
            m &= s.subtask.isin(subtasks)
        if annotation:
            m &= s.annotation_class.isin(annotation)
        if min_var_ratio is not None:
            m &= (s.var_obs / np.maximum(s.noise, 1e-9)) >= min_var_ratio
        idx = np.flatnonzero(m.to_numpy())
        if n is not None and n < len(idx):
            idx = np.random.default_rng(seed).choice(idx, n, replace=False)
        return np.sort(idx)

    # ----------------------------------------------------------- per site
    def targets(self, i: int):
        c = self.counts[i].astype(np.float32)  # (n_acc, 2)
        mc, total = c[:, 0], c[:, 1]
        p = np.where(total >= self.min_cov, mc / np.maximum(total, 1), np.nan).astype(np.float32)
        return p, total.astype(np.int32)

    def _anchor(self, i: int):
        r = self.sites.iloc[i]
        c = int(r.pos) + self.pos_offset  # 1-based coordinate of the cytosine
        return str(r.chrom), c + (1 if r.strand == "-" else 0), r.strand == "-"

    def _ref_bytes(self, chrom, anchor):
        w = self.window
        lo, hi = anchor - 1 - w // 2, anchor - 1 - w // 2 + w
        ref = self.wl.ref[chrom]
        s = np.full(w, ord("N"), dtype=np.uint8)
        a, b = max(lo, 0), min(hi, len(ref))
        s[a - lo:b - lo] = ref[a:b]
        return s

    def windows(self, i: int):
        chrom, anchor, minus = self._anchor(i)
        raw = [self.wl.window_bytes(e, chrom, anchor).tobytes() for e in self.acc.ecotype_id]
        raw.append(self._ref_bytes(chrom, anchor).tobytes())
        uniq, inverse = np.unique(np.array(raw, dtype=object), return_inverse=True)
        seqs = [u.decode() for u in uniq]
        if minus:
            seqs = [revcomp(s) for s in seqs]
        return seqs, inverse[:-1].astype(np.int32), int(inverse[-1])

    def genotypes(self, i: int):
        chrom, anchor, _ = self._anchor(i)
        w = self.window
        lo, hi = anchor - 1 - w // 2, anchor - 1 - w // 2 + w  # 0-based [lo, hi)
        per_acc = []
        for e in self.acc.ecotype_id:
            pos, alt = self.wl._snvs(str(e), chrom)
            i0, i1 = np.searchsorted(pos, [lo + 1, hi + 1])
            per_acc.append(pos[i0:i1])
        allpos = np.unique(np.concatenate(per_acc)) if per_acc else np.empty(0, np.int64)
        dos = np.zeros((len(allpos), self.n_acc), dtype=np.int8)
        for a, p in enumerate(per_acc):
            if len(p):
                dos[np.searchsorted(allpos, p), a] = 1
        return allpos.astype(np.int64), dos

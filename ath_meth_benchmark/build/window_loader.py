#!/usr/bin/env python3
"""Accession-specific sequence windows (spec §5).

TAIR10 is stored once; per-accession biallelic hom-ALT SNVs are patched into
the requested window on the fly. All coordinates stay in TAIR10 space.

Usage:
    wl = WindowLoader(fasta, snv_h5, window=512)
    seq = wl.window("6909", chrom="1", pos=1234567)        # str, len 512
    oh  = wl.window_onehot("9764", "5", 2000000)           # (512, 4) float32

The target cytosine sits at index window//2 (0-based) of the returned window.
"""
import numpy as np
import h5py
from pyfaidx import Fasta

CHROMS = ["1", "2", "3", "4", "5"]
BASE2COL = np.full(256, -1, dtype=np.int8)
for i, b in enumerate("ACGT"):
    BASE2COL[ord(b)] = i
    BASE2COL[ord(b.lower())] = i


class WindowLoader:
    def __init__(self, fasta_path, snv_h5_path, window=512):
        self.size = window
        fa = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
        self.ref = {c: np.frombuffer(str(fa[c][:]).encode(), dtype=np.uint8).copy()
                    for c in CHROMS}
        self.h5 = h5py.File(snv_h5_path, "r")
        self._cache = {}

    def _snvs(self, ecotype, chrom):
        key = (ecotype, chrom)
        if key not in self._cache:
            g = self.h5["acc"][ecotype]
            ch = g["chrom"][:]          # stored as small ints 1..5
            m = ch == int(chrom)
            pos = g["pos"][:][m]
            alt = g["alt"][:][m]
            order = np.argsort(pos)
            self._cache[key] = (pos[order], alt[order])
        return self._cache[key]

    def window_bytes(self, ecotype, chrom, pos):
        """pos is the 1-based TAIR10 coordinate of the target cytosine."""
        w = self.size
        lo = pos - 1 - w // 2          # 0-based inclusive start
        hi = lo + w
        ref = self.ref[chrom]
        s = np.full(w, ord("N"), dtype=np.uint8)
        a, b = max(lo, 0), min(hi, len(ref))
        s[a - lo:b - lo] = ref[a:b]
        snv_pos, snv_alt = self._snvs(str(ecotype), chrom)
        i0, i1 = np.searchsorted(snv_pos, [lo + 1, hi + 1])
        if i1 > i0:
            s[snv_pos[i0:i1] - 1 - lo] = snv_alt[i0:i1]
        return s

    def window(self, ecotype, chrom, pos):
        return self.window_bytes(ecotype, chrom, pos).tobytes().decode()

    def window_onehot(self, ecotype, chrom, pos):
        s = self.window_bytes(ecotype, chrom, pos)
        cols = BASE2COL[s]
        oh = np.zeros((self.size, 4), dtype=np.float32)
        valid = cols >= 0
        oh[np.arange(self.size)[valid], cols[valid]] = 1.0
        return oh

"""
training_ge/data.py — SIEVE per-gene batches for the variant cache.

One batch = one gene: the tokenized reference window (TSS-centred, length a
multiple of 6 so 6-mer boundaries are stable) plus every mutant line carrying
>=1 SNV inside it. Rows share `cache_idx` (the union of mutated token
positions); a line's row holds reference token ids everywhere except its own
mutations. Row 0 is the pure reference — through the cache it reproduces the
normal forward exactly (identity probe), so (row_i − row_0) at line i's own
positions is the model's representation of *what the mutation changed*.

Targets are the per-gene z-scored deviations (control-line sd), minus the
per-line offset estimated from that line's background genes (same correction
as model_dev/sieve_signal_gate.py).

Token ids are patched via the base-4 6-mer formula (verified against the
tokenizer at startup) — no per-line re-tokenization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

BASE = "/90daydata/small_grains/andrew.dickson/datasets/brachypodium_sieve/"
DATA = BASE + "dataset/"
FASTA = BASE + "reference/Bd21_3.fa"

KMER_BASE, ORDER = 151672, "ATCG"  # Carbon 6-mer block (BENCHMARK.md, verified)
DNA_MARKER = 151669
PAIRS_HW = 5000  # the pairs builder's cis window: gene body +/- 5 kb


def id6(kmer: str) -> int:
    v = 0
    for ch in kmer:
        v = v * 4 + ORDER.index(ch)
    return KMER_BASE + v


def verify_id6(tokenizer, rng: np.random.Generator, n: int = 32) -> None:
    """Assert the formula matches the tokenizer on n random 6-mers."""
    for _ in range(n):
        kmer = "".join(rng.choice(list("ACGT"), 6))
        ids = tokenizer("<dna>" + kmer, add_special_tokens=False)["input_ids"]
        assert ids == [DNA_MARKER, id6(kmer)], f"id6 mismatch for {kmer}: {ids}"


@dataclass
class GeneBatch:
    gene_id: str
    fam_split: str          # train / val / test (family-aware gene split)
    ref_ids: torch.Tensor   # (T,) long — <dna> marker + body tokens
    cache_idx: torch.Tensor  # (C,) long — union of mutated token positions
    hap_ids: torch.Tensor   # (N+1, C) long — row 0 = reference (identity)
    own_mask: torch.Tensor  # (N, C) bool — line's own mutated positions
    z: torch.Tensor         # (N,) float — offset-corrected z targets
    lines: list             # line ids, parallel to z


class SieveWindowSource:
    """Builds `GeneBatch`es for a sample of scoreable SIEVE genes."""

    def __init__(self, tokenizer, half_window: int = 4000,
                 max_lines: int = 64, seed: int = 42):
        import h5py
        import pandas as pd
        import pysam

        self.hw = half_window
        self.max_lines = max_lines
        self.body_len = 6 * ((2 * half_window) // 6)
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        verify_id6(tokenizer, self.rng)

        with h5py.File(DATA + "sieve_dataset.h5", "r") as f:
            self.dev = f["deviation"][:]
            self.gene_id = f["genes/gene_id"][:].astype(str)
            self.chrom = f["genes/chrom"][:].astype(str)
            start, end = f["genes/start"][:], f["genes/end"][:]
            strand = f["genes/strand"][:].astype(str)
            self.fam_split = f["genes/family_split"][:].astype(str)
            is_ctrl = f["lines/is_control"][:]
            self.line_id = f["lines/line_id"][:].astype(str)
        self.tss = np.where(strand == "+", start, end)
        self.sd = self.dev[:, is_ctrl].std(axis=1, ddof=1)
        self.scoreable = self.sd > 1e-3
        self.is_ctrl = is_ctrl
        lix = {l: i for i, l in enumerate(self.line_id)}
        gix = {g: i for i, g in enumerate(self.gene_id)}

        # per-line SNVs (hom-ALT only, per the arrays' builder)
        self.snv = {}
        with h5py.File(DATA + "sieve_snv_arrays.h5", "r") as f:
            for line in f["acc"]:
                g = f["acc"][line]
                self.snv[line] = (g["chrom"][:].astype(str), g["pos"][:],
                                  g["alt"][:])

        # per-line offset from background pairs (see sieve_signal_gate)
        pairs = pd.read_parquet(DATA + "sieve_pairs.parquet")
        bg = pairs[pairs.role == "background"]
        bgi = bg["gene_id"].map(gix).to_numpy()
        bli = bg["line"].map(lix).to_numpy()
        m = self.scoreable[bgi]
        zbg = self.dev[bgi[m], bli[m]] / self.sd[bgi[m]]
        off = np.zeros(len(self.line_id))
        cnt = np.zeros(len(self.line_id))
        np.add.at(off, bli[m], zbg)
        np.add.at(cnt, bli[m], 1)
        self.line_offset = np.where(cnt > 0, off / np.maximum(cnt, 1), 0.0)
        self.lix = lix

        self.fa = pysam.FastaFile(FASTA)
        self.fa_chroms = set(self.fa.references)
        self.skip_counts = {"no_window": 0, "n_in_window": 0, "no_lines": 0,
                            "ref_eq_alt": 0, "edge": 0}

    def sample_genes(self, n_genes: int, split: Optional[str] = None) -> np.ndarray:
        """Random scoreable gene indices, optionally restricted to a family split."""
        ok = self.scoreable & np.isin(self.chrom, list(self.fa_chroms))
        if split is not None:
            ok &= self.fam_split == split
        ix = np.flatnonzero(ok)
        return self.rng.permutation(ix)[:n_genes]

    def build(self, gi: int) -> Optional[GeneBatch]:
        """GeneBatch for gene index gi, or None (reason tallied in skip_counts)."""
        c = self.chrom[gi]
        w0 = int(self.tss[gi]) - self.hw  # 1-based inclusive window start
        if w0 < 1:
            self.skip_counts["edge"] += 1
            return None
        seq = self.fa.fetch(c, w0 - 1, w0 - 1 + self.body_len).upper()
        if len(seq) < self.body_len:
            self.skip_counts["edge"] += 1
            return None
        if "N" in seq:
            self.skip_counts["n_in_window"] += 1
            return None

        # per-line mutations inside the window: {line -> [(tok, patched_id)]}
        per_line = {}
        for j, line in enumerate(self.line_id):
            if self.is_ctrl[j] or line not in self.snv:
                continue
            lc, lp, la = self.snv[line]
            m = (lc == c) & (lp >= w0) & (lp < w0 + self.body_len)
            if not m.any():
                continue
            by_tok = {}
            for pos, alt in zip(lp[m], la[m]):
                offset = int(pos) - w0
                altb = chr(alt)
                if seq[offset] == altb:      # alt == reference: no substitution
                    self.skip_counts["ref_eq_alt"] += 1
                    continue
                by_tok.setdefault(offset // 6, {})[offset % 6] = altb
            muts = []
            for tok, subs in by_tok.items():
                kmer = list(seq[tok * 6:(tok + 1) * 6])
                for k, b in subs.items():
                    kmer[k] = b
                muts.append((1 + tok, id6("".join(kmer))))  # +1: <dna> marker
            if muts:
                per_line[line] = muts
        # drop lines whose target is unavailable
        per_line = {l: mm for l, mm in per_line.items()
                    if np.isfinite(self.dev[gi, self.lix[l]])}
        if not per_line:
            self.skip_counts["no_lines"] += 1
            return None
        lines = sorted(per_line)
        if len(lines) > self.max_lines:
            lines = list(self.rng.choice(lines, self.max_lines, replace=False))

        enc = self.tokenizer("<dna>" + seq, add_special_tokens=False)["input_ids"]
        ref_ids = torch.tensor(enc, dtype=torch.long)
        assert ref_ids.shape[0] == 1 + self.body_len // 6

        cache = sorted({tok for l in lines for tok, _ in per_line[l]})
        cpos = {t: k for k, t in enumerate(cache)}
        C, N = len(cache), len(lines)
        hap = ref_ids[cache].unsqueeze(0).repeat(N + 1, 1)  # row 0 = reference
        own = torch.zeros(N, C, dtype=torch.bool)
        z = np.empty(N, dtype=np.float32)
        for i, l in enumerate(lines):
            for tok, pid in per_line[l]:
                hap[i + 1, cpos[tok]] = pid
                own[i, cpos[tok]] = True
            j = self.lix[l]
            z[i] = self.dev[gi, j] / self.sd[gi] - self.line_offset[j]

        return GeneBatch(gene_id=self.gene_id[gi], fam_split=self.fam_split[gi],
                         ref_ids=ref_ids,
                         cache_idx=torch.tensor(cache, dtype=torch.long),
                         hap_ids=hap, own_mask=own,
                         z=torch.from_numpy(z), lines=lines)

    def iter_batches(self, gene_ix: np.ndarray) -> Iterator[GeneBatch]:
        for gi in gene_ix:
            b = self.build(int(gi))
            if b is not None:
                yield b

"""
pretrained_eval/baselines.py
----------------------------
Naive reference points for the Carbon log-likelihood numbers, plus the bits/nt
conversion that makes losses comparable across tokenizations.

Two baselines, both model-free:

  * uniform over the k-mer DNA vocabulary  -> ``k * ln(4)`` nats/token
    (Carbon's DNA mode has exactly 4**k = 4096 six-mer tokens; see eval notes).
  * empirical k-mer frequency of the rice genome -> the honest "naive model":
    a frequency table with no context. Rice k-mer composition is nearly flat,
    so this sits just below uniform — and is the bar a real LM must clear.

bits/nt is the only unit comparable across models with different tokenizations
(Carbon 6-mer, DNABERT2 BPE, PlantCAD single-nucleotide): divide the per-token
NLL by the bases-per-token, then convert nats->bits.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from pyfaidx import Fasta

LN2 = math.log(2.0)


def uniform_kmer_nll(k: int = 6) -> float:
    """NLL (nats/token) of a uniform model over the 4**k DNA k-mers."""
    return k * math.log(4.0)


def nats_per_token_to_bits_per_nt(nll_per_token: float, bases_per_token: int) -> float:
    """Convert a per-token NLL (nats) to bits per nucleotide.

    ``bases_per_token`` is the k-mer size (Carbon=6) or 1 for single-nucleotide
    tokenizers (PlantCAD). Use the *mean* tokens-per-base for variable-length
    tokenizers like DNABERT2's BPE.
    """
    return (nll_per_token / bases_per_token) / LN2


def empirical_kmer_entropy(
    fasta_path: str,
    k: int = 6,
    n_windows: int = 4000,
    window: int = 1000,
    max_n_frac: float = 0.10,
    numeric_chroms_only: bool = True,
    seed: int = 0,
) -> dict:
    """Entropy of the empirical non-overlapping k-mer distribution of a genome.

    Samples ``n_windows`` random ``window``-bp spans (skipping >`max_n_frac` N,
    matching the eval's window screening), counts non-overlapping k-mers, and
    returns the Shannon entropy — the NLL a context-free frequency table would
    achieve. This is the naive baseline a context model must beat.

    Returns ``{"nats_per_kmer", "bits_per_nt", "n_kmers", "n_distinct"}``.
    """
    ref = Fasta(fasta_path)
    chroms = [c for c in ref.keys() if (c.isdigit() or not numeric_chroms_only)]
    rng = random.Random(seed)
    counts: Counter = Counter()
    got = 0
    while got < n_windows:
        c = rng.choice(chroms)
        L = len(ref[c])
        if L <= window:
            continue
        s = rng.randint(0, L - window)
        seq = str(ref[c][s:s + window]).upper()
        if seq.count("N") / window > max_n_frac:
            continue
        for i in range(0, window - (k - 1), k):
            kmer = seq[i:i + k]
            if "N" not in kmer:
                counts[kmer] += 1
        got += 1

    total = sum(counts.values())
    probs = np.array(list(counts.values()), dtype=float) / total
    h = float(-(probs * np.log(probs)).sum())   # nats per k-mer
    return {
        "nats_per_kmer": h,
        "bits_per_nt": nats_per_token_to_bits_per_nt(h, k),
        "n_kmers": total,
        "n_distinct": len(counts),
    }

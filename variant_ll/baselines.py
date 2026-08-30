"""
variant_ll/baselines.py
-----------------------
Model-free reference points for the allele log-likelihood benchmark, all in the
same unit as the model: **bits per SNP on held-out accessions**.

  B0  uniform over the two alleles                          -> exactly 1 bit
  B1  per-site marginal allele frequency (independent sites) -> the go/no-go bar
  B2  allele of the nearest upstream site in the same window -> the LD headroom

B1 is the naive independent-sites model: fit one allele frequency per site on the
training accessions, score held-out accessions with it. B2 is the cheapest thing
that uses linkage — a 2x2 table per adjacent site pair — and it exists to show
how much headroom above B1 actually exists in the data. A model that cannot beat
B1 has learned nothing individual-specific; a model that cannot beat B2 has
learned less than a lookup table.

Both are smoothed (add-``alpha``, Jeffreys by default): a site that is monomorphic
in train would otherwise assign probability 0 to a held-out minor allele and give
an infinite loss.
"""
from __future__ import annotations

import numpy as np

LN2 = np.log(2.0)


def _xent_bits(p: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-observation cross entropy in bits. ``p`` = P(alt), ``obs`` = 0/1."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(obs * np.log2(p) + (1 - obs) * np.log2(1 - p))


def marginal_probs(train_alleles: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """B1: P(alt) per site from the training accessions. ``train_alleles`` is
    (n_train, S) 0/1; returns (S,)."""
    n = train_alleles.shape[0]
    return (train_alleles.sum(0) + alpha) / (n + 2 * alpha)


def markov_probs(train_alleles: np.ndarray, site_tok: np.ndarray,
                 alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """B2: P(alt at site j | allele at the nearest strictly-upstream site).

    Returns ``(table, upstream)`` where ``table`` is (S, 2) — P(alt) given the
    upstream site carries ref (col 0) or alt (col 1) — and ``upstream`` is (S,)
    with the index of that upstream site, or -1 when there is none (in which case
    the caller falls back to B1, exactly as the model must).
    """
    S = train_alleles.shape[1]
    # `WindowSource.build` emits sites in token order, one per token (multi-site
    # tokens are dropped), so the nearest strictly-upstream site is just j-1.
    if S > 1 and not np.all(np.diff(site_tok) > 0):
        raise ValueError("site_tok must be strictly increasing")
    upstream = np.arange(-1, S - 1, dtype=np.int64)

    table = np.zeros((S, 2), dtype=np.float64)
    for j in range(S):
        i = upstream[j]
        if i < 0:
            continue
        for a in (0, 1):
            m = train_alleles[:, i] == a
            table[j, a] = (train_alleles[m, j].sum() + alpha) / (m.sum() + 2 * alpha)
    return table, upstream


def evaluate(train_alleles: np.ndarray, eval_alleles: np.ndarray,
             site_tok: np.ndarray, alpha: float = 0.5) -> dict:
    """B0/B1/B2 in bits/SNP over ``eval_alleles`` (n_eval, S), fit on train.

    Also returns the per-site B1 probabilities and the per-(site, accession) bit
    matrices, so the runner can slice them the same way it slices the model.
    """
    p1 = marginal_probs(train_alleles, alpha)
    bits1 = _xent_bits(np.broadcast_to(p1, eval_alleles.shape), eval_alleles)

    table, upstream = markov_probs(train_alleles, site_tok, alpha)
    p2 = np.broadcast_to(p1, eval_alleles.shape).copy()
    for j in np.nonzero(upstream >= 0)[0]:
        p2[:, j] = table[j, eval_alleles[:, upstream[j]]]
    bits2 = _xent_bits(p2, eval_alleles)

    return {
        "b0_bits": 1.0,
        "b1_bits": float(bits1.mean()),
        "b2_bits": float(bits2.mean()),
        "b1_per_call": bits1,
        "b2_per_call": bits2,
        "p1": p1,
        "upstream": upstream,
    }

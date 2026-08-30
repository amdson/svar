"""
variant_ll/loss.py
------------------
The allele log-likelihood objective (BENCHMARK.md §2), in bits per SNP.

For each scored token the logits that matter are the ones at the *preceding*
token — in a causal LM that is what predicts it — restricted to the 6-mers
reachable by varying that token's segregating sites, and renormalised over them.
The autoregressive target is simply the 6-mer carrying this haplotype's alleles.
All candidates share the token's non-segregating bases, so the renormalisation
cancels the reference-sequence predictability and leaves a clean 2**m-class
problem directly comparable to a per-site allele-frequency baseline.

For the usual m = 1 that is the two-candidate ref/alt case. For m > 1 it is the
same formula, and it matters that it is the *joint* over the token rather than
each site scored with its co-token neighbours pinned to their true values: the
joint stays a chain-rule factorisation of P(haplotype | reference), so the sum
over tokens is a real log-likelihood on B1's scale, and it conditions only on
strictly upstream tokens rather than on a neighbour 1-5 bp away.

Two backends, scored on identical candidates so their numbers are comparable:

  cache : freeze the all-reference window, substitute the haplotype's alleles at
          the cached positions, recompute only those (the real amortisation).
  exact : full causal forward of the haplotype window (the ceiling the cache is
          approximating).
"""
from __future__ import annotations

import math

import torch

LN2 = math.log(2.0)


def _token_bits(logits: torch.Tensor, cand: torch.Tensor, cand_mask: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
    """(N, K) per-token NLL in bits.

    ``logits`` (N, K, V) at the predicting position, ``cand`` (K, Q) the candidate
    token ids right-padded over tokens with fewer sites, ``cand_mask`` (K, Q) True
    on the real slots, ``target`` (N, K) the index of the true candidate.
    """
    N, K = target.shape
    sel = logits.gather(-1, cand.unsqueeze(0).expand(N, *cand.shape))   # (N, K, Q)
    # Padding slots hold candidate id 0, a real vocab entry — mask them out of the
    # denominator or a token with m < max(m) would be renormalised over other
    # tokens' candidate counts.
    sel = sel.float().masked_fill(~cand_mask.unsqueeze(0), float("-inf"))
    logp = torch.log_softmax(sel, dim=-1)
    return -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) / LN2


def window_weights(batch) -> torch.Tensor:
    """(N, K) bits/SNP weights: haplotype multiplicity x sites in the token.

    A token holding m sites is scored once and accounts for m SNPs, so the
    denominator of "bits per SNP" has to count it m times — that is what keeps the
    metric on B1's per-site scale whatever the token occupancy is.
    """
    return batch.hap_count.unsqueeze(1) * batch.tok_nsite.unsqueeze(0)


def window_bits(model, batch, backend: str = "cache",
                hap_chunk: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token bits for one window: returns ``(bits (N, K), weights (N, K))``.

    Keeps grad, so the caller chooses the context. The weights fold in
    ``batch.hap_count`` — two accessions with the same alleles in a window produce
    byte-identical sequences, so they are deduplicated and carried as a
    multiplicity rather than recomputed — and ``batch.tok_nsite``.
    """
    N = batch.n_hap
    step = hap_chunk or N
    out = [_chunk_bits(model, batch, lo, min(lo + step, N), backend)
           for lo in range(0, N, step)]
    return torch.cat(out, dim=0), window_weights(batch)


def _chunk_bits(model, batch, lo: int, hi: int, backend: str) -> torch.Tensor:
    """(hi-lo, K) bits for one slice of the window's haplotypes."""
    if backend == "cache":
        res = model(batch.ref_ids,
                    variant_positions=batch.cache_idx,
                    variant_input_ids=batch.hap_ids[lo:hi])
        logits = res.logits.index_select(1, batch.pred_row)          # (n, K, V)
    else:
        full = batch.ref_ids.unsqueeze(0).repeat(hi - lo, 1)
        full[:, batch.cache_idx] = batch.hap_ids[lo:hi]
        logits = model(input_ids=full).logits.index_select(
            1, (batch.tok_idx - 1).clamp(min=0))                     # (n, K, V)
    return _token_bits(logits, batch.cand_ids, batch.cand_mask,
                       batch.hap_target[lo:hi])


def window_backward(model, batch, backend: str = "cache",
                    hap_chunk: int | None = None,
                    denom: float | None = None) -> tuple[float, float]:
    """Forward + backward one window, accumulating grad chunk by chunk.

    Returns ``(weighted bits summed, weight summed)`` as floats — no graph is
    handed back, so nothing from this window survives the call.

    ``denom`` is the normalizer the loss is divided by. Pass the **total weight of
    every window in the accumulation group** to make the accumulated gradient the
    gradient of that group's mean bits per SNP — i.e. the actual objective, where
    every scored site counts once. Leaving it None normalizes by this window's own
    weight, which makes each window contribute equally to the step regardless of
    how many sites and haplotypes it holds; that is only correct when one window
    *is* the batch, and even then it disagrees with what ``evaluate`` reports.

    Why chunking rather than ``window_bits(...).backward()``: that path builds
    every chunk's graph and holds all of them until the caller's single backward,
    so ``hap_chunk`` bounds nothing during training and a window with an unusually
    large haplotype count decides peak memory for the whole run. Here each chunk is
    backwarded and freed before the next runs.

    The cost is that the frozen-reference forward is repeated per chunk. At the
    window sizes this benchmark uses (T~168 at half_window=500) that is minor; if
    it ever isn't, hoist `forward_reference` out of the loop rather than raising
    the chunk size past what fits.
    """
    w_all = window_weights(batch)                    # (N, K)
    w_sum = w_all.sum()
    scale = w_sum if denom is None else denom
    N = batch.n_hap
    step = hap_chunk or N
    total_bits = 0.0
    for lo in range(0, N, step):
        hi = min(lo + step, N)
        bits = _chunk_bits(model, batch, lo, hi, backend)
        wsum = (bits * w_all[lo:hi]).sum()
        (wsum / scale).backward()
        total_bits += float(wsum.detach())
    return total_bits, float(w_sum)


def group_weight(batches) -> float:
    """Total scored-site weight over an accumulation group.

    Cheap — ``hap_count`` x ``tok_nsite``, no forward — so the normalizer is known
    before the first backward, which is what lets the group be normalized as one
    batch rather than as a sum of per-window means.
    """
    return float(sum(float(window_weights(b).sum()) for b in batches))


@torch.no_grad()
def evaluate(model, batches, backend: str = "cache",
             hap_chunk: int | None = None) -> dict:
    """Token-weighted mean bits/SNP over windows, split by upstream context.

    The ``noup`` slice is capped at the per-site marginal by construction (those
    tokens have no upstream variant context, so the model's prediction is
    identical for every accession); any real win has to appear in the ``up``
    slice. Both slices are weighted by sites-per-token, so the reported number
    stays bits per SNP.

    The LD-destroying control is applied upstream, in
    ``WindowSource.build(shuffle_sites=True)`` — it has to happen before haplotype
    deduplication, so it is a property of the batch rather than of the scoring.
    """
    was_training = model.training
    model.eval()
    tot = {"all": [0.0, 0.0], "up": [0.0, 0.0], "noup": [0.0, 0.0]}

    for batch in batches:
        bits, w = window_bits(model, batch, backend, hap_chunk)
        up = batch.has_upstream.unsqueeze(0).expand_as(bits)
        for key, mask in (("all", torch.ones_like(up)), ("up", up), ("noup", ~up)):
            tot[key][0] += float((bits * w)[mask].sum())
            tot[key][1] += float(w[mask].sum())

    if was_training:
        model.train()
    out = {f"{k}_bits": (v[0] / v[1] if v[1] else float("nan"))
           for k, v in tot.items()}
    out.update({f"{k}_n": v[1] for k, v in tot.items()})
    return out

"""
variant_ll/loss.py
------------------
The allele log-likelihood objective (BENCHMARK.md §2), in bits per SNP.

For each site the logits that matter are the ones at the *preceding* token — in a
causal LM that is what predicts the token containing the site — restricted to the
two candidate 6-mers (site carrying the reference allele vs the alt allele) and
renormalised over that pair. Both candidates share the other five bases of the
6-mer, so the renormalisation cancels the reference-sequence predictability and
leaves a clean 2-class problem directly comparable to a per-site allele-frequency
baseline.

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


def _pair_bits(logits: torch.Tensor, cand: torch.Tensor,
               allele: torch.Tensor) -> torch.Tensor:
    """(N, S) per-call NLL in bits.

    ``logits`` (N, S, V) at the predicting position, ``cand`` (S, 2) the two
    candidate token ids, ``allele`` (N, S) bool with True = alt.
    """
    N, S = allele.shape
    pair = logits.gather(-1, cand.unsqueeze(0).expand(N, S, 2))   # (N, S, 2)
    logp = torch.log_softmax(pair.float(), dim=-1)
    return -logp.gather(-1, allele.long().unsqueeze(-1)).squeeze(-1) / LN2


def window_bits(model, batch, backend: str = "cache",
                hap_chunk: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-call bits for one window: returns ``(bits (N, S), counts (N,))``.

    Keeps grad, so the caller chooses the context. Weight by ``batch.hap_count``
    to recover the population mean — two accessions with the same alleles in a
    window produce byte-identical sequences, so they are deduplicated and carried
    as a multiplicity rather than recomputed.
    """
    N = batch.n_hap
    chunks = range(0, N, hap_chunk) if hap_chunk else [0]
    step = hap_chunk or N
    out = []
    for lo in chunks:
        hi = min(lo + step, N)
        if backend == "cache":
            res = model(batch.ref_ids,
                        variant_positions=batch.cache_idx,
                        variant_input_ids=batch.hap_ids[lo:hi])
            logits = res.logits.index_select(1, batch.pred_row)      # (n, S, V)
        else:
            full = batch.ref_ids.unsqueeze(0).repeat(hi - lo, 1)
            full[:, batch.cache_idx] = batch.hap_ids[lo:hi]
            logits = model(input_ids=full).logits.index_select(
                1, (batch.site_tok - 1).clamp(min=0))               # (n, S, V)
        out.append(_pair_bits(logits, batch.cand_ids, batch.hap_allele[lo:hi]))
    return torch.cat(out, dim=0), batch.hap_count


@torch.no_grad()
def evaluate(model, batches, backend: str = "cache",
             hap_chunk: int | None = None) -> dict:
    """Token-weighted mean bits/SNP over windows, split by upstream context.

    The ``noup`` slice is capped at the per-site marginal by construction (those
    sites have no variant context, so the model's prediction is identical for
    every accession); any real win has to appear in the ``up`` slice.

    The LD-destroying control is applied upstream, in
    ``WindowSource.build(shuffle_sites=True)`` — it has to happen before haplotype
    deduplication, so it is a property of the batch rather than of the scoring.
    """
    was_training = model.training
    model.eval()
    tot = {"all": [0.0, 0.0], "up": [0.0, 0.0], "noup": [0.0, 0.0]}

    for batch in batches:
        bits, counts = window_bits(model, batch, backend, hap_chunk)
        w = counts.unsqueeze(1).expand_as(bits)
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

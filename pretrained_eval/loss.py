"""
pretrained_eval/loss.py
-----------------------
Autoregressive log-likelihood / perplexity of a *pretrained* Carbon checkpoint
on rice windows, reported at two granularities from a single forward pass:

  * whole-sequence : next-token NLL over **every** DNA token in the window
                     (how well Carbon models the rice sequence overall).
  * per-SNP        : the same NLL restricted to the 6-mer tokens that contain a
                     SNP/alt position (how surprised Carbon is at the variant-
                     bearing tokens specifically).

Carbon is a decoder-only causal LM, so this is plain next-token cross-entropy:
for a token at position ``t`` the loss is ``-log P(token_t | token_<t)``. The
``<dna>`` marker that routes the tokenizer into 6-mer DNA mode sits at position 0
and is only ever context, never a target.

SNP-token positions come from ``crop_embed``'s analytic 6-mer mask
(:func:`_build_snp_mask_kmer`): Carbon's slow tokenizer exposes no offset
mapping, but its DNA mode is a fixed contiguous 6-mer split after the leading
``<dna>`` marker, so genomic position ``p`` lands in token ``1 + (p - w_start) // 6``.

The per-SNP path mirrors the (now-removed) ``variant_ar.loss.snp_ar_nll``; this
module adds the whole-sequence number and computes both at once so the two share
one forward pass.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from crop_embed.embedder import _build_snp_mask_kmer

CARBON_DNA_PREFIX = "<dna>"
CARBON_KMER = 6
CARBON_N_PREFIX_TOKENS = 1


def _batch_row_nll(
    model,
    tokenizer,
    sequences: list[str],
    fingerprints: list,
    max_length: int = 2048,
    device=None,
) -> dict:
    """One Carbon forward pass → **per-row** summed NLLs and token counts.

    Returns CPU tensors, one entry per window in the batch::

        {"seq_nll": (B,) float, "seq_tokens": (B,) long,
         "snp_nll": (B,) float, "snp_tokens": (B,) long}

    Both :func:`carbon_ar_nll` (batch totals) and :func:`collect_per_window`
    (per-window records) build on this so they share identical math.
    """
    if device is None:
        device = next(model.parameters()).device

    seqs = [f"{CARBON_DNA_PREFIX}{s}" for s in sequences]
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # (B, T) bool: tokens whose 6-mer span contains an alt position.
    snp_mask = _build_snp_mask_kmer(
        tuple(input_ids.shape), fingerprints,
        k=CARBON_KMER, n_prefix_tokens=CARBON_N_PREFIX_TOKENS,
    ).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Causal shift: logits at position t-1 predict token t, so a target token at
    # position t lines up with shift index t-1 across every (B, T-1) tensor below.
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]

    # Real (non-padding) target positions = the whole-sequence token set. The
    # <dna> marker is at position 0 and is never a target, so every counted
    # position is a DNA 6-mer token.
    seq_target = (attention_mask[:, 1:].bool()).float()
    snp_target = (snp_mask[:, 1:] & attention_mask[:, 1:].bool()).float()

    # Per-token NLL once, then reduce under each mask per row (reduction="none"
    # so the single forward feeds both metrics).
    per_tok = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_targets.reshape(-1),
        reduction="none",
    ).reshape(shift_targets.shape)

    return {
        "seq_nll": (per_tok * seq_target).sum(dim=1).cpu(),
        "seq_tokens": seq_target.sum(dim=1).long().cpu(),
        "snp_nll": (per_tok * snp_target).sum(dim=1).cpu(),
        "snp_tokens": snp_target.sum(dim=1).long().cpu(),
    }


def carbon_ar_nll(
    model,
    tokenizer,
    sequences: list[str],
    fingerprints: list,
    max_length: int = 2048,
    device=None,
) -> dict:
    """Summed AR negative log-likelihood for one batch of windows.

    Runs a single Carbon forward pass and accumulates two token-weighted sums:
    over all DNA tokens (whole-sequence) and over SNP-containing tokens (per-SNP).

    Parameters
    ----------
    sequences    : variant-haplotype window strings (straight off
                   ``UniqueWindowDataset``; alts already substituted).
    fingerprints : matching ``(chrom, w_start, w_end, alt_positions)`` tuples.

    Returns
    -------
    dict with summed NLLs and token counts so callers can accumulate a
    token-weighted mean across batches::

        {"seq_nll": float, "seq_tokens": int,
         "snp_nll": float, "snp_tokens": int}
    """
    rows = _batch_row_nll(model, tokenizer, sequences, fingerprints,
                          max_length, device)
    return {
        "seq_nll": float(rows["seq_nll"].sum().item()),
        "seq_tokens": int(rows["seq_tokens"].sum().item()),
        "snp_nll": float(rows["snp_nll"].sum().item()),
        "snp_tokens": int(rows["snp_tokens"].sum().item()),
    }


def _summarize(total_nll: float, total_tokens: int) -> dict:
    mean = total_nll / max(total_tokens, 1)
    return {
        "mean_nll": mean,
        "perplexity": math.exp(mean) if total_tokens else float("nan"),
        "n_tokens": total_tokens,
    }


def evaluate(
    model,
    tokenizer,
    loader,
    max_length: int = 2048,
    max_batches: int | None = None,
    device=None,
    progress: bool = True,
) -> dict:
    """Token-weighted mean AR NLL + perplexity over a window loader.

    The loader yields ``{"sequences", "fingerprints"}`` batches (the
    ``variant_mlm.make_loader`` collate). Returns whole-sequence and per-SNP
    summaries plus the window count::

        {"sequence": {"mean_nll", "perplexity", "n_tokens"},
         "snp":      {"mean_nll", "perplexity", "n_tokens"},
         "n_windows": int}
    """
    seq_nll, seq_tok = 0.0, 0
    snp_nll, snp_tok = 0.0, 0
    n_windows = 0

    it = loader
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(loader)
        except ImportError:
            pass

    was_training = model.training
    model.eval()
    for i, batch in enumerate(it):
        if max_batches is not None and i >= max_batches:
            break
        r = carbon_ar_nll(model, tokenizer, batch["sequences"],
                          batch["fingerprints"], max_length, device)
        seq_nll += r["seq_nll"]; seq_tok += r["seq_tokens"]
        snp_nll += r["snp_nll"]; snp_tok += r["snp_tokens"]
        n_windows += len(batch["sequences"])
    if was_training:
        model.train()

    return {
        "sequence": _summarize(seq_nll, seq_tok),
        "snp": _summarize(snp_nll, snp_tok),
        "n_windows": n_windows,
    }


def collect_per_window(
    model,
    tokenizer,
    loader,
    max_length: int = 2048,
    max_batches: int | None = None,
    device=None,
    progress: bool = True,
) -> "list[dict]":
    """Per-window AR loss records, for distribution plots / the eval notebook.

    One record per window with its coordinates, SNP count, and both losses::

        {"chrom", "w_start", "w_end", "n_snps",
         "seq_nll_sum", "seq_tokens", "seq_mean_nll", "seq_ppl",
         "snp_nll_sum", "snp_tokens", "snp_mean_nll", "snp_ppl"}

    ``snp_mean_nll`` / ``snp_ppl`` are NaN for windows with no in-range SNP token
    (e.g. the SNP was truncated past ``max_length``). The token-weighted means in
    :func:`evaluate` equal the token-weighted aggregates of these records.
    """
    records: list[dict] = []
    it = loader
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(loader)
        except ImportError:
            pass

    was_training = model.training
    model.eval()
    for i, batch in enumerate(it):
        if max_batches is not None and i >= max_batches:
            break
        rows = _batch_row_nll(model, tokenizer, batch["sequences"],
                              batch["fingerprints"], max_length, device)
        for b, fp in enumerate(batch["fingerprints"]):
            seq_n = int(rows["seq_tokens"][b])
            snp_n = int(rows["snp_tokens"][b])
            seq_nll = float(rows["seq_nll"][b])
            snp_nll = float(rows["snp_nll"][b])
            seq_mean = seq_nll / seq_n if seq_n else float("nan")
            snp_mean = snp_nll / snp_n if snp_n else float("nan")
            records.append({
                "chrom": fp[0], "w_start": fp[1], "w_end": fp[2],
                "n_snps": len(fp[3]),
                "seq_nll_sum": seq_nll, "seq_tokens": seq_n,
                "seq_mean_nll": seq_mean,
                "seq_ppl": math.exp(seq_mean) if seq_n else float("nan"),
                "snp_nll_sum": snp_nll, "snp_tokens": snp_n,
                "snp_mean_nll": snp_mean,
                "snp_ppl": math.exp(snp_mean) if snp_n else float("nan"),
            })
    if was_training:
        model.train()

    return records

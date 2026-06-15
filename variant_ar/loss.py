"""
variant_ar/loss.py
------------------
Autoregressive log-likelihood loss of Carbon on *SNP tokens*.

Carbon is a causal LM, so the natural variant objective is: feed the variant
haplotype window (alt alleles already substituted by `UniqueWindowDataset`) and
score how well the model predicts the SNP-containing tokens from their left
context. This is plain next-token cross-entropy restricted to the SNP token
positions — a ground-truth loss for sanity-checking the variant-cache forward
(run the same windows through the cache and compare).

SNP-token positions come from `crop_embed`'s analytic 6-mer mask
(`_build_snp_mask_kmer`): Carbon's slow tokenizer exposes no offset mapping, but
its DNA mode is a fixed contiguous 6-mer split after the leading ``<dna>`` marker,
so genomic position ``p`` lands in token ``1 + (p - w_start) // 6``.

Kept deliberately small — purely for testing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from crop_embed.embedder import _build_snp_mask_kmer

CARBON_DNA_PREFIX = "<dna>"
CARBON_KMER = 6
CARBON_N_PREFIX_TOKENS = 1
IGNORE_INDEX = -100


def snp_ar_nll(
    model,
    tokenizer,
    sequences: list[str],
    fingerprints: list,
    max_length: int = 2048,
    device=None,
) -> tuple[float, int]:
    """Summed AR negative log-likelihood of the SNP tokens in a batch of windows.

    Returns ``(sum_nll, n_tokens)`` so callers can accumulate a token-weighted
    mean across batches. ``sequences`` are the variant haplotype strings and
    ``fingerprints`` the matching ``(chrom, w_start, w_end, alt_positions)``
    tuples (both straight off ``UniqueWindowDataset``).
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

    # Causal shift: the logits at position t-1 predict token t. A SNP token at
    # position t is therefore a target at shift index t-1, i.e. snp_mask[:, 1:].
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    target_mask = snp_mask[:, 1:] & attention_mask[:, 1:].bool()

    labels = shift_targets.clone()
    labels[~target_mask] = IGNORE_INDEX

    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return float(nll.item()), int(target_mask.sum().item())


def evaluate(model, tokenizer, loader, max_length: int = 2048,
             max_batches: int | None = None, device=None,
             progress: bool = True) -> dict:
    """Token-weighted mean AR NLL of SNP tokens over a window loader.

    Returns ``{"mean_nll", "perplexity", "n_tokens", "n_windows"}``.
    """
    import math

    total_nll, total_tok, n_windows = 0.0, 0, 0
    it = enumerate(loader)
    if progress:
        try:
            from tqdm import tqdm
            it = enumerate(tqdm(loader))
        except ImportError:
            pass

    for i, batch in it:
        if max_batches is not None and i >= max_batches:
            break
        nll, n = snp_ar_nll(model, tokenizer, batch["sequences"],
                            batch["fingerprints"], max_length, device)
        total_nll += nll
        total_tok += n
        n_windows += len(batch["sequences"])

    mean_nll = total_nll / max(total_tok, 1)
    return {
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if total_tok else float("nan"),
        "n_tokens": total_tok,
        "n_windows": n_windows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Variant-aware loss for *training/eval through the variant cache*.
#
# Why a different objective from `snp_ar_nll` above: in a causal model the SNP
# token is predicted from positions strictly to its left, which cannot see the
# SNP — so "predict the SNP token" is allele-independent and gives the cache no
# variant-specific gradient. The signal that *does* depend on the variant is
# next-token prediction **from** each SNP position (the SNP only influences
# positions >= itself). That is also exactly what the cache natively emits:
# logits at the recomputed variant positions.
#
# Backends scored on the identical objective (so their losses are comparable):
#   exact : full causal forward of the alt haplotype; take logits at SNP positions.
#   cache : freeze the *reference-genome* window, substitute the alt tokens at the
#           SNP positions, recompute just those positions (the real amortization).
# ──────────────────────────────────────────────────────────────────────────────
def snp_next_token_nll(
    model,
    tokenizer,
    ref_sequence: str,
    alt_sequence: str,
    fingerprint,
    backend: str = "cache",
    max_length: int = 2048,
    device=None,
):
    """Summed NLL of the token following each SNP token in one window.

    ``ref_sequence`` is the reference-genome window (no alts), ``alt_sequence``
    the variant haplotype (alts substituted) — both for the same coordinates, so
    their fixed-stride 6-mer tokenizations align position-for-position and differ
    only at SNP tokens. The prediction target is always the alt-haplotype next
    token. Returns ``(sum_nll_tensor, n_tokens)``; the tensor keeps grad for
    training (caller chooses the grad context).

    backend: ``"exact"`` (full alt forward) or ``"cache"`` / anything else
    (variant-cache through the frozen reference genome).
    """
    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(
        [f"{CARBON_DNA_PREFIX}{ref_sequence}", f"{CARBON_DNA_PREFIX}{alt_sequence}"],
        return_tensors="pt", padding="longest", truncation=True,
        max_length=max_length, add_special_tokens=False,
    )
    ids = enc["input_ids"].to(device)          # (2, T): row 0 = ref, row 1 = alt
    am = enc["attention_mask"].to(device)
    ref_ids, alt_ids, am0 = ids[0], ids[1], am[0]
    T = alt_ids.shape[0]
    zero = torch.zeros((), device=device)

    snp = _build_snp_mask_kmer(
        (1, T), [fingerprint], k=CARBON_KMER,
        n_prefix_tokens=CARBON_N_PREFIX_TOKENS).to(device)[0]
    pos = torch.nonzero(snp & am0.bool(), as_tuple=False).flatten()  # SNP token positions
    if pos.numel() == 0:
        return zero, 0

    # Logits at every SNP position (predicting that position's next token).
    if backend == "exact":
        logits = model(input_ids=alt_ids.unsqueeze(0),
                       attention_mask=am0.unsqueeze(0)).logits[0]  # (T, V)
        logit_at = logits.index_select(0, pos)                     # (cs, V)
    else:
        # Reference-genome frozen forward + alt substitutions at the SNP positions.
        out = model(ref_ids, variant_positions=pos, attention_mask=am0,
                    variant_input_ids=alt_ids.index_select(0, pos))
        logit_at = out.logits[0]                                   # (cs, V)

    # Keep SNP positions whose next token exists and is real (not padding).
    nxt = pos + 1
    keep = nxt < T
    pos_k, nxt_k, logit_k = pos[keep], nxt[keep], logit_at[keep]
    if pos_k.numel() > 0:
        real = am0.index_select(0, nxt_k).bool()
        nxt_k, logit_k = nxt_k[real], logit_k[real]
    if logit_k.shape[0] == 0:
        return zero, 0

    targets = alt_ids.index_select(0, nxt_k)
    nll = torch.nn.functional.cross_entropy(logit_k.float(), targets, reduction="sum")
    return nll, int(logit_k.shape[0])


def evaluate_next_token(model, tokenizer, loader, backend: str = "cache",
                        max_length: int = 2048, max_batches: int | None = None,
                        device=None, progress: bool = True) -> dict:
    """Token-weighted mean of `snp_next_token_nll` over a ref/alt window loader.

    The loader yields dicts with ``ref_sequences`` / ``alt_sequences`` /
    ``fingerprints`` (see `variant_ar/train.py`).
    """
    import math

    total_nll, total_tok, n_windows = 0.0, 0, 0
    it = loader
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(loader)
        except ImportError:
            pass

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(it):
            if max_batches is not None and i >= max_batches:
                break
            for rs, asq, fp in zip(batch["ref_sequences"], batch["alt_sequences"],
                                   batch["fingerprints"]):
                nll, n = snp_next_token_nll(model, tokenizer, rs, asq, fp, backend,
                                            max_length, device)
                total_nll += float(nll)
                total_tok += n
                n_windows += 1
    if was_training:
        model.train()

    mean = total_nll / max(total_tok, 1)
    return {
        "mean_nll": mean,
        "perplexity": math.exp(mean) if total_tok else float("nan"),
        "n_tokens": total_tok,
        "n_windows": n_windows,
    }

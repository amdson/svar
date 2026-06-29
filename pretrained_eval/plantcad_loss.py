"""
pretrained_eval/plantcad_loss.py
--------------------------------
Masked-LM log-likelihood / perplexity of a *pretrained* PlantCAD (PlantCaduceus)
checkpoint on rice windows, reported at two granularities and written with the
**same per-window CSV schema** as the Carbon path (``pretrained_eval.loss``) so the
viewer notebook and baselines work unchanged.

PlantCAD is a **bidirectional masked LM** with **single-nucleotide** tokenization
(one token per bp), so — unlike Carbon's next-token cross-entropy — there is no
autoregressive likelihood. We score by **masking**:

  * per-SNP        : mask every SNP/alt token in a window, read
                     ``-log P(true_allele | rest)`` over the 4 nucleotide logits.
                     One forward pass per window — identical in shape to
                     ``variant_mlm``'s objective and to the canonical PlantCaduceus
                     zero-shot variant score (``zero_shot_score.py``).
  * whole-sequence : Salazar-style **pseudo-perplexity** — mask each scored
                     position one at a time and average ``-log P(true | context)``.
                     This is ``L`` forward passes per window, so it is gated behind
                     ``seq_mode`` (full / subsample / none); a subsample of
                     positions per window is the default (cost lever, logged — no
                     silent truncation).

Single-nucleotide tokenization makes bits/nt trivial: the per-token NLL *is* the
per-nucleotide NLL, so ``nats_per_token_to_bits_per_nt(nll, bases_per_token=1)``
= ``nll / ln 2``. That is the clean common unit against Carbon (k=6) and DNABERT2
(mean tokens/base). The 4-way softmax over ``a/c/g/t`` matches the naive baselines
(``empirical_kmer_entropy(fasta, k=1)`` and ``uniform_kmer_nll(1) = ln 4``).

SNP-token positions reuse ``crop_embed``'s analytic single-nucleotide mask
(:func:`_build_snp_mask_kmer` with ``k=1``): PlantCAD's tokenizer exposes no offset
mapping, but one-token-per-bp means genomic position ``p`` lands in token
``n_prefix + (p - w_start)``. ``n_prefix`` (any leading BOS/CLS) is detected from
the tokenizer at load time by :func:`plantcad_n_prefix_tokens` rather than
hard-coded, and asserted against the reference base.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from crop_embed.embedder import _build_snp_mask_kmer

PLANTCAD_NT = ("a", "c", "g", "t")  # PlantCAD nucleotide tokens are LOWERCASE
PLANTCAD_KMER = 1                   # single-nucleotide tokenization
PLANTCAD_BASES_PER_TOKEN = 1


def nt_token_ids(tokenizer) -> list[int]:
    """Vocab ids of the four lowercase nucleotide tokens, in ``a,c,g,t`` order."""
    vocab = tokenizer.get_vocab()
    missing = [c for c in PLANTCAD_NT if c not in vocab]
    if missing:
        raise ValueError(
            f"tokenizer vocab is missing nucleotide token(s) {missing}; "
            "PlantCAD scoring expects lowercase a/c/g/t tokens.")
    return [vocab[c] for c in PLANTCAD_NT]


def _id_to_nt_index(tokenizer) -> dict[int, int]:
    """Map a nucleotide token id -> its 0..3 index in ``a,c,g,t`` (else absent)."""
    return {tid: i for i, tid in enumerate(nt_token_ids(tokenizer))}


def plantcad_n_prefix_tokens(tokenizer) -> int:
    """Number of leading special tokens (BOS/CLS) PlantCAD prepends to the DNA.

    Detected, not assumed: tokenize a known all-nucleotide string and locate the
    first position whose token id is a nucleotide. The PlantCAD Colab masked-scoring
    recipe indexes ``input_ids[0, pos]`` directly with the sequence character
    ``pos`` (cells 16-18), i.e. token index == character index, which implies 0 —
    but we verify rather than trust it, since the leading-token count drives the
    SNP-token math (:func:`_build_snp_mask_kmer`'s ``n_prefix_tokens``).
    """
    probe = "acgtacgtac"
    ids = tokenizer(probe, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    nt_ids = set(nt_token_ids(tokenizer))
    for i, tid in enumerate(ids.tolist()):
        if tid in nt_ids:
            return i
    raise ValueError("could not locate any nucleotide token in a probe encoding; "
                     "PlantCAD tokenizer behaviour is unexpected.")


def _encode_batch(tokenizer, sequences: list[str], max_length: int):
    """Tokenize a batch of windows (lowercased) to padded ``(input_ids, attn)``.

    PlantCAD's vocab is lowercase; soft-masked genome bases arrive upper/lower
    mixed, so we lowercase here. ``add_special_tokens=True`` keeps whatever
    leading marker the checkpoint uses; :func:`plantcad_n_prefix_tokens` accounts
    for it.
    """
    enc = tokenizer(
        [s.lower() for s in sequences],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return enc["input_ids"], enc["attention_mask"]


def _score_positions(model, input_ids, attention_mask, positions_per_row, nt_ids,
                     id_to_nt, mask_id: int, device, sub_batch: int):
    """Pseudo-likelihood NLL summed over the given positions, per row.

    For each row ``b`` and each position ``p`` in ``positions_per_row[b]``, create a
    copy of the row with token ``p`` replaced by ``[MASK]``, run the model, and read
    ``-log P(true_base | rest)`` as a 4-way softmax over the ``a/c/g/t`` logits.
    Positions whose true token is not a nucleotide (e.g. an ``N``/unk) are skipped.

    Masked copies are batched across ``(row, position)`` pairs in chunks of
    ``sub_batch`` so the ``L``-forward-passes-per-window cost stays on the GPU
    efficiently. Returns ``(nll_sum_per_row, n_scored_per_row)``.
    """
    B = input_ids.shape[0]
    nll_sum = [0.0] * B
    n_scored = [0] * B
    nt_ids_t = torch.tensor(nt_ids, device=device)

    jobs: list[tuple[int, int, int]] = []
    for b in range(B):
        row = input_ids[b]
        for p in positions_per_row[b]:
            true_idx = id_to_nt.get(int(row[p]))
            if true_idx is None:
                continue
            jobs.append((b, p, true_idx))

    for start in range(0, len(jobs), sub_batch):
        chunk = jobs[start:start + sub_batch]
        rows = torch.stack([input_ids[b] for (b, _p, _t) in chunk]).clone()
        attn = torch.stack([attention_mask[b] for (b, _p, _t) in chunk])
        for j, (_b, p, _t) in enumerate(chunk):
            rows[j, p] = mask_id
        with torch.no_grad():
            logits = model(input_ids=rows.to(device),
                           attention_mask=attn.to(device)).logits
        pos_idx = torch.tensor([p for (_b, p, _t) in chunk], device=device)
        gathered = logits[torch.arange(len(chunk), device=device), pos_idx]  # (C, V)
        nt_logits = gathered.index_select(-1, nt_ids_t).float()              # (C, 4)
        logp = F.log_softmax(nt_logits, dim=-1)
        for j, (b, _p, true_idx) in enumerate(chunk):
            nll_sum[b] += float(-logp[j, true_idx])
            n_scored[b] += 1
    return nll_sum, n_scored


def _snp_positions_per_row(input_ids, fingerprints, n_prefix_tokens) -> list[list[int]]:
    """Token positions of SNP/alt alleles in each row (single-nucleotide mask)."""
    snp_mask = _build_snp_mask_kmer(
        tuple(input_ids.shape), fingerprints,
        k=PLANTCAD_KMER, n_prefix_tokens=n_prefix_tokens,
    )
    return [snp_mask[b].nonzero(as_tuple=True)[0].tolist()
            for b in range(input_ids.shape[0])]


def _seq_positions_per_row(input_ids, attention_mask, fingerprints, n_prefix_tokens,
                           seq_mode: str, subsample: int, seq_seed: int) -> list[list[int]]:
    """Whole-sequence scored positions per row, honouring ``seq_mode``.

    ``full`` scores every real DNA token; ``subsample`` keeps a deterministic,
    evenly-spaced ``subsample`` of them per window; ``none`` scores nothing. The
    DNA token span is ``[n_prefix, n_prefix + window_len)`` clamped to the real
    (non-padding) tokens; any trailing special token is excluded by the attention
    span and by the per-position nucleotide check downstream.
    """
    if seq_mode == "none":
        return [[] for _ in range(input_ids.shape[0])]

    out: list[list[int]] = []
    for b, fp in enumerate(fingerprints):
        w_start, w_end = fp[1], fp[2]
        win_len = w_end - w_start
        n_real = int(attention_mask[b].sum())
        last = min(n_prefix_tokens + win_len, n_real, input_ids.shape[1])
        positions = list(range(n_prefix_tokens, last))
        if seq_mode == "subsample" and subsample > 0 and len(positions) > subsample:
            # Deterministic evenly-spaced subsample (seed kept for parity / future
            # randomised variants; spacing itself is independent of it).
            idx = torch.linspace(0, len(positions) - 1, subsample).round().long().tolist()
            positions = [positions[i] for i in sorted(set(idx))]
        out.append(positions)
    return out


def _batch_row_nll(
    model,
    tokenizer,
    sequences: list[str],
    fingerprints: list,
    max_length: int = 512,
    device=None,
    *,
    n_prefix_tokens: int | None = None,
    seq_mode: str = "subsample",
    subsample: int = 128,
    seq_seed: int = 0,
    sub_batch: int = 128,
    verify_ref: bool = False,
) -> dict:
    """PlantCAD masked scoring → **per-row** summed NLLs and token counts.

    Mirrors ``loss._batch_row_nll``'s return shape so :func:`collect_per_window`
    emits the identical CSV schema::

        {"seq_nll": (B,) float, "seq_tokens": (B,) long,
         "snp_nll": (B,) float, "snp_tokens": (B,) long}

    Per-SNP uses one masked forward per window (all SNP tokens masked together);
    whole-sequence uses the pseudo-PPL position loop gated by ``seq_mode``.
    """
    if device is None:
        device = next(model.parameters()).device
    if n_prefix_tokens is None:
        n_prefix_tokens = plantcad_n_prefix_tokens(tokenizer)

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("PlantCAD tokenizer has no mask_token_id; masked scoring needs one.")
    nt_ids = nt_token_ids(tokenizer)
    id_to_nt = _id_to_nt_index(tokenizer)

    input_ids, attention_mask = _encode_batch(tokenizer, sequences, max_length)

    snp_pos = _snp_positions_per_row(input_ids, fingerprints, n_prefix_tokens)
    if verify_ref:
        _assert_snp_alignment(input_ids, fingerprints, snp_pos, sequences,
                              n_prefix_tokens, id_to_nt)

    snp_nll, snp_tok = _score_positions(
        model, input_ids, attention_mask, snp_pos, nt_ids, id_to_nt,
        mask_id, device, sub_batch)

    seq_pos = _seq_positions_per_row(
        input_ids, attention_mask, fingerprints, n_prefix_tokens,
        seq_mode, subsample, seq_seed)
    seq_nll, seq_tok = _score_positions(
        model, input_ids, attention_mask, seq_pos, nt_ids, id_to_nt,
        mask_id, device, sub_batch)

    return {
        "seq_nll": torch.tensor(seq_nll),
        "seq_tokens": torch.tensor(seq_tok, dtype=torch.long),
        "snp_nll": torch.tensor(snp_nll),
        "snp_tokens": torch.tensor(snp_tok, dtype=torch.long),
    }


def _assert_snp_alignment(input_ids, fingerprints, snp_pos, sequences,
                          n_prefix_tokens, id_to_nt) -> None:
    """Sanity-check that a SNP token index decodes to its window-string base.

    Cheap guard against an off-by-one in ``n_prefix_tokens``: the token at the
    computed index must be the nucleotide the (alt-substituted) window string
    carries at that genomic offset.
    """
    for b, fp in enumerate(fingerprints):
        w_start = fp[1]
        seq = sequences[b].lower()
        for p in fp[3]:
            tok_idx = n_prefix_tokens + (p - w_start)
            if not (0 <= tok_idx < input_ids.shape[1]):
                continue
            char = seq[p - w_start] if 0 <= p - w_start < len(seq) else None
            tok_nt = id_to_nt.get(int(input_ids[b, tok_idx]))
            if char in ("a", "c", "g", "t") and tok_nt is not None:
                assert PLANTCAD_NT[tok_nt] == char, (
                    f"SNP token misalignment at window {fp[0]}:{w_start} pos {p}: "
                    f"token decodes to {PLANTCAD_NT[tok_nt]!r} but window base is {char!r} "
                    f"(check n_prefix_tokens={n_prefix_tokens})")
            return  # one check per batch is enough


def collect_per_window(
    model,
    tokenizer,
    loader,
    max_length: int = 512,
    max_batches: int | None = None,
    device=None,
    progress: bool = True,
    *,
    seq_mode: str = "subsample",
    subsample: int = 128,
    seq_seed: int = 0,
    sub_batch: int = 128,
) -> "list[dict]":
    """Per-window PlantCAD loss records — **same keys** as ``loss.collect_per_window``::

        {"chrom", "w_start", "w_end", "n_snps",
         "seq_nll_sum", "seq_tokens", "seq_mean_nll", "seq_ppl",
         "snp_nll_sum", "snp_tokens", "snp_mean_nll", "snp_ppl"}

    so ``pretrained_eval/view_losses.ipynb`` reads it unchanged (set ``K=1``).
    ``seq_*`` columns are NaN when ``seq_mode='none'``; ``snp_*`` are NaN for
    windows whose SNP fell outside ``max_length``.
    """
    n_prefix_tokens = plantcad_n_prefix_tokens(tokenizer)

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
        rows = _batch_row_nll(
            model, tokenizer, batch["sequences"], batch["fingerprints"],
            max_length=max_length, device=device, n_prefix_tokens=n_prefix_tokens,
            seq_mode=seq_mode, subsample=subsample, seq_seed=seq_seed,
            sub_batch=sub_batch, verify_ref=(i == 0))
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

"""
notebooks/pll_compare.py  (also rendered as pll_compare.ipynb)
--------------------------------------------------------------
Pseudo-loglikelihood comparison: DNABERT-2 vs PlantCAD on the same DNA
sequence. For each non-special token we replace it with [MASK], run a forward
pass, and read the log-prob of the true token. Summing gives the standard
masked PLL.

Cross-tokenizer caveat: DNABERT-2 tokenizes with BPE (one token spans several
bp), while PlantCAD tokenizes at single-nucleotide resolution. We report PLL
two ways:

  * total PLL              — directly comparable between two runs of the same
                              model on the same input, not across models.
  * mean nats / bp         — total PLL / sequence_length_in_bp; gives a rough
                              cross-tokenizer comparison, but biased: single-bp
                              masking (PlantCAD) is easier per bp than chunk
                              masking (DNABERT-2 BPE).

This script runs cleanly as a Jupyter notebook (`# %%` cells are picked up by
Jupytext / VS Code) and as a standalone script.
"""

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".").resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import pandas as pd

from DNABERT2_modules import load_dnabert2_mlm
from PlantCAD_modules import load_plantcad_mlm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# %% [markdown]
# ## Shared test sequence
#
# 500 bp from the PlantCAD Colab example — short enough to fit PlantCAD's 512
# bp max input, long enough that DNABERT-2 BPE produces many tokens.

# %%
SEQUENCE = (
    "CTTAATTAATATTGCCTTTGTAATAACGCGCGAAACACAAATCTTCTCTGCCTAATGCAG"
    "TAGTCATGTGTTGACTCCTTCAAAATTTCCAAGAAGTTAGTGGCTGGTGTGTCATTGTCT"
    "TCATCTTTTTTTTTTTTTTTTTAAAAATTGAATGCGACATGTACTCCTCAACGTATAAGC"
    "TCAATGCTTGTTACTGAAACATCTCTTGTCTGATTTTTTCAGGCTAAGTCTTACAGAAAG"
    "TGATTGGGCACTTCAATGGCTTTCACAAATGAAAAAGATGGATCTAAGGGATTTGTGAAG"
    "AGAGTGGCTTCATCTTTCTCCATGAGGAAGAAGAAGAATGCAACAAGTGAACCCAAGTTG"
    "CTTCCAAGATCGAAATCAACAGGTTCTGCTAACTTTGAATCCATGAGGCTACCTGCAACG"
    "AAGAAGATTTCAGATGTCACAAACAAAACAAGGATCAAACCATTAGGTGGTGTAGCACCA"
    "GCACAACCAAGAAGGGAA"
)
print(f"sequence length: {len(SEQUENCE)} bp")
assert len(SEQUENCE) <= 512, "PlantCAD max input is 512 bp"

# %% [markdown]
# ## Pseudo-loglikelihood utility
#
# Generic masked PLL — works for any HuggingFace MaskedLM. Stacks N copies of
# the input (one mask per content position), batches the forward pass in chunks
# of `batch_size`, and reads the log-prob of the true token at each mask.

# %%
def pseudo_loglikelihood(
    model,
    tokenizer,
    sequence: str,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict:
    """
    Standard masked pseudo-loglikelihood.

    For each non-special token position p in the tokenized sequence:
      input' = input with token p replaced by [MASK]
      logprob[p] = log softmax(model(input').logits[p])[true_token_at_p]
    Returns total_logprob = sum(logprob[p]) plus per-position diagnostics.

    Parameters
    ----------
    model       : eval-mode MaskedLM with `model(input_ids).logits` of shape (B, L, V).
    tokenizer   : matching HuggingFace tokenizer; must define `mask_token_id`
                  and `all_special_ids`.
    sequence    : raw DNA string.
    batch_size  : how many masked variants to forward at once.

    Returns
    -------
    dict with keys:
      total_logprob       : float, sum of per-token log-probs (nats).
      n_content_tokens    : int.
      per_token_logprob   : Tensor[n_content].
      content_positions   : LongTensor[n_content] — indices into `input_ids`.
      content_token_ids   : LongTensor[n_content] — the true tokens.
    """
    if device is None:
        device = next(model.parameters()).device
    if tokenizer.mask_token_id is None:
        raise ValueError("tokenizer has no mask_token_id; not an MLM tokenizer?")

    enc = tokenizer(sequence, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)

    special = set(tokenizer.all_special_ids)
    content_positions = [i for i, t in enumerate(input_ids.tolist()) if t not in special]
    if not content_positions:
        raise ValueError("no non-special tokens to score")

    n = len(content_positions)
    mask_id = tokenizer.mask_token_id
    per_token = torch.empty(n, device=device)

    for start in range(0, n, batch_size):
        positions = content_positions[start:start + batch_size]
        batch = input_ids.unsqueeze(0).repeat(len(positions), 1)
        rows = torch.arange(len(positions))
        col_idx = torch.tensor(positions)
        batch[rows, col_idx] = mask_id

        with torch.inference_mode():
            out = model(input_ids=batch)
        logits = out.logits if hasattr(out, "logits") else out[0]
        log_probs = F.log_softmax(logits.float(), dim=-1)

        true_ids = input_ids[col_idx]
        per_token[start:start + len(positions)] = log_probs[rows, col_idx, true_ids]

    return {
        "total_logprob":     per_token.sum().item(),
        "n_content_tokens":  n,
        "per_token_logprob": per_token.cpu(),
        "content_positions": torch.tensor(content_positions),
        "content_token_ids": input_ids[torch.tensor(content_positions)].cpu(),
    }


# %% [markdown]
# ## DNABERT-2 PLL
#
# BPE tokenizer — one token spans several bp. Expect ~85–100 content tokens
# for a 500 bp sequence.

# %%
print("Loading DNABERT-2 ...")
bert_model, bert_tok = load_dnabert2_mlm(device=device)
print(f"  vocab size: {len(bert_tok)}, mask_token_id: {bert_tok.mask_token_id}")

bert_result = pseudo_loglikelihood(bert_model, bert_tok, SEQUENCE, batch_size=32, device=device)
bert_bpp = bert_result["total_logprob"] / len(SEQUENCE)

print(
    f"DNABERT-2 PLL: total={bert_result['total_logprob']:.2f} nats over "
    f"{bert_result['n_content_tokens']} BPE tokens "
    f"({bert_bpp:.4f} nats/bp; {bert_result['total_logprob'] / bert_result['n_content_tokens']:.4f} nats/token)"
)

# %% [markdown]
# ## PlantCAD PLL
#
# Single-nucleotide tokenizer — exactly len(sequence) content tokens.

# %%
print("Loading PlantCAD ...")
pc_model, pc_tok = load_plantcad_mlm(device=device, dtype=torch.float16)
print(f"  vocab size: {len(pc_tok)}, mask_token_id: {pc_tok.mask_token_id}")

pc_result = pseudo_loglikelihood(pc_model, pc_tok, SEQUENCE, batch_size=16, device=device)
pc_bpp = pc_result["total_logprob"] / len(SEQUENCE)

print(
    f"PlantCAD  PLL: total={pc_result['total_logprob']:.2f} nats over "
    f"{pc_result['n_content_tokens']} bp tokens "
    f"({pc_bpp:.4f} nats/bp)"
)

# %% [markdown]
# ## Single-position sanity check
#
# Replicates cell 18 of the PlantCAD Colab: mask one position, softmax over
# {a,c,g,t}, true base should get the highest probability.

# %%
PROBE_POS = 255
print(f"Probing position {PROBE_POS} (true base = {SEQUENCE[PROBE_POS]!r})")
for name, model, tok in [("DNABERT-2", bert_model, bert_tok), ("PlantCAD", pc_model, pc_tok)]:
    enc = tok(SEQUENCE, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    if name == "PlantCAD":
        # find the token at bp PROBE_POS — PlantCAD has a leading special token
        special = set(tok.all_special_ids)
        token_pos = [i for i, t in enumerate(ids[0].tolist()) if t not in special][PROBE_POS]
    else:
        # DNABERT-2 BPE: find which token contains bp PROBE_POS
        cumlen, token_pos = 0, None
        for i, tid in enumerate(ids[0].tolist()):
            if tid in set(bert_tok.all_special_ids):
                continue
            piece = bert_tok.convert_ids_to_tokens([tid])[0].replace("##", "")
            cumlen += len(piece)
            if cumlen > PROBE_POS:
                token_pos = i
                break
    ids[0, token_pos] = tok.mask_token_id
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, token_pos]
    probs = F.softmax(logits.float().cpu(), dim=-1)
    top5 = torch.topk(probs, k=5)
    top_tokens = tok.convert_ids_to_tokens(top5.indices.tolist())
    print(f"  {name:9s} top-5: {list(zip(top_tokens, [round(p, 4) for p in top5.values.tolist()]))}")

# %% [markdown]
# ## Side-by-side summary

# %%
summary = pd.DataFrame([
    {
        "model":             "DNABERT-2",
        "tokenizer":         "BPE",
        "n_content_tokens":  bert_result["n_content_tokens"],
        "total_PLL (nats)":  round(bert_result["total_logprob"], 3),
        "PLL / token":       round(bert_result["total_logprob"] / bert_result["n_content_tokens"], 4),
        "PLL / bp":          round(bert_bpp, 4),
    },
    {
        "model":             "PlantCAD",
        "tokenizer":         "1bp",
        "n_content_tokens":  pc_result["n_content_tokens"],
        "total_PLL (nats)":  round(pc_result["total_logprob"], 3),
        "PLL / token":       round(pc_result["total_logprob"] / pc_result["n_content_tokens"], 4),
        "PLL / bp":          round(pc_bpp, 4),
    },
])
print(summary.to_string(index=False))

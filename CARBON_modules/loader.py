"""
CARBON_modules/loader.py
------------------------
Loader for the Carbon genome language model (HuggingFaceBio/Carbon-*), a
decoder-only causal LM over DNA. Mirrors the surface of `DNABERT2_modules` and
`PlantCAD_modules` so `crop_embed.encoders` can dispatch to it uniformly.

  load_carbon -> (AutoModelForCausalLM, tokenizer)  for embeddings (and LM scoring)

Two Carbon-specific quirks the embedder must honour (see crop_embed/embedder.py):

  * The tokenizer is dual-mode. A sequence must be wrapped in a leading ``<dna>``
    marker to route it into 6-mer DNA tokenization; that marker is supplied as
    literal text, so tokenization is done with ``add_special_tokens=False``.
  * The checkpoint is a causal LM, so per-position representations come from
    ``model(..., output_hidden_states=True).hidden_states`` rather than a
    ``last_hidden_state`` field.

Runtime notes
-------------
Carbon ships its modeling code on the Hub, so loading needs
``trust_remote_code=True``. The published example loads in bfloat16; that is the
default here too (override via ``dtype``). CPU-only boxes can import the class but
inference realistically needs CUDA for the 500M/3B sizes.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The 500M Carbon checkpoint. The published example uses "HuggingFaceBio/Carbon-3B";
# the size variants share a repo-naming scheme, so 500M is the -500M sibling.
_DEFAULT_REPO = "HuggingFaceBio/Carbon-500M"


def load_carbon(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Carbon for embedding extraction (and causal-LM scoring).

    The returned model is an `AutoModelForCausalLM`. Per-position hidden states
    are available via `model(..., output_hidden_states=True).hidden_states[-1]`
    and next-token logits via `.logits`. The matching tokenizer routes DNA into
    6-mer mode when the input string is prefixed with ``<dna>`` — the embedder
    adds that prefix and disables auto special tokens (see WindowEmbedder).

    Parameters
    ----------
    repo_id : str
        HuggingFace repo or local directory. Defaults to the 500M checkpoint.
    device : str or torch.device, optional
        Defaults to CUDA when available, else CPU.
    dtype : torch.dtype, optional
        Compute/storage dtype. Defaults to ``torch.bfloat16`` (matching the
        published Carbon example); pass e.g. ``torch.float32`` to override.

    Returns
    -------
    model : eval-mode AutoModelForCausalLM on `device`.
    tokenizer : AutoTokenizer for Carbon's dual-mode (text / 6-mer DNA) vocab.
    """
    local = os.path.isdir(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, local_files_only=local
    )

    if dtype is None:
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer


def load_carbon_lm(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Identical to `load_carbon` — Carbon's backbone and LM head live in one
    AutoModelForCausalLM checkpoint. Kept as a separate symbol for naming parity
    with `DNABERT2_modules.load_dnabert2_mlm` / `PlantCAD_modules.load_plantcad_mlm`.
    """
    return load_carbon(repo_id, device=device, dtype=dtype)

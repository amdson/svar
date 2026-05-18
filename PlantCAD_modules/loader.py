"""
PlantCAD_modules
----------------
Loaders for the PlantCaduceus DNA language model (Caduceus / Mamba backbone,
bi-directional with reverse-complement parameter sharing). Mirrors the surface
of `DNABERT2_modules`.

  load_plantcad     -> (AutoModelForMaskedLM, tokenizer)  for embeddings
  load_plantcad_mlm -> (AutoModelForMaskedLM, tokenizer)  for masked-LM scoring
  average_rc_embeddings(hidden_states) -> single-strand collapsed embedding

PlantCAD's encoder and MLM head ship inside a single `AutoModelForMaskedLM`
checkpoint, so both load functions return the same object — keeping them
separate matches DNABERT2's surface and signals intent at the call site.

Runtime requirements
--------------------
PlantCaduceus depends on `mamba-ssm`. The upstream Colab guide pins
`torch==2.3.1+cu121`, `mamba-ssm==2.2.2`, `triton==2.3.1`, and uses
`trust_remote_code=True` for the Caduceus modeling files. CPU-only boxes can
import the model class but inference paths require CUDA.

Sequence length: PlantCAD has a max input of 512 bp (single-nucleotide
tokenization, one token per bp plus a small number of special tokens).
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

_DEFAULT_REPO = "kuleshov-group/PlantCaduceus_l32"


def load_plantcad(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Load PlantCAD (PlantCaduceus) for embedding extraction and masked-LM scoring.

    The returned model is an `AutoModelForMaskedLM`. Hidden states are
    available via `model(..., output_hidden_states=True).hidden_states[-1]`,
    and logits via `.logits`. Use `average_rc_embeddings()` to collapse the
    last hidden state from (..., D) to single-strand (..., D/2).

    Parameters
    ----------
    repo_id : str
        HuggingFace repo or local directory.
    device : str or torch.device, optional
        Defaults to CUDA when available, else CPU.
    dtype : torch.dtype, optional
        e.g. `torch.float16`. Defaults to the checkpoint's native dtype.

    Returns
    -------
    model : eval-mode AutoModelForMaskedLM on `device`.
    tokenizer : AutoTokenizer for PlantCAD's single-nucleotide vocab.
    """
    local = os.path.isdir(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=local)

    load_kwargs = {"trust_remote_code": True, "local_files_only": local}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForMaskedLM.from_pretrained(repo_id, **load_kwargs)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer


def load_plantcad_mlm(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Identical to `load_plantcad` — PlantCAD's encoder and MLM head live in one
    AutoModelForMaskedLM checkpoint. Kept as a separate symbol for naming
    parity with `DNABERT2_modules.load_dnabert2_mlm`.
    """
    return load_plantcad(repo_id, device=device, dtype=dtype)


def average_rc_embeddings(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    PlantCAD has reverse-complement parameter sharing: the last hidden state
    concatenates [forward_repr, reverse_complement_repr] along the embedding
    dim. Averaging gives a single-strand representation.

    Follows the canonical recipe from the PlantCAD Colab example:

        forward = h[..., :D/2]
        reverse = h[..., D/2:][..., ::-1]
        averaged = (forward + reverse) / 2

    Parameters
    ----------
    hidden_states : Tensor (..., L, D) — e.g. `outputs.hidden_states[-1]`.

    Returns
    -------
    Tensor (..., L, D/2)
    """
    hidden_size = hidden_states.shape[-1] // 2
    forward = hidden_states[..., :hidden_size]
    reverse = hidden_states[..., hidden_size:].flip(-1)
    return (forward + reverse) / 2

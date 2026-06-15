"""
encoders.py
-----------
The single place the codebase loads a DNA encoder. Centralizes backend dispatch
(DNABERT-2 vs PlantCAD), default model paths, and the dnabert2 pad-token wiring
that every script used to duplicate.

    load_encoder_model(backend, model_path) -> (model, tokenizer)
        Low-level: what fill_embedding_table consumes.
    build_window_embedder(backend, model_path, ...) -> WindowEmbedder
        Wraps the above in a WindowEmbedder for the training / e2e paths.

Backend loaders are imported lazily so importing crop_embed doesn't pull in
DNABERT2_modules / PlantCAD_modules unless a model is actually loaded. (The MLM
loaders in DNABERT2_modules / PlantCAD_modules are a separate concern — scoring
heads, not embedders — and aren't routed through here.)
"""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from crop_embed.embedder import WindowEmbedder

DEFAULT_MODEL_PATHS = {
    "dnabert2": "zhihan1996/DNABERT-2-117M",
    "plantcad": "kuleshov-group/PlantCaduceus_l32",
    "carbon":   "HuggingFaceBio/Carbon-500M",
}


def load_encoder_model(
    backend: str = "dnabert2",
    model_path: str | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    attn_impl: str = "torch",
):
    """
    Load an encoder + tokenizer for `backend`, defaulting `model_path` per backend.

    Returns (model, tokenizer) — the pair `SampleEmbedder.fill_embedding_table`
    and `WindowEmbedder` consume. `device`/`dtype` are forwarded to the backend
    loader (dnabert2 ignores `dtype`).

    `attn_impl` selects the DNABERT-2 attention backend: "torch" (the default,
    pure-PyTorch matmul — needs no flash-attn/CUDA) or "flash" (FlashAttention,
    requires CUDA + the flash-attn package). Ignored by the plantcad/carbon
    backends.
    """
    if backend not in DEFAULT_MODEL_PATHS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {list(DEFAULT_MODEL_PATHS)}.")
    model_path = model_path or DEFAULT_MODEL_PATHS[backend]
    local = os.path.isdir(model_path)

    if backend == "dnabert2":
        from DNABERT2_modules import load_dnabert2
        # Load the tokenizer first so pad_token_id can be baked into the config
        # before the model is built (the checkpoint doesn't carry it).
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False, local_files_only=local,
        )
        model, _ = load_dnabert2(
            repo_id=model_path,
            add_pooling_layer=False,
            config_overrides={"pad_token_id": tokenizer.pad_token_id, "attn_impl": attn_impl},
            device=device,
        )
        return model, tokenizer

    if backend == "carbon":
        from CARBON_modules import load_carbon
        # dtype=None lets load_carbon pick its bfloat16 default (the published
        # Carbon recipe); pass dtype through to override.
        model, tokenizer = load_carbon(repo_id=model_path, device=device, dtype=dtype)
        return model, tokenizer

    # plantcad
    from PlantCAD_modules import load_plantcad
    model, tokenizer = load_plantcad(repo_id=model_path, device=device, dtype=dtype)
    return model, tokenizer


def build_window_embedder(
    backend: str = "dnabert2",
    model_path: str | None = None,
    *,
    device: str | torch.device | None = None,
    max_length: int = 2048,
    snp_only: bool = False,
    output_layer: int | None = None,
    dtype: torch.dtype | None = None,
    attn_impl: str = "torch",
) -> WindowEmbedder:
    """Load an encoder and wrap it in a WindowEmbedder (moved to `device` if given)."""
    model, tokenizer = load_encoder_model(
        backend, model_path, device=device, dtype=dtype, attn_impl=attn_impl,
    )
    embedder = WindowEmbedder(
        model, tokenizer, max_length=max_length,
        snp_only=snp_only, output_layer=output_layer, backend=backend,
    )
    return embedder.to(device) if device is not None else embedder

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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=local,
        )
    except AttributeError:
        # transformers <4.43 crashes in from_pretrained when a safetensors
        # checkpoint was saved without a `__metadata__` header: it does
        # `metadata.get("format")` on a `None` metadata. Carbon's published
        # model.safetensors has no metadata, so build the (vanilla Llama) model
        # from config and load the weights directly, bypassing that code path.
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            repo_id, trust_remote_code=True, local_files_only=local
        )
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        state_dict = _download_carbon_weights(repo_id)
        # `lm_head.weight` is tied to the embeddings and absent from the file;
        # load non-strict and re-tie.
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=dtype).eval()
    return model, tokenizer


def load_carbon_local(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    config_overrides: dict | None = None,
):
    """
    Load Carbon weights into the local, hackable `CarbonForCausalLM`
    (`CARBON_modules.carbon_layers`) instead of the stock transformers Llama
    classes. This is the analogue of `DNABERT2_modules.load_dnabert2_mlm`: same
    HuggingFace checkpoint, but materialized in our own modeling code so a
    customized (variant-cache) forward pass can be built on top of it.

    Carbon's `config.json` declares a vanilla `LlamaForCausalLM`, so the
    safetensors keys (`model.embed_tokens`, `model.layers.*`, `model.norm`)
    line up with `CarbonForCausalLM` attribute-for-attribute and load with no
    remapping. The checkpoint ships tied embeddings, so `lm_head.weight` is
    absent from the file and is re-tied here after loading.

    Outputs match `AutoModelForCausalLM` (eager) to fp32 tolerance — see
    `CARBON_modules/test_carbon_equivalence.py`.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo or local directory. Defaults to the 500M checkpoint.
    device : str or torch.device, optional
        Defaults to CUDA when available, else CPU.
    dtype : torch.dtype, optional
        Compute/storage dtype. Defaults to ``torch.float32`` (the checkpoint's
        own dtype, and what the equivalence tests run in); pass
        ``torch.bfloat16`` to match `load_carbon`'s runtime default.
    config_overrides : dict, optional
        Fields to set on the LlamaConfig before building the model.

    Returns
    -------
    model : eval-mode CarbonForCausalLM on `device`.
    tokenizer : AutoTokenizer for Carbon's dual-mode (text / 6-mer DNA) vocab.
    """
    from transformers import LlamaConfig

    from .carbon_layers import CarbonForCausalLM

    local = os.path.isdir(repo_id)
    config = LlamaConfig.from_pretrained(repo_id, local_files_only=local)
    if config_overrides:
        for key, val in config_overrides.items():
            setattr(config, key, val)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, local_files_only=local
    )

    model = CarbonForCausalLM(config)
    state_dict = _download_carbon_weights(repo_id)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # `lm_head.weight` is tied to the embeddings, so it's expected to be absent
    # from the checkpoint; re-tie and treat only other gaps as noteworthy.
    model.tie_weights()
    non_trivial_missing = [k for k in missing if "lm_head.weight" not in k]
    if non_trivial_missing:
        import warnings
        warnings.warn(f"Missing keys not in checkpoint (random init): {non_trivial_missing}")
    if unexpected:
        import warnings
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected}")

    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=dtype).eval()
    return model, tokenizer


def load_carbon_variant_cache(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    backend: str = "efficient",
    config_overrides: dict | None = None,
):
    """
    Load Carbon as a `VariantCacheCarbonForCausalLM` — the variant-cache encoder
    plus the (tied) LM head. Analogue of
    `DNABERT2_modules.load_dnabert2_variant_cache_mlm`: same checkpoint as
    `load_carbon_local`, but the forward pass recomputes a set of variant
    positions against a frozen reference window (see
    `CARBON_modules/variant_cache_layers.py`).

    The variant-cache encoder reuses the standard `CarbonDecoderLayer`, so the
    Llama checkpoint keys (`model.embed_tokens`, `model.layers.*`, `model.norm`)
    map onto `embed_tokens` / `encoder.layers.*` / `encoder.norm` after stripping
    the leading ``model.`` prefix. ``lm_head.weight`` is tied (absent from the
    checkpoint) and re-tied here.

    backend : 'efficient' (log-sum-exp merge) or 'bruteforce' (oracle).
    """
    from transformers import LlamaConfig

    from .variant_cache_layers import VariantCacheCarbonForCausalLM

    local = os.path.isdir(repo_id)
    config = LlamaConfig.from_pretrained(repo_id, local_files_only=local)
    if config_overrides:
        for key, val in config_overrides.items():
            setattr(config, key, val)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, local_files_only=local
    )

    model = VariantCacheCarbonForCausalLM(config)
    model.set_backend(backend)

    state_dict = _download_carbon_weights(repo_id)
    # 'model.embed_tokens.*' -> 'embed_tokens.*'; 'model.layers.*'/'model.norm.*'
    # -> 'encoder.layers.*'/'encoder.norm.*'. Everything else (none here) as-is.
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model.embed_tokens."):
            remapped[k[len("model."):]] = v
        elif k.startswith("model.layers.") or k.startswith("model.norm."):
            remapped["encoder." + k[len("model."):]] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    model.tie_weights()
    non_trivial_missing = [k for k in missing if "lm_head.weight" not in k]
    if non_trivial_missing:
        import warnings
        warnings.warn(f"Missing keys not in checkpoint (random init): {non_trivial_missing}")
    if unexpected:
        import warnings
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected}")

    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=dtype).eval()
    return model, tokenizer


def _download_carbon_weights(repo_id: str) -> dict[str, torch.Tensor]:
    """Load Carbon weights from a local directory or the HuggingFace Hub."""
    if os.path.isdir(repo_id):
        safetensors_path = os.path.join(repo_id, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            return load_file(safetensors_path)
        bin_path = os.path.join(repo_id, "pytorch_model.bin")
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id, "model.safetensors")
        from safetensors.torch import load_file
        return load_file(path)
    except Exception:
        pass

    path = hf_hub_download(repo_id, "pytorch_model.bin")
    return torch.load(path, map_location="cpu", weights_only=True)


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

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .configuration_bert import BertConfig
from .bert_layers import BertForMaskedLM, BertModel

_DEFAULT_REPO = "zhihan1996/DNABERT-2-117M"


def load_dnabert2(
    repo_id: str = _DEFAULT_REPO,
    config_overrides: dict | None = None,
    add_pooling_layer: bool = False,
    device: str | torch.device | None = None,
) -> tuple[BertModel, AutoTokenizer]:
    """
    Load DNABERT-2 pretrained weights from HuggingFace into the local BertModel.

    The HuggingFace checkpoint was saved from BertForMaskedLM, so all encoder
    weights carry a 'bert.' prefix that is stripped here before loading into the
    bare BertModel.

    Attention runs on the pure-PyTorch matmul path by default (attn_impl="torch"):
    it needs neither CUDA nor the flash-attn package, and avoids the bundled
    flash_attn_triton.py, which targets Triton ~1.x/2.x and produces
    non-deterministic garbage on Triton >= 3.x. Pass
    ``config_overrides={"attn_impl": "flash"}`` (or "triton") to opt back in.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID, defaults to zhihan1996/DNABERT-2-117M.
    config_overrides : dict, optional
        Fields to override on the loaded BertConfig before instantiating the
        model. Architecture-altering overrides (e.g. num_hidden_layers) are
        loaded with strict=False so partial weight tables are accepted.
    add_pooling_layer : bool
        Include the BertPooler head. The pretrained checkpoint has no pooler
        weights, so this initializes it randomly.
    device : str or torch.device, optional
        Target device. Defaults to CUDA if available, else CPU.

    Returns
    -------
    model : BertModel (eval mode, on device)
    tokenizer : AutoTokenizer
    """
    import os
    local = os.path.isdir(repo_id)
    config = BertConfig.from_pretrained(repo_id, local_files_only=local)
    config.attn_impl = "torch"  # pure-PyTorch matmul path; override below if set
    if config_overrides:
        for key, val in config_overrides.items():
            setattr(config, key, val)

    model = BertModel(config, add_pooling_layer=add_pooling_layer)

    state_dict = _download_weights(repo_id)

    # Checkpoint was saved as BertForMaskedLM; encoder lives under 'bert.*'
    encoder_sd = {
        k[len("bert."):]: v
        for k, v in state_dict.items()
        if k.startswith("bert.")
    }

    strict = config_overrides is None or not _architecture_changed(config_overrides)
    missing, unexpected = model.load_state_dict(encoder_sd, strict=strict)

    if unexpected:
        import warnings
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected}")
    if missing:
        import warnings
        warnings.warn(f"Missing keys not in checkpoint (random init): {missing}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, use_fast=False, local_files_only=local)
    return model, tokenizer


def load_dnabert2_mlm(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    config_overrides: dict | None = None,
) -> tuple[BertForMaskedLM, AutoTokenizer]:
    """
    Load DNABERT-2 as BertForMaskedLM (encoder + MLM prediction head).

    The checkpoint was saved from BertForMaskedLM, so the cls.* keys map
    directly onto the model without any prefix manipulation.
    Use this instead of load_dnabert2 when you need output logits (e.g.
    pseudo-perplexity, masked token prediction).

    config_overrides : fields to set on the config before building the model.
        Attention defaults to the pure-PyTorch matmul path ("torch"); pass
        e.g. ``{"attn_impl": "flash"}`` to opt into flash-attn (needs CUDA +
        the flash-attn package).
    """
    import os
    local = os.path.isdir(repo_id)
    config = BertConfig.from_pretrained(repo_id, local_files_only=local)
    config.attn_impl = "torch"  # pure-PyTorch matmul path; override below if set
    if config_overrides:
        for key, val in config_overrides.items():
            setattr(config, key, val)

    # transformers >= 5 no longer auto-populates these on the config, and the
    # DNABERT-2 checkpoint's config.json doesn't carry them. BertForMaskedLM /
    # BertEmbeddings read both, so set them before building the model: pad_token_id
    # from the tokenizer, is_decoder=False (this is a bidirectional MLM encoder).
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, use_fast=False, local_files_only=local
    )
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id
    config.is_decoder = getattr(config, "is_decoder", False)

    model = BertForMaskedLM(config)
    state_dict = _download_weights(repo_id)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    non_trivial_missing = [k for k in missing if "decoder.weight" not in k]
    if non_trivial_missing:
        import warnings
        warnings.warn(f"Missing keys not in checkpoint (random init): {non_trivial_missing}")
    if unexpected:
        import warnings
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    return model, tokenizer


def load_dnabert2_variant_cache_mlm(
    repo_id: str = _DEFAULT_REPO,
    device: str | torch.device | None = None,
    backend: str = "efficient",
    config_overrides: dict | None = None,
):
    """
    Load DNABERT-2 as a VariantCacheBertForMaskedLM — the MLM head on top of the
    variant-cache encoder. Drop-in for load_dnabert2_mlm in the variant-MLM
    scripts (same call + MaskedLMOutput return), used to measure variant-MLM loss
    *through the cache approximation*.

    The DNABERT-2 checkpoint stores encoder/embeddings under a 'bert.' prefix and
    the MLM head under 'cls.'. The variant-cache encoder reuses the standard
    BertLayer, so 'bert.encoder.layer.*' maps straight onto 'encoder.layer.*'.

    backend          : 'efficient' (log-sum-exp merge) or 'bruteforce' (oracle).
    config_overrides : attention defaults to the pure-PyTorch matmul path
        ("torch"); pass e.g. {"attn_impl": "flash"} to opt into flash-attn.
    """
    import os
    from .variant_cache_layers import VariantCacheBertForMaskedLM

    local = os.path.isdir(repo_id)
    config = BertConfig.from_pretrained(repo_id, local_files_only=local)
    config.attn_impl = "torch"  # pure-PyTorch matmul path; override below if set
    if config_overrides:
        for key, val in config_overrides.items():
            setattr(config, key, val)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, use_fast=False, local_files_only=local
    )
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id

    model = VariantCacheBertForMaskedLM(config)
    model.set_backend(backend)

    state_dict = _download_weights(repo_id)
    # Strip the 'bert.' prefix so encoder/embeddings line up; keep 'cls.*' as-is.
    remapped = {
        (k[len("bert."):] if k.startswith("bert.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    non_trivial_missing = [k for k in missing if "decoder.weight" not in k]
    if non_trivial_missing:
        import warnings
        warnings.warn(f"Missing keys not in checkpoint (random init): {non_trivial_missing}")
    if unexpected:
        import warnings
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    return model, tokenizer


def _download_weights(repo_id: str) -> dict[str, torch.Tensor]:
    """Load model weights from a local directory or HuggingFace Hub."""
    import os

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


_ARCHITECTURE_KEYS = {
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
}


def _architecture_changed(overrides: dict) -> bool:
    return bool(_ARCHITECTURE_KEYS & overrides.keys())

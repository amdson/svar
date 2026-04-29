from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .configuration_bert import BertConfig
from .bert_layers import BertModel

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

    The bundled flash_attn_triton.py targets Triton ~1.x/2.x and is incompatible
    with Triton >= 3.x, producing non-deterministic garbage outputs. This loader
    disables it by setting p_dropout=1e-9 on every attention layer, which routes
    each forward pass through the standard PyTorch attention path. nn.Dropout is
    a no-op in eval() mode, so this has no effect on outputs.

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

    _disable_triton_attention(model)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, use_fast=False, local_files_only=local)
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


def _disable_triton_attention(model) -> None:
    for module in model.modules():
        if hasattr(module, "p_dropout"):
            module.p_dropout = 1e-9

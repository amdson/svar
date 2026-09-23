"""Backbone-agnostic per-site embeddings for fixed-length accession windows.

The only backbone-specific knowledge needed for a *site* read-out is where the
centre base lands in the token stream. Two mechanisms cover every model here:

  * fast tokenizers expose ``offset_mapping`` -> exact char->token spans (BPE etc.)
  * fixed-width tokenizers (Carbon 6-mers after a ``<dna>`` marker, single-base
    GPN / PlantCAD) are probed once with two synthetic sequences to recover
    ``prefix_tokens`` and ``chars_per_token`` analytically.

``embed(seqs)`` returns, per sequence, the hidden state averaged over the tokens
covering the centre base +- ``pool_bp`` bases. With ``bidir=True`` the same
read-out on the reverse complement is concatenated, which matters for causal
models (Carbon): the forward pass only sees the left context of the centre.

Backends
  carbon    HuggingFaceBio/Carbon-*         decoder-only, 6-mer tokens, bf16
  gpn       songlab/gpn-brassicales (etc.)  needs ``pip install gpn`` (registers the
                                            ConvNet AutoModel class)
  plantcad  kuleshov-group/PlantCaduceus_*  needs mamba-ssm; RC-averaged hidden
  hf        any AutoModel with a fast tokenizer (DNABERT-2, NT, ...)
"""
from __future__ import annotations

import numpy as np
import torch

_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_RC)[::-1]


DEFAULT_MODEL_PATHS = {
    "carbon": "HuggingFaceBio/Carbon-500M",
    "gpn": "songlab/gpn-brassicales",
    "plantcad": "kuleshov-group/PlantCaduceus_l32",
    "hf": None,
}


class SiteEmbedder:
    def __init__(self, backend: str, model_path: str | None = None, *,
                 device=None, dtype=None, layer: int = -1, pool_bp: int = 0,
                 bidir: bool = False, batch_size: int = 64):
        if backend not in DEFAULT_MODEL_PATHS:
            raise ValueError(f"unknown backend {backend!r}; one of {list(DEFAULT_MODEL_PATHS)}")
        self.backend = backend
        self.model_path = model_path or DEFAULT_MODEL_PATHS[backend]
        if self.model_path is None:
            raise ValueError("backend 'hf' needs --model-path")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.layer = layer
        self.pool_bp = pool_bp
        self.bidir = bidir
        self.batch_size = batch_size
        self.model, self.tokenizer = self._load()
        self.model.eval()
        self.prefix_tokens, self.chars_per_token = self._probe_layout()

    # ------------------------------------------------------------ loading
    def _load(self):
        import os
        local = os.path.isdir(self.model_path)
        if self.backend == "carbon":
            from CARBON_modules import load_carbon
            return load_carbon(repo_id=self.model_path, device=self.device, dtype=self.dtype)
        if self.backend == "plantcad":
            from PlantCAD_modules import load_plantcad
            return load_plantcad(repo_id=self.model_path, device=self.device, dtype=self.dtype)
        from transformers import AutoModel, AutoTokenizer
        if self.backend == "gpn":
            try:
                import gpn.model  # noqa: F401  registers GPN model classes with AutoModel
            except ImportError as e:
                raise ImportError("backend 'gpn' needs the gpn package: "
                                  "pip install git+https://github.com/songlab-cal/gpn.git") from e
        kw = {"trust_remote_code": True, "local_files_only": local}
        if self.dtype is not None:
            kw["torch_dtype"] = self.dtype
        tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=local)
        model = AutoModel.from_pretrained(self.model_path, **kw).to(self.device)
        return model, tok

    def _tok_kwargs(self, seqs):
        kw = dict(return_tensors="pt", padding="longest")
        if self.backend == "carbon":
            seqs = ["<dna>" + s for s in seqs]
            kw["add_special_tokens"] = False
        return seqs, kw

    def _probe_layout(self):
        """Infer (prefix_tokens, chars_per_token) for fixed-width tokenizers."""
        if getattr(self.tokenizer, "is_fast", False) and self.backend != "carbon":
            return None, None  # offsets available; located exactly per batch
        n = []
        for L in (120, 240):
            s, kw = self._tok_kwargs(["ACGT" * (L // 4)])
            n.append(self.tokenizer(s, **kw)["input_ids"].shape[1])
        per = 120 / (n[1] - n[0])
        if abs(per - round(per)) > 1e-6:
            raise RuntimeError(f"cannot infer fixed token width for {self.backend}: {n}")
        cpt = int(round(per))
        prefix = n[0] - 120 // cpt
        return prefix, cpt

    # ------------------------------------------------------------ forward
    def _hidden(self, inputs) -> torch.Tensor:
        if self.backend == "carbon":
            out = self.model(**inputs, output_hidden_states=True)
            return out.hidden_states[self.layer].float()
        if self.backend == "plantcad":
            from PlantCAD_modules import average_rc_embeddings
            out = self.model(**inputs, output_hidden_states=True)
            return average_rc_embeddings(out.hidden_states[self.layer].float())
        out = self.model(**inputs, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is not None and len(hs) > 0:
            return hs[self.layer].float()
        return out.last_hidden_state.float()

    def _token_mask(self, offsets, B: int, T: int, lo: int, hi: int) -> torch.Tensor:
        """Bool (B, T): tokens whose char span intersects [lo, hi)."""
        if offsets is not None:
            st, en = offsets[..., 0], offsets[..., 1]
            m = (en > lo) & (st < hi) & (en > st)
        else:
            t0 = self.prefix_tokens + lo // self.chars_per_token
            t1 = self.prefix_tokens + (hi - 1) // self.chars_per_token
            m = torch.zeros(B, T, dtype=torch.bool)
            m[:, t0:t1 + 1] = True
        return m

    def _readout(self, seqs: list[str], lo: int, hi: int) -> torch.Tensor:
        seqs2, kw = self._tok_kwargs(seqs)
        use_offsets = self.prefix_tokens is None
        if use_offsets:
            kw["return_offsets_mapping"] = True
        inputs = self.tokenizer(seqs2, **kw)
        offsets = inputs.pop("offset_mapping") if use_offsets else None
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        h = self._hidden(inputs)                       # (B, T, D)
        m = self._token_mask(offsets, h.shape[0], h.shape[1], lo, hi).to(h.device)
        if not bool(m.any(dim=1).all()):
            raise RuntimeError("centre span maps to no token for some sequence")
        m = m.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)

    @torch.no_grad()
    def embed(self, seqs: list[str]) -> np.ndarray:
        """(B, D) or (B, 2D) float32. All seqs must share one length; centre = len//2."""
        if not seqs:
            return np.zeros((0, self.dim), np.float32)
        L = len(seqs[0])
        assert all(len(s) == L for s in seqs), "windows must share one length"
        c = L // 2
        lo, hi = max(0, c - self.pool_bp), min(L, c + self.pool_bp + 1)
        outs = []
        for i in range(0, len(seqs), self.batch_size):
            chunk = seqs[i:i + self.batch_size]
            f = self._readout(chunk, lo, hi)
            if self.bidir:
                r = self._readout([revcomp(s) for s in chunk], L - hi, L - lo)
                f = torch.cat([f, r], dim=1)
            outs.append(f.cpu())
        return torch.cat(outs).numpy().astype(np.float32)

    @property
    def dim(self) -> int:
        d = self.embed(["ACGT" * 32]).shape[1] if not hasattr(self, "_dim") else self._dim
        self._dim = d
        return d

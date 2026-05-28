"""
crop_embed/embedder.py
----------------------
SampleEmbedder: aggregates window embeddings into one vector per sample.

After UniqueWindowDataset items have been embedded (e.g. with DNABERT-2) and
stored in an embedding_table keyed by fingerprint, SampleEmbedder looks up
each sample's windows, retrieves the cached embeddings, and pools them.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn

from crop_embed.dataset import UniqueWindowDataset
from crop_embed.fingerprint import Fingerprint


Backend = Literal["dnabert2", "plantcad"]


class WindowEmbedder(nn.Module):
    """
    Tokenize → forward → masked pool, returning one vector per input sequence.

    This is the abstraction boundary between the data pipeline and the DNA
    language model. Two backends are supported:

      * ``backend="dnabert2"`` (default) — HuggingFace-style model whose forward
        returns an object with ``last_hidden_state`` of shape (B, T, D). If
        ``output_layer`` is set, the model must additionally accept that kwarg
        and return an ``intermediate_hidden_state`` field. DNABERT-2 satisfies
        both.

      * ``backend="plantcad"`` — PlantCaduceus (kuleshov-group/PlantCaduceus_*).
        Hidden states come from ``output_hidden_states=True``; reverse-complement
        parameter sharing means the (B, T, 2D) last hidden state is averaged
        with its RC-half-flip to (B, T, D) via
        :func:`PlantCAD_modules.average_rc_embeddings`. ``output_layer`` indexes
        into ``outputs.hidden_states`` (negatives count from the end). Caduceus
        tokenizers don't expose offset mappings, so ``snp_only`` isn't
        supported on this backend yet.

    Parameters
    ----------
    model        : the underlying DNA encoder (nn.Module)
    tokenizer    : matching HuggingFace tokenizer
    max_length   : tokenizer max_length (windows longer than this are truncated)
    snp_only     : pool only over tokens whose character span contains a SNP.
                   Windows with no alt alleles fall back to full-attention pool.
                   Requires ``backend="dnabert2"``.
    output_layer : if set, pull hidden states from this encoder layer index
                   (0-based; negative counts from the end) instead of the final.
    backend      : which encoder family to drive — ``"dnabert2"`` or
                   ``"plantcad"``. Controls how ``model(...)`` is called and
                   which output field is read.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_length: int = 512,
        snp_only: bool = False,
        output_layer: int | None = None,
        backend: Backend = "dnabert2",
    ) -> None:
        super().__init__()
        if backend not in ("dnabert2", "plantcad"):
            raise ValueError(f"Unknown backend {backend!r}; expected 'dnabert2' or 'plantcad'.")
        if snp_only and backend == "plantcad":
            raise NotImplementedError(
                "snp_only=True is not supported with backend='plantcad' — the "
                "Caduceus tokenizer doesn't return offset_mapping. Pass "
                "snp_only=False or fall back to the dnabert2 backend."
            )
        self.model        = model
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.snp_only     = snp_only
        self.output_layer = output_layer
        self.backend      = backend

    def forward(
        self,
        sequences: list[str],
        fingerprints: list[Fingerprint],
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device

        tok_kwargs = dict(
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )
        if self.snp_only:
            tok_kwargs["return_offsets_mapping"] = True

        inputs = self.tokenizer(sequences, **tok_kwargs)
        if self.snp_only:
            offset_mapping = inputs.pop("offset_mapping")    # not a model input
            snp_mask = _build_snp_mask(offset_mapping, fingerprints)
        inputs = inputs.to(device)

        if self.backend == "dnabert2":
            fwd_kwargs = {"output_layer": self.output_layer} if self.output_layer is not None else {}
            outputs   = self.model(**inputs, **fwd_kwargs)
            if self.output_layer is not None and outputs.intermediate_hidden_state is not None:
                hidden = outputs.intermediate_hidden_state
            else:
                hidden = outputs.last_hidden_state
        else:  # plantcad — Caduceus exposes layers via output_hidden_states
            from PlantCAD_modules import average_rc_embeddings
            outputs   = self.model(**inputs, output_hidden_states=True)
            layer_idx = -1 if self.output_layer is None else self.output_layer
            # Hidden states are (B, T, 2D) due to RC parameter sharing; collapse
            # to single-strand (B, T, D) before pooling. Cast up so downstream
            # statistics aren't quantized by an fp16 checkpoint.
            hidden = average_rc_embeddings(outputs.hidden_states[layer_idx].float())

        attn_mask = inputs["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
        if self.snp_only:
            s_mask    = snp_mask.to(device).unsqueeze(-1).float()
            # Pure-reference windows have no SNP tokens: fall back to full-window pool.
            has_snp   = s_mask.squeeze(-1).any(dim=1, keepdim=True).unsqueeze(-1)
            pool_mask = torch.where(has_snp, s_mask, attn_mask)
        else:
            pool_mask = attn_mask

        return (hidden * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)


class BatchedWindowEmbedder(nn.Module):
    """
    Sample-batch interface around a WindowEmbedder + dataset.

    `forward(global_indices)` returns the per-window embeddings for a batch of
    samples — shape (B, n_windows, D). Deduplicates windows across the batch
    via `dataset.gather_batch`, runs the underlying WindowEmbedder once, then
    scatters back. Use for end-to-end / trainable-embedder training.
    """

    def __init__(self, window_embedder: WindowEmbedder, dataset: UniqueWindowDataset) -> None:
        super().__init__()
        self.window_embedder = window_embedder
        # Not registered as a submodule — it has no parameters/buffers and
        # holds file handles (FASTA) that shouldn't move with .to(device).
        self._dataset = dataset

    @property
    def dataset(self) -> UniqueWindowDataset:
        return self._dataset

    def forward(self, global_indices: torch.Tensor) -> torch.Tensor:
        sequences, fingerprints, inverse = self._dataset.gather_batch(global_indices.cpu())
        emb = self.window_embedder(sequences, fingerprints)
        return emb[inverse.to(emb.device)]


class FixedWindowEmbedder(nn.Module):
    """
    Precomputed per-sample window embeddings — drop-in for training a head
    against a frozen embedder without re-running the backbone.

    `forward(global_indices)` returns (B, n_windows, D) via a single tensor
    index. No parameters; the cache and sample→fingerprint index are stored
    as buffers, so `state_dict()` round-trips the full precomputed table.

    Parameters
    ----------
    cache            : Tensor(n_unique_windows, D) of precomputed embeddings,
                       indexed in the same order as `dataset.unique_fingerprints`.
    sample_fp_index  : LongTensor(n_samples, n_windows) — `dataset.sample_fp_index`.
    """

    def __init__(self, cache: torch.Tensor, sample_fp_index: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("cache", cache)
        self.register_buffer("sample_fp_index", sample_fp_index)

    def forward(self, global_indices: torch.Tensor) -> torch.Tensor:
        return self.cache[self.sample_fp_index[global_indices]]

    @classmethod
    def from_embedder(
        cls,
        window_embedder: WindowEmbedder,
        dataset: UniqueWindowDataset,
        batch_size: int = 64,
        device: str | torch.device | None = None,
    ) -> "FixedWindowEmbedder":
        """Run window_embedder over every unique window and build the cache."""
        if device is None:
            device = next(window_embedder.parameters()).device
        else:
            device = torch.device(device)

        was_training = window_embedder.training
        window_embedder.to(device).eval()

        fps = dataset.unique_fingerprints
        vecs = []
        with torch.no_grad():
            for i in range(0, len(fps), batch_size):
                batch_fps = fps[i : i + batch_size]
                seqs = [dataset.extract_sequence(fp) for fp in batch_fps]
                vecs.append(window_embedder(seqs, batch_fps).cpu())
        cache = torch.cat(vecs, dim=0)

        window_embedder.train(was_training)
        return cls(cache, dataset.sample_fp_index.clone())

    @classmethod
    def from_embedding_table(
        cls,
        embedding_table: dict[Fingerprint, torch.Tensor],
        dataset: UniqueWindowDataset,
    ) -> "FixedWindowEmbedder":
        """Build from a SampleEmbedder.fill_embedding_table-style dict."""
        cache = torch.stack([embedding_table[fp] for fp in dataset.unique_fingerprints])
        return cls(cache, dataset.sample_fp_index.clone())

    @classmethod
    def from_checkpoint(cls, path: str) -> "FixedWindowEmbedder":
        """
        Reconstruct from a training checkpoint that was saved while training
        a FixedWindowEmbedder (so `embedder_state_dict` carries the cache).
        Avoids re-running precompute on resume.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd   = ckpt["embedder_state_dict"]
        return cls(sd["cache"], sd["sample_fp_index"])

    @classmethod
    def from_file(
        cls,
        path: str,
        dataset: UniqueWindowDataset | None = None,
    ) -> "FixedWindowEmbedder":
        """
        Load from disk, auto-detecting format:
          1. state_dict from FixedWindowEmbedder.save()  → preferred
          2. training checkpoint with embedder_state_dict (phase1/cached ckpt.pt)
          3. SampleEmbedder.fill_embedding_table dict    → requires `dataset`
        """
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            if "cache" in obj and "sample_fp_index" in obj:
                return cls(obj["cache"], obj["sample_fp_index"])
            if "embedder_state_dict" in obj:
                sd = obj["embedder_state_dict"]
                return cls(sd["cache"], sd["sample_fp_index"])
        if dataset is None:
            raise ValueError(
                f"{path} looks like a fill_embedding_table dict; pass `dataset` "
                "to align fingerprints, or save a FixedWindowEmbedder.state_dict()."
            )
        return cls.from_embedding_table(obj, dataset)

    def save(self, path: str) -> None:
        """Atomically write this embedder's cache to disk for later reuse."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)


class SampleEmbedder:
    """
    Parameters
    ----------
    dataset         : the UniqueWindowDataset used to produce the embeddings
    embedding_table : dict mapping Fingerprint → 1-D Tensor (embedding_dim,)
    aggregation     : pooling strategy across windows — "mean" or "sum"
    """

    def __init__(
        self,
        dataset: UniqueWindowDataset,
        embedding_table: dict[Fingerprint, torch.Tensor],
        aggregation: Literal["mean", "sum"] = "mean",
    ) -> None:
        self.dataset          = dataset
        self.embedding_table  = embedding_table
        self.aggregation      = aggregation

    def embed_sample(self, sample_id: str) -> torch.Tensor:
        """
        Aggregate all window embeddings for one sample.

        Windows whose fingerprint is missing from embedding_table are skipped
        with a warning rather than crashing, so a partial embedding table can
        still produce useful (though incomplete) vectors.

        Returns
        -------
        Tensor of shape (embedding_dim,)

        Raises
        ------
        ValueError if no embeddings are found for the sample at all.
        """
        sample_idx  = self.dataset.samples.index(sample_id)
        partitioner = self.dataset.partitioner

        vecs: list[torch.Tensor] = []
        missing = 0

        for window in partitioner:
            key = (sample_idx, window.index)
            fp  = self.dataset.sample_window_to_fp.get(key)
            # if fp is None or not fp[3]:  # skip pure-reference windows — no sample-specific signal
            #     continue
            if fp is None:
                raise ValueError(f"Missing fingerprint for {window}. ({key})")
            vec = self.embedding_table.get(fp)
            if vec is None:
                missing += 1
                continue
            vecs.append(vec)

        if not vecs:
            raise ValueError(
                f"No embeddings found for sample '{sample_id}'. "
                "Run the embedding step before calling SampleEmbedder."
            )
        if missing:
            import warnings
            warnings.warn(
                f"Sample '{sample_id}': {missing} windows missing from "
                "embedding_table and were skipped."
            )

        stacked = torch.stack(vecs, dim=0)   # (n_windows, embedding_dim)
        if self.aggregation == "mean":
            return stacked.mean(dim=0)
        elif self.aggregation == "sum":
            return stacked.sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    def embed_all(
        self, samples: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Embed every sample (or a provided subset).

        Returns
        -------
        {sample_id: Tensor(embedding_dim,)}
        """
        targets = samples if samples is not None else self.dataset.samples
        return {s: self.embed_sample(s) for s in targets}

    # ── Convenience: run model + fill embedding_table in one call ─────────────

    @staticmethod
    def fill_embedding_table(
        dataset: UniqueWindowDataset,
        model,
        tokenizer,
        batch_size: int = 64,
        max_length: int = 512,
        device: str | torch.device | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 500,
        snp_only: bool = False,
        output_layer: int | None = None,
        backend: Backend = "dnabert2",
    ) -> dict[Fingerprint, torch.Tensor]:
        """
        Run `model` over every unique window in `dataset` and return a
        completed embedding_table ready for SampleEmbedder.

        Parameters
        ----------
        model            : a HuggingFace model returning last_hidden_state
                           (``backend="dnabert2"``) or one supporting
                           ``output_hidden_states=True`` (``backend="plantcad"``).
        tokenizer        : matching HuggingFace tokenizer
        batch_size       : sequences per forward pass
        max_length       : tokenizer max_length
        device           : torch device; defaults to CUDA if available
        checkpoint_path  : if set, periodically save the embedding table here
                           and resume from it automatically on restart
        checkpoint_every : save a checkpoint every this many batches
        snp_only         : if True, pool only over tokens whose character span
                           contains a SNP position rather than all tokens.
                           Windows with no alt alleles fall back to full-window
                           pooling. Incompatible with checkpoints produced
                           without this flag — delete any existing checkpoint
                           before switching modes. Requires
                           ``backend="dnabert2"``.
        output_layer     : if set, extract hidden states from this encoder layer
                           index (0-based; negative indices count from the end)
                           instead of the final layer. Incompatible with
                           checkpoints produced with a different layer setting.
        backend          : ``"dnabert2"`` (default) or ``"plantcad"`` — see
                           :class:`WindowEmbedder`. PlantCAD outputs are RC-
                           averaged inside the embedder, so the returned
                           tensors are ``hidden_size // 2`` wide.

        Returns
        -------
        embedding_table : {Fingerprint: Tensor(embedding_dim,)}
        """
        import os
        from torch.utils.data import DataLoader, Subset
        from tqdm import tqdm

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        embedder = WindowEmbedder(
            model, tokenizer, max_length=max_length,
            snp_only=snp_only, output_layer=output_layer,
            backend=backend,
        ).to(device).eval()

        # ── Resume from checkpoint if one exists ─────────────────────────────
        embedding_table: dict[Fingerprint, torch.Tensor] = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            embedding_table = torch.load(checkpoint_path, weights_only=False)
            print(f"Resumed from checkpoint: {len(embedding_table):,} windows already embedded")

        already_done = set(embedding_table.keys())
        remaining_indices = [
            i for i, fp in enumerate(dataset.unique_fingerprints)
            if fp not in already_done
        ]

        if not remaining_indices:
            print("All windows already embedded; returning checkpoint table.")
            return embedding_table

        subset = Subset(dataset, remaining_indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,       # FASTA not fork-safe; use 0 or spawn workers
            collate_fn=_collate,
        )

        def _save_checkpoint():
            tmp = checkpoint_path + ".tmp"
            torch.save(embedding_table, tmp)
            os.replace(tmp, checkpoint_path)  # atomic on POSIX

        n_remaining = len(remaining_indices)
        batches_since_save = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Embedding windows", unit="batch",
                              total=(n_remaining + batch_size - 1) // batch_size):
                sequences    = batch["sequences"]      # list[str]
                fingerprints = batch["fingerprints"]   # list[Fingerprint]

                vecs = embedder(sequences, fingerprints).cpu()

                for fp, vec in zip(fingerprints, vecs):
                    embedding_table[fp] = vec

                if checkpoint_path:
                    batches_since_save += 1
                    if batches_since_save >= checkpoint_every:
                        _save_checkpoint()
                        batches_since_save = 0

        if checkpoint_path:
            _save_checkpoint()

        return embedding_table


def _collate(items: list[dict]) -> dict:
    return {
        "sequences":    [item["sequence"]    for item in items],
        "fingerprints": [item["fingerprint"] for item in items],
        "chroms":       [item["chrom"]       for item in items],
        "starts":       [item["start"]       for item in items],
        "ends":         [item["end"]         for item in items],
        "indices":      [item["idx"]         for item in items],
    }


def _build_snp_mask(
    offset_mapping: torch.Tensor,
    fingerprints: list[Fingerprint],
) -> torch.Tensor:
    """
    Build a (B, T) bool mask that is True for tokens whose character span
    contains at least one SNP position.

    offset_mapping : (B, T, 2) int tensor from the tokenizer.  Special tokens
                     and padding entries both have offset (0, 0) and will never
                     match a SNP position (the interval [0, 0) is empty).
    fingerprints   : list of Fingerprint tuples (chrom, w_start, w_end, alt_positions).
                     alt_positions are 0-based genomic coordinates; the character
                     index within the sequence string is pos - w_start.
    """
    B, T, _ = offset_mapping.shape
    mask   = torch.zeros(B, T, dtype=torch.bool)
    starts = offset_mapping[:, :, 0]  # (B, T)
    ends   = offset_mapping[:, :, 1]  # (B, T)

    for b, fp in enumerate(fingerprints):
        alt_positions = fp[3]
        if not alt_positions:
            continue  # pure-reference window; caller falls back to attention mask

        w_start   = fp[1]
        snp_chars = torch.tensor([p - w_start for p in alt_positions], dtype=torch.long)

        # Broadcast (S, 1) against (1, T) to get (S, T) hit matrix
        p_col  = snp_chars.unsqueeze(1)    # (S, 1)
        s_row  = starts[b].unsqueeze(0)    # (1, T)
        e_row  = ends[b].unsqueeze(0)      # (1, T)
        mask[b] = ((s_row <= p_col) & (p_col < e_row)).any(dim=0)

    return mask

"""
crop_embed/embedder.py
----------------------
SampleEmbedder: aggregates window embeddings into one vector per sample.

After UniqueWindowDataset items have been embedded (e.g. with DNABERT-2) and
stored in an embedding_table keyed by fingerprint, SampleEmbedder looks up
each sample's windows, retrieves the cached embeddings, and pools them.
"""

from __future__ import annotations

from typing import Literal

import torch

from crop_embed.dataset import UniqueWindowDataset
from crop_embed.fingerprint import Fingerprint


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
            if fp is None:
                continue
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
    ) -> dict[Fingerprint, torch.Tensor]:
        """
        Run `model` over every unique window in `dataset` and return a
        completed embedding_table ready for SampleEmbedder.

        Parameters
        ----------
        model            : a HuggingFace model returning last_hidden_state
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
                           before switching modes.

        Returns
        -------
        embedding_table : {Fingerprint: Tensor(embedding_dim,)}
        """
        import os
        from torch.utils.data import DataLoader, Subset
        from tqdm import tqdm

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

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

                if snp_only:
                    inputs = tokenizer(
                        sequences,
                        return_tensors="pt",
                        padding="longest",
                        max_length=max_length,
                        truncation=True,
                        return_offsets_mapping=True,
                    )
                    offset_mapping = inputs.pop("offset_mapping")  # (B, T, 2); not a model input
                    snp_mask = _build_snp_mask(offset_mapping, fingerprints)  # (B, T) bool
                    inputs = inputs.to(device)
                else:
                    inputs = tokenizer(
                        sequences,
                        return_tensors="pt",
                        padding="longest",
                        max_length=max_length,
                        truncation=True,
                    ).to(device)

                outputs = model(**inputs)
                hidden  = outputs.last_hidden_state        # (B, T, D)

                if snp_only:
                    attn_mask = inputs["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
                    s_mask    = snp_mask.to(device).unsqueeze(-1).float()        # (B, T, 1)
                    # Windows with no SNPs in alt_positions: fall back to full-window pooling
                    has_snp   = s_mask.squeeze(-1).any(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
                    pool_mask = torch.where(has_snp, s_mask, attn_mask)
                else:
                    pool_mask = inputs["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)

                vecs = (hidden * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)
                vecs = vecs.cpu()

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

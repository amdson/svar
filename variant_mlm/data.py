"""
variant_mlm/data.py
-------------------
Dataset + masking collator for the *variant* MLM objective: mask the BPE tokens
that overlap a SNP/alt position in a window and ask the model to predict them.

This is the objective from the variant-cache design discussion — "predict the
variant nucleotide" — used both as a fine-tuning target and as a benign-ness
diagnostic for the cache approximation (compare variant-MLM loss through the
exact model vs. the cache model).

Windowing reuses the embed_windows machinery (SNPWindowPartitioner +
UniqueWindowDataset), so the windows here are identical to the ones the embedder
caches, with ``--half-window`` left as a parameter. We keep only the unique
windows that actually contain an alt allele (pure-reference windows carry no
variant-MLM signal).

The collator maps alt genomic positions to token indices with the tokenizer's
offset mapping (via crop_embed's ``_build_snp_mask``), then builds BERT-style
MLM ``labels``: the true token id at selected variant tokens and -100 elsewhere.
DNABERT2's BertForMaskedLM treats ``labels > 0`` as the masked set, so -100 is
ignored exactly as in standard HF MLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset

from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.dataset import UniqueWindowDataset
from crop_embed.embedder import _build_snp_mask
from crop_embed.partitioner import SNPWindowPartitioner

IGNORE_INDEX = -100


def build_variant_window_dataset(
    vcf_path: str,
    fasta_path: str,
    half_window: int,
    buffer: int = 0,
    samples: list[str] | None = None,
    variant_only: bool = True,
) -> tuple[UniqueWindowDataset, list[int]]:
    """Build the unique-window dataset and the indices to train/eval on.

    Returns
    -------
    dataset : UniqueWindowDataset (full dedup set, same as embed_windows builds)
    indices : list[int] into ``dataset.unique_fingerprints`` — when
              ``variant_only`` (default), restricted to windows with >=1 alt
              allele (the only ones that produce a variant-MLM target).
    """
    snps_by_chrom, _ = load_snps_from_vcf(vcf_path, samples)
    partitioner = SNPWindowPartitioner(
        snps_by_chrom, half_window=half_window, buffer=buffer
    )
    dataset = UniqueWindowDataset(vcf_path, fasta_path, partitioner, samples=samples)

    if variant_only:
        indices = [
            i for i, fp in enumerate(dataset.unique_fingerprints) if fp[3]
        ]
    else:
        indices = list(range(len(dataset.unique_fingerprints)))
    return dataset, indices


def split_indices(
    indices: list[int], val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic train/val split over a list of dataset indices."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(indices), generator=g).tolist()
    n_val = int(round(len(indices) * val_frac))
    val = [indices[i] for i in perm[:n_val]]
    train = [indices[i] for i in perm[n_val:]]
    return train, val


def _collate_windows(items: list[dict]) -> dict:
    return {
        "sequences": [it["sequence"] for it in items],
        "fingerprints": [it["fingerprint"] for it in items],
    }


def make_loader(dataset: UniqueWindowDataset, indices: list[int], batch_size: int,
                shuffle: bool):
    """DataLoader over a subset of windows yielding {sequences, fingerprints}.

    num_workers=0: UniqueWindowDataset holds pyfaidx FASTA handles that are not
    fork-safe (the embedder pipeline uses the same constraint).
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate_windows,
    )


@dataclass
class VariantMLMCollator:
    """Tokenize a batch of windows and build variant-MLM inputs/labels.

    Parameters
    ----------
    tokenizer            : DNABERT2 tokenizer (use_fast=False still returns
                           offset mappings, as the embedder's snp_only path relies on).
    max_length           : tokenizer truncation length.
    select_prob          : probability a variant token is chosen as a prediction
                           target. 1.0 = predict every variant token (the clean
                           diagnostic); <1.0 subsamples for noisier training.
    mask_replace_prob    : of the selected targets, fraction replaced with [MASK].
    random_replace_prob  : fraction replaced with a random token. The remainder
                           are left unchanged (the standard 80/10/10 split is
                           mask=0.8, random=0.1, keep=0.1). For evaluation use
                           mask_replace_prob=1.0, random_replace_prob=0.0 so the
                           loss is a clean masked-token prediction.
    generator            : optional torch.Generator for reproducible masking.

    Returns a dict with input_ids, attention_mask, labels (all LongTensor
    (B, T)) plus ``n_targets`` (int, total masked tokens in the batch).
    """

    tokenizer: object
    max_length: int = 2048
    select_prob: float = 1.0
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    generator: torch.Generator | None = None

    def __post_init__(self):
        self.mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if self.mask_token_id is None:
            raise ValueError(
                "tokenizer has no mask_token_id; variant MLM needs a [MASK] token."
            )
        self.vocab_size = len(self.tokenizer)

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator)

    def __call__(self, batch: dict) -> dict:
        sequences = batch["sequences"]
        fingerprints = batch["fingerprints"]

        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # (B, T) bool: tokens whose character span contains an alt position.
        variant_tok = _build_snp_mask(offset_mapping, fingerprints)
        variant_tok &= attention_mask.bool()  # never select padding/special tokens

        # Choose prediction targets among the variant tokens.
        if self.select_prob >= 1.0:
            targets = variant_tok
        else:
            targets = variant_tok & (self._rand(variant_tok.shape) < self.select_prob)

        labels = input_ids.clone()
        labels[~targets] = IGNORE_INDEX

        # 80/10/10-style corruption applied only to selected targets.
        r = self._rand(input_ids.shape)
        mask_tok = targets & (r < self.mask_replace_prob)
        rand_tok = targets & (r >= self.mask_replace_prob) & (
            r < self.mask_replace_prob + self.random_replace_prob
        )
        input_ids[mask_tok] = self.mask_token_id
        if rand_tok.any():
            rand_ids = torch.randint(
                0, self.vocab_size, (int(rand_tok.sum()),), generator=self.generator
            )
            input_ids[rand_tok] = rand_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_targets": int(targets.sum()),
        }

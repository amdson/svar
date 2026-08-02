"""
crop_embed/dataset.py
---------------------
UniqueWindowDataset: a PyTorch Dataset over the deduplicated set of genomic
windows implied by a VCF + reference FASTA + SNPWindowPartitioner.

Two (sample, window) pairs map to the same dataset item when they produce the
same DNA sequence — i.e., the same genomic coordinates AND the same set of
alt-allele positions within the window.

__len__      →  number of unique fingerprints (not samples × windows)
__getitem__  →  dict with the DNA sequence string and metadata for one unique window

Intended use: feed into a DataLoader to embed only unique windows, then
use SampleEmbedder to aggregate per-sample vectors from the embedding table.
"""

from __future__ import annotations

import bisect
from typing import Any

import torch
from pyfaidx import Fasta
from torch.utils.data import Dataset

from crop_embed.data.vcf import SNPRecord, load_snps_from_vcf
from crop_embed.fingerprint import (
    Fingerprint,
    build_sample_window_map,
)
from crop_embed.partitioner import SNPWindowPartitioner

class UniqueWindowDataset(Dataset):
    """
    Parameters
    ----------
    vcf_path    : path to a biallelic VCF (plain or bgzipped)
    fasta_path  : path to the reference FASTA (indexed with .fai)
    partitioner : pre-built SNPWindowPartitioner
    samples     : sample IDs to include; None = all samples in VCF

    Attributes
    ----------
    samples              : ordered list of sample IDs
    unique_fingerprints  : list of all unique Fingerprints (dataset index)
    fp_to_idx            : dict[Fingerprint → int]
    sample_window_to_fp  : dict[(sample_idx, window_idx) → Fingerprint]
    """

    def __init__(
        self,
        vcf_path: str,
        fasta_path: str,
        partitioner: SNPWindowPartitioner,
        samples: list[str] | None = None,
    ) -> None:
        self.fasta_path   = fasta_path
        self.partitioner  = partitioner

        # Load VCF (single pass)
        self._snps_by_chrom, self.samples = load_snps_from_vcf(vcf_path, samples)

        # FASTA reference cache + per-position alt-byte tables (cheap; VCF + reference
        # only). Shared with from_cached_windows via _build_reference_tables.
        self._build_reference_tables()

        # Build fingerprint index — the expensive dedup over every sample×window pair
        # (hundreds of millions on soy; see from_cached_windows to skip it on re-embed).
        self.sample_window_to_fp, self.unique_fingerprints, self.fp_to_idx = (
            build_sample_window_map(partitioner, len(self.samples))
        )

        # Tensor form of sample_window_to_fp, for fast batched gather during training.
        # Shape (n_samples, n_windows), values index into self.unique_fingerprints.
        self.sample_fp_index = torch.empty(
            (len(self.samples), len(partitioner)), dtype=torch.long
        )
        for (s, w), fp in self.sample_window_to_fp.items():
            self.sample_fp_index[s, w] = self.fp_to_idx[fp]

    # ── Reference tables + window reuse ───────────────────────────────────────

    def _build_reference_tables(self) -> None:
        """FASTA reference cache + per-position alt-byte lookup — everything
        extract_sequence() needs, derived only from the reference and the VCF SNP
        records (NOT from the sample×window enumeration). Requires self.fasta_path
        and self._snps_by_chrom to be set. Shared by __init__ and from_cached_windows."""
        _fasta = Fasta(self.fasta_path)
        self._chrom_name: dict[int, str] = {
            int(name): name for name in _fasta.keys() if name.isdigit()
        }
        # Cache reference chromosomes as bytearrays (lazy: filled on first access)
        self._ref_cache: dict[int, bytearray] = {}
        self._fasta_keys = list(_fasta.keys())
        # Warm the ref cache for every chromosome that carries SNPs (workers don't race).
        for chrom in self._snps_by_chrom:
            self._ref_seq(chrom)
        # Per-position alt_byte lookup: {chrom: {pos: alt_byte}}
        self._alt_byte: dict[int, dict[int, int]] = {
            chrom: {rec.pos: rec.alt_byte for rec in snps}
            for chrom, snps in self._snps_by_chrom.items()
        }

    @classmethod
    def from_cached_windows(
        cls, cache_path: str, fasta_path: str, *, vcf_path: str | None = None
    ) -> "UniqueWindowDataset":
        """Reconstruct a dataset for RE-EMBEDDING ONLY, reusing the window structure
        already computed in an existing embedding cache.

        The unique-window set and the sample→window index are backbone-independent,
        so a second backbone (e.g. Carbon-3B after Carbon-500M) loads them straight
        from the prior cache and skips ``build_sample_window_map`` — the pathological
        pass that materialises one fingerprint per sample×window pair (hundreds of
        millions; ~130 GB / >1 h on soy). Only the VCF (per-position alt bytes) and
        FASTA (reference) are still needed, both cheap.

        The returned object supports __len__/__getitem__/extract_sequence and exposes
        samples / unique_fingerprints / sample_fp_index — everything
        fill_embedding_table + FixedWindowEmbedder.from_embedding_table read. The
        training-only lookups (sample_window_to_fp, fp_to_idx) are left None.
        """
        prev = torch.load(cache_path, map_location="cpu", weights_only=False)
        missing = [k for k in ("unique_fingerprints", "sample_fp_index", "sample_ids")
                   if k not in prev]
        if missing:
            raise ValueError(
                f"cache {cache_path} lacks {missing}; it predates window reuse "
                f"and cannot seed a re-embed.")
        vcf_path = vcf_path or prev.get("metadata", {}).get("vcf_path")
        if not vcf_path:
            raise ValueError(
                "no --vcf-path given and none recorded in the source cache metadata; "
                "extract_sequence still needs the VCF for per-position alt bytes.")

        obj = cls.__new__(cls)
        obj.fasta_path = fasta_path
        obj.partitioner = None
        obj.samples = list(prev["sample_ids"])
        obj.unique_fingerprints = prev["unique_fingerprints"]
        obj.sample_fp_index = prev["sample_fp_index"]
        obj.sample_window_to_fp = None      # training-only; not needed to embed
        obj.fp_to_idx = None
        obj.source_metadata = prev.get("metadata", {})
        # VCF only for alt bytes (position-keyed, sample-order-independent); load all
        # SNPs so every fingerprint's alt positions resolve.
        obj._snps_by_chrom, _ = load_snps_from_vcf(vcf_path, None)
        obj._build_reference_tables()
        return obj

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.unique_fingerprints)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Returns
        -------
        {
            "sequence":    str        — DNA string of length 2*half_window
            "fingerprint": tuple      — the cache key
            "chrom":       int
            "start":       int        — 0-based, may be negative near chr start
            "end":         int        — 0-based exclusive
            "idx":         int        — dataset index
        }
        """
        fp    = self.unique_fingerprints[idx]
        seq   = self.extract_sequence(fp)
        chrom, start, end, _ = fp
        return {
            "sequence":    seq,
            "fingerprint": fp,
            "chrom":       chrom,
            "start":       start,
            "end":         end,
            "idx":         idx,
        }

    # ── Direct lookup by (sample, window) ────────────────────────────────────

    def get_fingerprint(self, sample_id: str, window_idx: int) -> Fingerprint:
        """Return the fingerprint for a specific (sample, window) pair."""
        sample_idx = self.samples.index(sample_id)
        return self.sample_window_to_fp[(sample_idx, window_idx)]

    def get_item_for_sample_window(
        self, sample_id: str, window_idx: int
    ) -> dict[str, Any]:
        """Same return shape as __getitem__, addressed by sample + window."""
        fp  = self.get_fingerprint(sample_id, window_idx)
        idx = self.fp_to_idx[fp]
        return self.__getitem__(idx)

    # ── Batched lookup for training ───────────────────────────────────────────

    def gather_batch(
        self, sample_indices: torch.Tensor
    ) -> tuple[list[str], list[Fingerprint], torch.Tensor]:
        """
        Gather the deduplicated set of windows for a batch of samples.

        Parameters
        ----------
        sample_indices : LongTensor[B] of row indices into self.samples

        Returns
        -------
        sequences    : list[str]            — DNA strings for each unique fingerprint
        fingerprints : list[Fingerprint]    — same order as `sequences`
        inverse      : LongTensor[B, n_windows]  — index into `sequences`,
                                                   suitable for `emb[inverse]` scatter
        """
        batch_idx = self.sample_fp_index[sample_indices]               # (B, n_windows)
        unique_global, inverse = torch.unique(batch_idx, return_inverse=True)
        fingerprints = [self.unique_fingerprints[i] for i in unique_global.tolist()]
        sequences    = [self.extract_sequence(fp) for fp in fingerprints]
        return sequences, fingerprints, inverse

    # ── Sequence extraction ───────────────────────────────────────────────────

    def extract_sequence(self, fp: Fingerprint) -> str:
        """
        Materialise the DNA string for a fingerprint.
        Applies alt alleles from fp[3] onto the reference window.
        """
        chrom, w_start, w_end, alt_positions = fp
        ref   = self._ref_seq(chrom)

        # Clip to chromosome boundaries
        clip_start = max(0, w_start)
        clip_end   = min(len(ref), w_end)

        window = bytearray(ref[clip_start:clip_end])

        alt_map = self._alt_byte.get(chrom, {})
        for pos in alt_positions:
            offset = pos - clip_start
            if 0 <= offset < len(window):
                window[offset] = alt_map[pos]
                
        seq = window.decode("ascii")

        # Pad with 'N' if the window extends past chromosome boundaries
        # TODO dubeous, check if should use N padding or attention mask padding
        left_pad  = max(0, -w_start)
        right_pad = max(0, w_end - len(ref))
        if left_pad or right_pad:
            seq = "N" * left_pad + seq + "N" * right_pad

        return seq

    def _ref_seq(self, chrom: int) -> bytearray:
        """Lazy-load and cache a chromosome as an uppercase bytearray."""
        if chrom not in self._ref_cache:
            key = self._chrom_name[chrom]
            fasta = Fasta(self.fasta_path)
            self._ref_cache[chrom] = bytearray(
                str(fasta[key]).upper().encode("ascii")
            )
        return self._ref_cache[chrom]


class SampleDataset(Dataset):
    """
    Sample-indexed view of a UniqueWindowDataset for use with DataLoader.

    __getitem__ returns (global_sample_idx, target_tensor) for position
    `local_idx` in the (possibly subsetted) sample list.  The global index
    is used by the training loop to look up window embeddings via
    `window_dataset.sample_fp_index`.

    Parameters
    ----------
    window_dataset  : underlying UniqueWindowDataset
    targets         : Tensor[n_total_samples, n_traits] aligned to
                      window_dataset.samples order; NaN = missing
    sample_indices  : LongTensor of row indices into window_dataset.samples
                      to include; None means all samples
    """

    def __init__(
        self,
        window_dataset: UniqueWindowDataset,
        targets: torch.Tensor,
        sample_indices: torch.Tensor | None = None,
    ) -> None:
        self.window_dataset = window_dataset
        self.targets = targets
        self.sample_indices = (
            sample_indices
            if sample_indices is not None
            else torch.arange(len(window_dataset.samples))
        )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, local_idx: int):
        global_idx = int(self.sample_indices[local_idx])
        return global_idx, self.targets[global_idx]

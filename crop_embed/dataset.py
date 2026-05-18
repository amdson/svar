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

        # Build chrom_name map from FASTA key names (numeric keys only)
        _fasta = Fasta(fasta_path)
        self._chrom_name: dict[int, str] = {
            int(name): name for name in _fasta.keys() if name.isdigit()
        }
        # Cache reference chromosomes as bytearrays (lazy: filled on first access)
        self._ref_cache: dict[int, bytearray] = {}
        self._fasta_keys = list(_fasta.keys())
        # Pre-load all chromosomes that appear in the partitioner
        for chrom in partitioner.snps_by_chrom:
            self._ref_seq(chrom)  # warm the cache now so workers don't race

        # Pre-compute per-position alt_byte lookup: {chrom: {pos: alt_byte}}
        self._alt_byte: dict[int, dict[int, int]] = {
            chrom: {rec.pos: rec.alt_byte for rec in snps}
            for chrom, snps in self._snps_by_chrom.items()
        }

        # Build fingerprint index
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

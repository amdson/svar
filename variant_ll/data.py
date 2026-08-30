"""
variant_ll/data.py
------------------
Window-centric data layer for the allele log-likelihood benchmark (see
BENCHMARK.md). One item = one genomic window, carrying:

  * the all-reference token ids for that window (the frozen cache context),
  * every SNP site in it, as *token* positions,
  * the deduplicated haplotypes of a set of accessions, with multiplicities,
  * the two candidate 6-mer token ids per site (ref allele vs alt allele).

The unit of work is the window, not the (sample, window) pair: all accessions
share the same *sites* in a window and differ only in *alleles*, so one frozen
reference forward serves every haplotype and they batch along the variant cache's
N dimension in a single call.

Two facts about Carbon's tokenizer this depends on (verified in
`_verify_tokenization`): in ``<dna>`` mode it splits the sequence into fixed,
non-overlapping 6-mers after a single marker token, and the 4096 DNA k-mer ids
form a contiguous base-4 block with alphabet order ``ATCG``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pyfaidx import Fasta

from crop_embed.partitioner import SNPWindowPartitioner

KMER = 6
DNA_MARKER_ID = 151669   # "<dna>"
OOV_ID = 151671          # any k-mer containing N
KMER_BASE = 151672       # first of the 4096 contiguous DNA 6-mer ids
KMER_ORDER = "ATCG"      # base-4 digit order within that block


def kmer_id(kmer: str) -> int:
    """Token id of a 6-mer. Non-ACGT (i.e. N) anywhere -> <oov>."""
    v = 0
    for ch in kmer:
        i = KMER_ORDER.find(ch)
        if i < 0:
            return OOV_ID
        v = v * 4 + i
    return KMER_BASE + v


def _verify_tokenization(tokenizer, seq: str) -> None:
    """Assert the analytic k-mer split + id formula match the real tokenizer.

    Cheap insurance: everything downstream builds token ids arithmetically rather
    than calling the tokenizer per window, so a checkpoint or tokenizer change
    that broke either assumption would otherwise corrupt the benchmark silently.
    """
    got = tokenizer("<dna>" + seq, add_special_tokens=False)["input_ids"]
    want = tokenize_window(seq)
    if got != want:
        raise RuntimeError(
            f"analytic tokenization disagrees with the tokenizer: "
            f"{got[:8]} != {want[:8]} (len {len(got)} vs {len(want)})")
    for k in ("AAAAAA", "ACGTAC", "TTTTTT", "CCCCCC"):
        ref = tokenizer("<dna>" + k, add_special_tokens=False)["input_ids"][-1]
        if ref != kmer_id(k):
            raise RuntimeError(f"kmer_id({k}) = {kmer_id(k)} != {ref}")


def tokenize_window(seq: str) -> list[int]:
    """`<dna>` marker + fixed-stride 6-mers. A trailing partial chunk is
    right-padded with 'A' and still occupies one token (matches the tokenizer)."""
    ids = [DNA_MARKER_ID]
    for i in range(0, len(seq), KMER):
        chunk = seq[i:i + KMER]
        if len(chunk) < KMER:
            chunk = chunk + "A" * (KMER - len(chunk))
        ids.append(kmer_id(chunk))
    return ids


@dataclass
class WindowBatch:
    """One window's frozen reference plus every haplotype observed in a split.

    The scored unit is a **token**, not a site. Carbon's DNA mode is a fixed
    contiguous 6-mer split, so a token holding m segregating sites has 2**m
    possible values; the autoregressive target is simply the one carrying this
    haplotype's alleles at all m, and the score is renormalised over those 2**m
    candidates. For m = 1 that is the two-candidate ref/alt case; for m > 1 it is
    the same formula, and — unlike scoring each site with its co-token neighbours
    pinned to their true values — it stays a chain-rule factorisation of
    P(haplotype | reference) and conditions only on strictly upstream tokens.

    Shapes: T = tokens in the window, K = scored tokens, S = sites (>= K),
    C = cached positions, N = unique haplotypes, Q = 2**max(m) candidate slots.
    """
    chrom: int
    start: int
    ref_ids: torch.Tensor      # (T,)     all-reference token ids
    tok_idx: torch.Tensor      # (K,)     token index of each scored token
    tok_nsite: torch.Tensor    # (K,)     m — sites in that token; the bits/SNP weight
    cache_idx: torch.Tensor    # (C,)     unique(tok_idx ∪ tok_idx-1), sorted
    pred_row: torch.Tensor     # (K,)     row of (tok_idx-1) within cache_idx
    hap_ids: torch.Tensor      # (N, C)   token ids at cached positions
    hap_target: torch.Tensor   # (N, K)   index of the true candidate, bit b = site b
    hap_count: torch.Tensor    # (N,)     multiplicity within this split
    cand_ids: torch.Tensor     # (K, Q)   candidate 6-mer ids, right-padded
    cand_mask: torch.Tensor    # (K, Q)   True where a candidate slot is real
    has_upstream: torch.Tensor  # (K,)    ≥1 scored token strictly earlier
    site_tok: torch.Tensor     # (S,)     token index per site (non-decreasing)
    hap_allele: torch.Tensor   # (N, S)   True = carries alt (baselines only)
    site_pos: np.ndarray       # (S,)     genomic positions (for baseline alignment)

    @property
    def n_hap(self) -> int:
        return self.hap_ids.shape[0]

    @property
    def n_site(self) -> int:
        return int(self.tok_nsite.sum())

    def to(self, device) -> "WindowBatch":
        move = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t
        return WindowBatch(**{k: move(v) for k, v in self.__dict__.items()})


def load_vcf(vcf_path: str, chroms: list[int] | None = None,
             engine: str = "auto") -> tuple[dict, list[str]]:
    """Read a VCF into ``({chrom: [SNPRecord]}, samples)``, chromosome-filtered.

    ``engine="polars"`` uses `crop_embed.data.vcf_polars`, which parses the whole
    file in one vectorized Rust pass; on the arabidopsis 1001G export (10 GB, 2.3M
    SNPs x 1,135 accessions) that is the difference between a minute and most of an
    hour, so it is what any large-VCF read here should use. It only handles plain
    text, so ``"auto"`` (the default) takes it for a ``.vcf`` and falls back to
    pysam for a ``.vcf.gz`` / ``.bcf``. Both push ``chroms`` down to the record
    loop, which for a single chromosome is most of the remaining cost.
    """
    if engine == "auto":
        engine = "pysam" if vcf_path.endswith((".gz", ".bgz", ".bcf")) else "polars"
    keep = set(chroms) if chroms is not None else None
    if engine == "polars":
        from crop_embed.data.vcf_polars import load_snps_from_vcf as _load
    elif engine == "pysam":
        from crop_embed.data.vcf import load_snps_from_vcf as _load
    else:
        raise ValueError(f"unknown VCF engine {engine!r} (want polars/pysam/auto)")
    snps_by_chrom, samples = _load(vcf_path, None, keep)
    if keep is not None:                       # defensive: engine honours it already
        snps_by_chrom = {c: v for c, v in snps_by_chrom.items() if c in keep}
    return snps_by_chrom, samples


class _RefGenome:
    """Chromosome bytearrays with the same window semantics as
    `UniqueWindowDataset.extract_sequence`: clip to the chromosome, then pad the
    overhang with 'N'."""

    def __init__(self, fasta_path: str):
        self.path = fasta_path
        fasta = Fasta(fasta_path)
        self._names = {int(n): n for n in fasta.keys() if n.isdigit()}
        self._cache: dict[int, bytes] = {}

    def chrom(self, chrom: int) -> bytes:
        if chrom not in self._cache:
            fasta = Fasta(self.path)
            self._cache[chrom] = str(fasta[self._names[chrom]]).upper().encode()
        return self._cache[chrom]

    def window(self, chrom: int, start: int, end: int) -> str:
        ref = self.chrom(chrom)
        lo, hi = max(0, start), min(len(ref), end)
        seq = ref[lo:hi].decode("ascii")
        return "N" * max(0, -start) + seq + "N" * max(0, end - len(ref))


class WindowSource:
    """Builds `WindowBatch`es from a VCF + reference FASTA.

    ``chroms`` restricts to a subset of chromosomes (the plan's "one chromosome"
    training set); ``sample_rows`` in :meth:`build` restricts the accessions whose
    haplotypes are collected, which is how the train/val split is applied.
    ``engine`` selects the VCF reader — see :func:`load_vcf`; the default picks
    the fast polars path for any plain-text VCF.
    """

    def __init__(self, vcf_path: str, fasta_path: str, half_window: int,
                 buffer: int = 0, chroms: list[int] | None = None,
                 engine: str = "auto"):
        snps_by_chrom, self.samples = load_vcf(vcf_path, chroms, engine)
        self.snps_by_chrom = snps_by_chrom
        self.partitioner = SNPWindowPartitioner(snps_by_chrom, half_window, buffer)
        self.ref = _RefGenome(fasta_path)
        self.stats = {"dropped_n_token": 0, "dropped_ref_mismatch": 0,
                      "dropped_past_window": 0, "sites": 0, "multi_site_tokens": 0}
        # Site accounting is per *window*, not per call — the same window is built
        # once per split (train/val) and again for the baselines, and those repeats
        # must not double-count.
        self._counted: set[int] = set()

    def __len__(self) -> int:
        return len(self.partitioner.windows)

    def build(self, win_idx: int, sample_rows: np.ndarray,
              shuffle_sites: bool = False, rng: np.random.Generator | None = None
              ) -> WindowBatch | None:
        """One window's batch over ``sample_rows`` (indices into ``self.samples``).

        Returns None if the window has no scoreable token. A token is dropped only
        when its 6-mer contains N (every candidate would collapse to <oov>) or when
        the VCF REF disagrees with the genome; a token holding several segregating
        sites is *kept* and scored over its 2**m possible values, which at
        arabidopsis density is 21% of sites that would otherwise be discarded.

        ``shuffle_sites`` is the LD-destroying control: permute each site's
        alleles across accessions *independently*, before deduplication. Every
        site's marginal frequency is preserved exactly, but the correlation
        between a site and its upstream neighbours is destroyed. A model that has
        only learned per-site frequencies scores the same; one that is reading the
        individual's haplotype gets worse. (Permuting whole haplotypes would not
        work — it moves context and target together, so the pairing survives.)
        """
        win = self.partitioner.windows[win_idx]
        rows = self.partitioner.window_snp_indices[win_idx]
        if not rows:
            return None
        snps = [self.snps_by_chrom[win.chrom][i] for i in rows]

        seq = self.ref.window(win.chrom, win.start, win.end)
        ref_ids = tokenize_window(seq)
        n_tok = len(ref_ids)

        # Group sites by the token whose 6-mer contains them. `snps` is
        # position-sorted, so each token's member list is too — which is what makes
        # bit b of a candidate index mean "the b-th site in this token".
        by_token: dict[int, list] = {}
        for snp in snps:
            off = snp.pos - win.start
            if not (0 <= off < len(seq)):
                continue
            by_token.setdefault(1 + off // KMER, []).append((snp, off))

        count = win_idx not in self._counted
        self._counted.add(win_idx)
        keep = []   # (token, [(snp, offset-within-6mer), ...], reference 6-mer)
        for tok, members in sorted(by_token.items()):
            if count:
                self.stats["sites"] += len(members)
            if tok >= n_tok:
                if count:
                    self.stats["dropped_past_window"] += len(members)
                continue
            lo = (tok - 1) * KMER
            mer = seq[lo:lo + KMER].ljust(KMER, "A")
            if "N" in mer:
                if count:
                    self.stats["dropped_n_token"] += len(members)
                continue
            sites = []
            for snp, off in members:
                within = off - lo
                if mer[within] != chr(snp.ref_byte):   # REF vs genome disagreement
                    if count:
                        self.stats["dropped_ref_mismatch"] += 1
                    continue
                sites.append((snp, within))
            if not sites:
                continue
            if count and len(sites) > 1:
                self.stats["multi_site_tokens"] += 1
            keep.append((tok, sites, mer))
        if not keep:
            return None

        tok_idx = torch.tensor([k[0] for k in keep], dtype=torch.long)
        tok_nsite = torch.tensor([len(k[1]) for k in keep], dtype=torch.long)
        site_tok = torch.tensor([t for t, sites, _ in keep for _ in sites],
                                dtype=torch.long)
        site_pos = np.array([snp.pos for _, sites, _ in keep for snp, _ in sites],
                            dtype=np.int64)

        # Candidates: every assignment of ref/alt to the token's m segregating
        # sites, so the target is just "the token with this haplotype's alleles
        # filled in". Bit b of the candidate index is the b-th site in the token.
        # Ragged over tokens (m varies), so right-pad and carry a mask.
        n_cand = 1 << int(tok_nsite.max())
        cand = torch.zeros((len(keep), n_cand), dtype=torch.long)
        cand_mask = torch.zeros((len(keep), n_cand), dtype=torch.bool)
        for k, (_, sites, mer) in enumerate(keep):
            for combo in range(1 << len(sites)):
                chars = list(mer)
                for b, (snp, within) in enumerate(sites):
                    chars[within] = chr(snp.alt_byte if (combo >> b) & 1
                                        else snp.ref_byte)
                cand[k, combo] = kmer_id("".join(chars))
                cand_mask[k, combo] = True

        # Cached positions: the scored tokens, plus the position that predicts each
        # of them. unique() also sorts and dedups, which forward() requires.
        pred_tok = (tok_idx - 1).clamp(min=0)
        cache_idx = torch.unique(torch.cat([tok_idx, pred_tok]))
        row_of = {int(p): i for i, p in enumerate(cache_idx)}
        pred_row = torch.tensor([row_of[int(p)] for p in pred_tok], dtype=torch.long)

        # Haplotypes of the requested accessions, deduplicated with counts.
        alt_flags = np.stack([np.frombuffer(snp.gt_alts, dtype=np.uint8)[sample_rows]
                              for _, sites, _ in keep for snp, _ in sites],
                             axis=1)                              # (n_samples, S)
        if shuffle_sites:
            rng = rng or np.random.default_rng(0)
            alt_flags = np.stack([rng.permutation(col) for col in alt_flags.T], axis=1)
        uniq, counts = np.unique(alt_flags, axis=0, return_counts=True)
        hap_allele = torch.from_numpy(uniq.astype(bool))          # (N, S)
        hap_count = torch.tensor(counts, dtype=torch.float32)

        # Per-token target index: pack each token's sites into the candidate index.
        bit = torch.zeros(len(site_tok), dtype=torch.long)
        tok_of_site = torch.zeros(len(site_tok), dtype=torch.long)
        c = 0
        for k, (_, sites, _) in enumerate(keep):
            for b in range(len(sites)):
                bit[c], tok_of_site[c] = b, k
                c += 1
        hap_target = torch.zeros((len(uniq), len(keep)), dtype=torch.long)
        hap_target.index_add_(1, tok_of_site,
                              hap_allele.long() << bit.unsqueeze(0))   # (N, K)

        # Token ids at cached positions: reference everywhere, then patch the
        # scored tokens with each haplotype's candidate.
        base = torch.tensor(ref_ids, dtype=torch.long)
        hap_ids = base.index_select(0, cache_idx).unsqueeze(0).repeat(len(uniq), 1)
        tok_row = torch.tensor([row_of[int(t)] for t in tok_idx], dtype=torch.long)
        hap_ids[:, tok_row] = torch.gather(
            cand.unsqueeze(0).expand(len(uniq), *cand.shape), 2,
            hap_target.unsqueeze(-1)).squeeze(-1)

        # A token has upstream context iff another scored token sits strictly
        # earlier. Sites sharing a token are predicted jointly, not sequentially,
        # so they give each other no context — which is exactly why this is a
        # per-token property.
        has_up = torch.arange(len(keep)) > 0

        return WindowBatch(
            chrom=win.chrom, start=win.start, ref_ids=base, tok_idx=tok_idx,
            tok_nsite=tok_nsite, cache_idx=cache_idx, pred_row=pred_row,
            hap_ids=hap_ids, hap_target=hap_target, hap_count=hap_count,
            cand_ids=cand, cand_mask=cand_mask, has_upstream=has_up,
            site_tok=site_tok, hap_allele=hap_allele, site_pos=site_pos)

    def verify(self, tokenizer, win_idx: int = 0) -> None:
        win = self.partitioner.windows[win_idx]
        _verify_tokenization(tokenizer, self.ref.window(win.chrom, win.start, win.end))

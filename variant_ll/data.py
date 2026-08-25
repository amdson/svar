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

import numpy as np
import torch
from pyfaidx import Fasta

from crop_embed.data.vcf import load_snps_from_vcf
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

    Shapes: T = tokens in the window, S = scoreable sites, C = cached positions,
    N = unique haplotypes.
    """
    chrom: int
    start: int
    ref_ids: torch.Tensor      # (T,)     all-reference token ids
    site_tok: torch.Tensor     # (S,)     token index of each site
    cache_idx: torch.Tensor    # (C,)     unique(site_tok ∪ site_tok-1), sorted
    pred_row: torch.Tensor     # (S,)     row of (site_tok-1) within cache_idx
    hap_ids: torch.Tensor      # (N, C)   token ids at cached positions
    hap_allele: torch.Tensor   # (N, S)   True = carries alt
    hap_count: torch.Tensor    # (N,)     multiplicity within this split
    cand_ids: torch.Tensor     # (S, 2)   [ref-allele 6-mer, alt-allele 6-mer]
    has_upstream: torch.Tensor  # (S,)    ≥1 site at a strictly earlier token
    site_pos: np.ndarray       # (S,)     genomic positions (for baseline alignment)

    @property
    def n_hap(self) -> int:
        return self.hap_ids.shape[0]

    def to(self, device) -> "WindowBatch":
        move = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t
        return WindowBatch(**{k: move(v) for k, v in self.__dict__.items()})


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
    training set); ``samples`` restricts the accessions whose haplotypes are
    collected, which is how the train/val split is applied.
    """

    def __init__(self, vcf_path: str, fasta_path: str, half_window: int,
                 buffer: int = 0, chroms: list[int] | None = None):
        snps_by_chrom, self.samples = load_snps_from_vcf(vcf_path, None)
        if chroms is not None:
            snps_by_chrom = {c: snps_by_chrom[c] for c in chroms if c in snps_by_chrom}
        self.snps_by_chrom = snps_by_chrom
        self.partitioner = SNPWindowPartitioner(snps_by_chrom, half_window, buffer)
        self.ref = _RefGenome(fasta_path)
        self.stats = {"dropped_multisite": 0, "dropped_n_token": 0,
                      "dropped_ref_mismatch": 0, "sites": 0}

    def __len__(self) -> int:
        return len(self.partitioner.windows)

    def build(self, win_idx: int, sample_rows: np.ndarray,
              shuffle_sites: bool = False, rng: np.random.Generator | None = None
              ) -> WindowBatch | None:
        """One window's batch over ``sample_rows`` (indices into ``self.samples``).

        Returns None if the window has no scoreable site. Sites are dropped when
        their 6-mer contains N (both candidates would collapse to <oov>), when the
        VCF REF disagrees with the genome, or when two sites share a 6-mer — the
        last is a ~0.05% case in rice but needs proper marginalisation at
        arabidopsis density (see PLAN.md Phase 1).

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

        # Group sites by the token whose 6-mer contains them.
        by_token: dict[int, list] = {}
        for snp in snps:
            off = snp.pos - win.start
            if not (0 <= off < len(seq)):
                continue
            by_token.setdefault(1 + off // KMER, []).append((snp, off))

        keep = []   # (token, snp, offset-within-6mer)
        for tok, members in sorted(by_token.items()):
            self.stats["sites"] += len(members)
            if tok >= n_tok:
                continue
            if len(members) > 1:                      # multi-site token
                self.stats["dropped_multisite"] += len(members)
                continue
            snp, off = members[0]
            lo = (tok - 1) * KMER
            mer = seq[lo:lo + KMER].ljust(KMER, "A")
            if "N" in mer:
                self.stats["dropped_n_token"] += 1
                continue
            within = off - lo
            if mer[within] != chr(snp.ref_byte):       # REF vs genome disagreement
                self.stats["dropped_ref_mismatch"] += 1
                continue
            keep.append((tok, snp, within, mer))
        if not keep:
            return None

        site_tok = torch.tensor([k[0] for k in keep], dtype=torch.long)
        site_pos = np.array([k[1].pos for k in keep], dtype=np.int64)
        cand = torch.tensor(
            [[kmer_id(m[:w] + chr(s.ref_byte) + m[w + 1:]),
              kmer_id(m[:w] + chr(s.alt_byte) + m[w + 1:])]
             for _, s, w, m in keep], dtype=torch.long)          # (S, 2)

        # Cached positions: the site tokens, plus the position that predicts each
        # of them. unique() also sorts and dedups, which forward() requires.
        pred_tok = (site_tok - 1).clamp(min=0)
        cache_idx = torch.unique(torch.cat([site_tok, pred_tok]))
        row_of = {int(p): i for i, p in enumerate(cache_idx)}
        pred_row = torch.tensor([row_of[int(p)] for p in pred_tok], dtype=torch.long)

        # Haplotypes of the requested accessions, deduplicated with counts.
        alt_flags = np.stack([np.frombuffer(s.gt_alts, dtype=np.uint8)[sample_rows]
                              for _, s, _, _ in keep], axis=1)    # (n_samples, S)
        if shuffle_sites:
            rng = rng or np.random.default_rng(0)
            alt_flags = np.stack([rng.permutation(col) for col in alt_flags.T], axis=1)
        uniq, counts = np.unique(alt_flags, axis=0, return_counts=True)
        hap_allele = torch.from_numpy(uniq.astype(bool))          # (N, S)
        hap_count = torch.tensor(counts, dtype=torch.float32)

        # Token ids at cached positions: reference everywhere, then patch the site
        # tokens with each haplotype's allele.
        base = torch.tensor(ref_ids, dtype=torch.long)
        hap_ids = base.index_select(0, cache_idx).unsqueeze(0).repeat(len(uniq), 1)
        site_row = torch.tensor([row_of[int(t)] for t in site_tok], dtype=torch.long)
        hap_ids[:, site_row] = torch.gather(
            cand.T.unsqueeze(0).expand(len(uniq), 2, len(keep)), 1,
            hap_allele.long().unsqueeze(1)).squeeze(1)

        # A site has upstream context iff another site sits at a strictly earlier
        # token — same-token sites are not visible to the predictor at site_tok-1.
        has_up = torch.tensor([bool((site_tok < t).any()) for t in site_tok])

        return WindowBatch(
            chrom=win.chrom, start=win.start, ref_ids=base, site_tok=site_tok,
            cache_idx=cache_idx, pred_row=pred_row, hap_ids=hap_ids,
            hap_allele=hap_allele, hap_count=hap_count, cand_ids=cand,
            has_upstream=has_up, site_pos=site_pos)

    def verify(self, tokenizer, win_idx: int = 0) -> None:
        win = self.partitioner.windows[win_idx]
        _verify_tokenization(tokenizer, self.ref.window(win.chrom, win.start, win.end))


def genotype_split(samples: list[str], seed: int = 42,
                   val: float = 0.15, test: float = 0.15
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 70/15/15 split over *genotyped* accessions.

    `training.common.splits` partitions phenotyped samples; this benchmark uses
    no traits, so it splits the VCF's sample list directly (PLAN.md Phase 2).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(samples))
    n_test = int(round(len(samples) * test))
    n_val = int(round(len(samples) * val))
    return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]

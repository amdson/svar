"""
training_ge/ath_data.py — arabidopsis 1001G per-gene batches for the variant
cache. Mirrors SieveWindowSource so training_ge/run.py drives either dataset.

Differences from SIEVE that matter:
  * variants are shared, not private: an accession differs from TAIR10 at many
    window positions, so `own_mask` marks every position where its tokens
    differ from reference, and cs per gene is ~100+ at hw=4000 — run with
    `encoder.variant_checkpointing = True`.
  * targets are per-gene z of `deviation`, standardized by TRAIN-accession
    mean/sd (no control lines; no background-gene offset exists here).
  * the holdout axis of interest is accessions (`accessions/acc_split`);
    relatedness is a real confound — read every result against
    training_ge/ath_kinship_baseline.py (GBLUP pooled val pearson ~0.17).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from training_ge.data import GeneBatch, id6, verify_id6

DATA = "/90daydata/small_grains/andrew.dickson/datasets/arabidopsis"
H5 = f"{DATA}/expression/expression_dataset.h5"
PFILE = f"{DATA}/arabidopsis_1001g_final"
FASTA = f"{DATA}/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa"


def load_pvar(path):
    chroms, poss, refs, alts = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 5)
            chroms.append(f[0]); poss.append(int(f[1]))
            refs.append(f[3]); alts.append(f[4])
    return (np.array(chroms), np.array(poss, dtype=np.int64),
            np.array(refs), np.array(alts))


class ArabidopsisWindowSource:
    def __init__(self, tokenizer, half_window: int = 4000,
                 max_lines: int = 700, seed: int = 42, max_ac: int = 0):
        import h5py
        import pgenlib
        import pysam

        self.hw = half_window
        self.max_lines = max_lines
        self.body_len = 6 * ((2 * half_window) // 6)
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        verify_id6(tokenizer, self.rng)

        with h5py.File(H5, "r") as f:
            dev = f["deviation"][:]
            self.gene_id = f["genes/gene_id"][:].astype(str)
            self.chrom = f["genes/chrom"][:].astype(str)
            self.tss = f["genes/tss"][:]
            self.fam_split = f["genes/family_split"][:].astype(str)
            self.eco = f["accessions/ecotype_id"][:].astype(str)
            self.acc_split = f["accessions/acc_split"][:].astype(str)

        tr = self.acc_split == "train"
        mu = dev[:, tr].mean(axis=1, keepdims=True)
        sd = dev[:, tr].std(axis=1, ddof=1, keepdims=True)
        self.scoreable = sd[:, 0] > 1e-3
        self.z_all = (dev - mu) / np.where(sd > 1e-3, sd, 1.0)  # (genes, 665)

        psam = [l.split()[0] for l in open(PFILE + ".psam")
                if not l.startswith("#")]
        col = {s: i for i, s in enumerate(psam)}
        self.panel_cols = np.array([col[e] for e in self.eco])
        self.n_psam = len(psam)

        self.v_chrom, self.v_pos, self.v_ref, self.v_alt = load_pvar(
            PFILE + ".pvar")
        self.by_chrom = {c: np.flatnonzero(self.v_chrom == c)
                         for c in np.unique(self.v_chrom)}
        self.reader = pgenlib.PgenReader(str(PFILE + ".pgen").encode())

        self.fa = pysam.FastaFile(FASTA)
        self.fa_chroms = set(self.fa.references)
        self.skip_counts = {"no_window": 0, "n_in_window": 0, "no_lines": 0,
                            "ref_mismatch_sites": 0, "edge": 0, "indel_sites": 0}

    def sample_genes(self, n_genes: int, split: Optional[str] = None) -> np.ndarray:
        ok = self.scoreable & np.isin(self.chrom, list(self.fa_chroms))
        if split is not None:
            ok &= self.fam_split == split
        return self.rng.permutation(np.flatnonzero(ok))[:n_genes]

    def build(self, gi: int) -> Optional[GeneBatch]:
        c = self.chrom[gi]
        w0 = int(self.tss[gi]) - self.hw          # 1-based inclusive
        if w0 < 1:
            self.skip_counts["edge"] += 1
            return None
        seq = self.fa.fetch(c, w0 - 1, w0 - 1 + self.body_len).upper()
        if len(seq) < self.body_len:
            self.skip_counts["edge"] += 1
            return None
        if "N" in seq:
            self.skip_counts["n_in_window"] += 1
            return None

        ix = self.by_chrom.get(c)
        pos_c = self.v_pos[ix]
        a, b = np.searchsorted(pos_c, [w0, w0 + self.body_len])
        if b <= a:
            self.skip_counts["no_lines"] += 1
            return None
        vsel = ix[a:b]
        # biallelic SNPs only; guard against indels and ref mismatches
        keep = np.ones(len(vsel), dtype=bool)
        for k, v in enumerate(vsel):
            r, alt = self.v_ref[v], self.v_alt[v]
            if len(r) != 1 or len(alt) != 1:
                keep[k] = False
                self.skip_counts["indel_sites"] += 1
            elif seq[int(self.v_pos[v]) - w0] != r:
                keep[k] = False
                self.skip_counts["ref_mismatch_sites"] += 1
        vsel = vsel[keep]
        if len(vsel) == 0:
            self.skip_counts["no_lines"] += 1
            return None

        # read_range is contiguous; select the kept variants within the range
        lo, hi = int(vsel[0]), int(vsel[-1]) + 1
        geno = np.empty((hi - lo, self.n_psam), dtype=np.int8)
        self.reader.read_range(lo, hi, geno)
        geno = geno[vsel - lo][:, self.panel_cols]            # (S, 665)
        alt_carrier = geno > 0                                # missing -> ref

        offs = self.v_pos[vsel] - w0                          # 0-based in window
        toks = offs // 6
        alt_b = self.v_alt[vsel]

        enc = self.tokenizer("<dna>" + seq, add_special_tokens=False)["input_ids"]
        ref_ids = torch.tensor(enc, dtype=torch.long)
        assert ref_ids.shape[0] == 1 + self.body_len // 6

        cache = sorted(set(int(t) + 1 for t in toks))         # +1: <dna> marker
        cpos = {t: k for k, t in enumerate(cache)}
        accs = np.arange(len(self.eco))
        if len(accs) > self.max_lines:
            accs = self.rng.choice(accs, self.max_lines, replace=False)
        N, C = len(accs), len(cache)
        hap = ref_ids[cache].unsqueeze(0).repeat(N + 1, 1)
        own = torch.zeros(N, C, dtype=torch.bool)
        # per accession: substitute alt base at every carried site, per token
        tok_groups = {}
        for s, (t, o) in enumerate(zip(toks, offs)):
            tok_groups.setdefault(int(t), []).append((int(o % 6), s))
        for i, ai in enumerate(accs):
            for t, sites in tok_groups.items():
                carried = [(k, s) for k, s in sites if alt_carrier[s, ai]]
                if not carried:
                    continue
                kmer = list(seq[t * 6:(t + 1) * 6])
                for k, s in carried:
                    kmer[k] = alt_b[s]
                hap[i + 1, cpos[t + 1]] = id6("".join(kmer))
                own[i, cpos[t + 1]] = True
        keep_rows = own.any(dim=1)
        if not keep_rows.any():
            self.skip_counts["no_lines"] += 1
            return None
        kr = keep_rows.nonzero(as_tuple=True)[0]
        z = self.z_all[gi][accs[kr.numpy()]].astype(np.float32)

        return GeneBatch(gene_id=self.gene_id[gi], fam_split=self.fam_split[gi],
                         ref_ids=ref_ids,
                         cache_idx=torch.tensor(cache, dtype=torch.long),
                         hap_ids=torch.cat([hap[:1], hap[1:][kr]]),
                         own_mask=own[kr],
                         z=torch.from_numpy(z),
                         lines=[self.eco[ai] for ai in accs[kr.numpy()]])

    def iter_batches(self, gene_ix: np.ndarray):
        for gi in gene_ix:
            b = self.build(int(gi))
            if b is not None:
                yield b

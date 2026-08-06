"""
scripts/snp_prior_window_lm.py
------------------------------
Carbon-derived per-SNP importance prior, source #1: **window-LM conservation**.

Score each window by how Carbon (a decoder-only causal DNA LM) reads its
*reference* sequence — allele-blind, computed from the reference window alone, so
it sidesteps the "the assayed SNP is just an LD tag" problem: we weight a SNP by
how functional its *neighborhood* looks, not by the variant itself.

Per window we compute, over the reference sequence's DNA tokens:
  * mean NLL  = −(1/T) Σ log p(tok_t | tok_<t)   — low = predictable/constrained
  * mean entropy of the next-token distribution   — low = the model is certain

"Important" has two opposite readings, resolved empirically (bake off on val):
  * ``--sign conserved``  : low score → high weight  (purifying-selection reading)
  * ``--sign surprising`` : high score → high weight (distinctive-element reading)

Writes the raw per-window scores (.npz, so alternate sign/transform priors can be
rebuilt with no GPU) and the per-variant prior (.pt) for ``--snp-prior``.

    python scripts/snp_prior_window_lm.py --carbon-size 500M --dataset soy \
        --half-window 500 --out $SVAR_SCRATCH/caches/soy/snp_prior_lm_hw500_500m.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.dataset import UniqueWindowDataset
from crop_embed.encoders import CARBON_MODEL_PATHS
from crop_embed.partitioner import SNPWindowPartitioner
from training.common import features as feat
from training.common.datasets import get_dataset

_CARBON_DNA_PREFIX = "<dna>"


def _reference_sequences(fasta_path, partitioner, snps_by_chrom):
    """One reference (variant-free) DNA string per window, in window order.

    Reuses UniqueWindowDataset's reference/FASTA machinery WITHOUT building the
    per-sample fingerprint index (not needed for reference-only scoring). Takes the
    already-loaded snps_by_chrom (reference windows carry no alts, so it's only used
    to warm the FASTA cache) to avoid a second multi-GB VCF parse."""
    ds = UniqueWindowDataset.__new__(UniqueWindowDataset)
    ds.fasta_path = fasta_path
    ds._snps_by_chrom = snps_by_chrom
    ds._build_reference_tables()
    return [ds.extract_sequence((w.chrom, w.start, w.end, ())) for w in partitioner.windows]


@torch.no_grad()
def score_windows(model, tokenizer, sequences, device, max_length, batch_size):
    """Per-window (mean NLL, mean next-token entropy) under the causal LM."""
    nll_out = np.empty(len(sequences), dtype=np.float64)
    ent_out = np.empty(len(sequences), dtype=np.float64)
    tokenizer.padding_side = "right"                       # causal-LM scoring wants right pad
    for i in range(0, len(sequences), batch_size):
        batch = [f"{_CARBON_DNA_PREFIX}{s}" for s in sequences[i:i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding="longest",
                           truncation=True, max_length=max_length,
                           add_special_tokens=False).to(device)
        logits = model(**inputs).logits                   # (B, T, V)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # causal shift: position t predicts token t+1
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)        # (B, T-1, V)
        labels = ids[:, 1:]                                             # (B, T-1)
        lab_mask = mask[:, 1:].float()                                  # valid label positions
        nll_tok = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)    # (B, T-1)
        ent_tok = -(logp.exp() * logp).sum(-1)                          # (B, T-1)
        denom = lab_mask.sum(1).clamp_min(1.0)
        nll = (nll_tok * lab_mask).sum(1) / denom
        ent = (ent_tok * lab_mask).sum(1) / denom
        nll_out[i:i + len(batch)] = nll.double().cpu().numpy()
        ent_out[i:i + len(batch)] = ent.double().cpu().numpy()
        if (i // batch_size) % 50 == 0:
            print(f"  scored {i}/{len(sequences)} windows", flush=True)
    return nll_out, ent_out


def weights_from_scores(scores, sign: str, transform: str) -> np.ndarray:
    """Turn a per-window score into a non-negative prior weight (higher = up-weight)."""
    scores = np.asarray(scores, dtype=np.float64)
    if transform == "rank":
        order = scores.argsort()
        rank = np.empty_like(order, dtype=np.float64)
        rank[order] = np.arange(len(scores)) / max(len(scores) - 1, 1)   # → [0, 1]
        base = rank if sign == "surprising" else (1.0 - rank)
        return base + 1e-3
    # linear
    if sign == "surprising":
        return scores - scores.min() + 1e-3
    return scores.max() - scores + 1e-3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="soy", help="registry dataset (VCF/FASTA/pvar).")
    p.add_argument("--carbon-size", choices=list(CARBON_MODEL_PATHS), default="500M")
    p.add_argument("--model-path", default=None, help="override the carbon checkpoint id/path.")
    p.add_argument("--half-window", type=int, default=500)
    p.add_argument("--buffer", type=int, default=0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--score", choices=["nll", "entropy"], default="nll")
    p.add_argument("--sign", choices=["conserved", "surprising"], default="conserved")
    p.add_argument("--transform", choices=["rank", "linear"], default="rank")
    p.add_argument("--scores-from", default=None,
                   help="reuse a saved .npz of window scores (skip the LM forward).")
    p.add_argument("--out", required=True, help="output prior file (.pt).")
    args = p.parse_args()

    spec = get_dataset(args.dataset)
    print(f"Building windows for {spec.name} (hw={args.half_window}, buffer={args.buffer}) …",
          flush=True)
    snps_by_chrom, _ = load_snps_from_vcf(spec.vcf_path)
    part = SNPWindowPartitioner(snps_by_chrom, half_window=args.half_window, buffer=args.buffer)
    n_windows = len(part)
    print(f"  {n_windows:,} windows", flush=True)

    out = Path(args.out)
    scores_path = out.with_suffix(".windowscores.npz")
    if args.scores_from:
        z = np.load(args.scores_from)
        nll, ent = z["nll"], z["entropy"]
        print(f"Loaded window scores from {args.scores_from}", flush=True)
    else:
        print(f"Extracting {n_windows:,} reference window sequences …", flush=True)
        seqs = _reference_sequences(spec.fasta_path, part, snps_by_chrom)
        model_path = args.model_path or CARBON_MODEL_PATHS[args.carbon_size]
        print(f"Loading Carbon LM {model_path} on {args.device} …", flush=True)
        from CARBON_modules import load_carbon_lm
        model, tokenizer = load_carbon_lm(repo_id=model_path, device=args.device)
        print("Scoring windows under the causal LM …", flush=True)
        nll, ent = score_windows(model, tokenizer, seqs, args.device,
                                 args.max_length, args.batch_size)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(scores_path, nll=nll, entropy=ent)
        print(f"Saved raw window scores → {scores_path}", flush=True)

    for name, arr in (("nll", nll), ("entropy", ent)):
        print(f"  {name}: min={arr.min():.4g} median={np.median(arr):.4g} max={arr.max():.4g}",
              flush=True)

    scores = nll if args.score == "nll" else ent
    weights = weights_from_scores(scores, args.sign, args.transform)
    print(f"Mapping SNPs → windows and building prior "
          f"(score={args.score} sign={args.sign} transform={args.transform}) …", flush=True)
    prior = feat.snp_prior_from_window_scores(spec, args.half_window, weights, buffer=args.buffer)
    print(f"  prior covers {len(prior['variant_ids'])} variants", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**prior, "source": "window_lm", "score": args.score, "sign": args.sign,
                "transform": args.transform, "carbon_size": args.carbon_size,
                "half_window": args.half_window, "buffer": args.buffer}, out)
    print(f"Saved prior → {out}", flush=True)


if __name__ == "__main__":
    main()

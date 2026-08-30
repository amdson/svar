"""
variant_ll/run.py
-----------------
Runner for the allele log-likelihood benchmark: build windows, score the naive
baselines, evaluate Carbon through the variant cache (and optionally the exact
forward), fine-tune, and re-evaluate. Everything is reported in **bits per SNP on
held-out accessions**, so the model and the baselines are directly comparable.

    python -m variant_ll.run --dataset rice --half-window 500 --windows 200 --epochs 0
    python -m variant_ll.run --dataset rice --half-window 5000 --windows 400 --epochs 1

See BENCHMARK.md for the objective and PLAN.md for the run sequence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CARBON_modules import load_carbon_local, load_carbon_variant_cache
from variant_ll import baselines, loss
from variant_ll.data import WindowSource, genotype_split


def _resolve(dataset: str) -> tuple[str, str]:
    from training.common.datasets import get_dataset
    spec = get_dataset(dataset)
    return spec.vcf_path, spec.fasta_path


def build_batches(src: WindowSource, win_ids, rows, shuffle_sites=False, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for w in win_ids:
        b = src.build(w, rows, shuffle_sites=shuffle_sites, rng=rng)
        if b is not None:
            out.append(b)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="rice")
    p.add_argument("--model-path", default="HuggingFaceBio/Carbon-500M")
    p.add_argument("--half-window", type=int, default=500)
    p.add_argument("--buffer", type=int, default=0)
    p.add_argument("--chrom", type=int, nargs="*", default=[1])
    p.add_argument("--windows", type=int, default=200,
                   help="Number of windows to sample (the chromosome is the frame).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--hap-chunk", type=int, default=64)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--exact", action="store_true",
                   help="Also score the exact full-forward control.")
    p.add_argument("--permute", action="store_true",
                   help="Also score with LD destroyed (the mechanism control).")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="Write results JSON here.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    vcf_path, fasta_path = _resolve(args.dataset)

    # ── Windows ───────────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"Loading {args.dataset} VCF (chrom {args.chrom}) …")
    src = WindowSource(vcf_path, fasta_path, args.half_window, args.buffer, args.chrom)
    print(f"  {len(src.samples)} accessions, {len(src)} windows "
          f"(hw={args.half_window}) in {time.time()-t0:.0f}s")

    tr_rows, va_rows, _ = genotype_split(src.samples, args.seed)
    rng = np.random.default_rng(args.seed)
    win_ids = np.sort(rng.choice(len(src), size=min(args.windows, len(src)),
                                 replace=False))

    tr_batches = build_batches(src, win_ids, tr_rows)
    va_batches = build_batches(src, win_ids, va_rows)
    n_sites = sum(len(b.site_tok) for b in va_batches)
    n_tok = int(np.mean([len(b.ref_ids) for b in va_batches]))
    print(f"  train {len(tr_rows)} acc / val {len(va_rows)} acc | "
          f"{len(va_batches)} scoreable windows, {n_sites} sites, "
          f"~{n_tok} tokens/window")
    print(f"  site accounting: {src.stats}")
    up_frac = np.mean(np.concatenate(
        [b.has_upstream.numpy() for b in va_batches])) if va_batches else 0.0
    print(f"  sites with upstream context: {100*up_frac:.1f}%")
    if not va_batches:
        print("No scoreable windows — nothing to do."); return 1

    # ── Baselines (per window, then pooled over sites) ────────────────────────
    b_bits = {"b0": [0.0, 0], "b1": [0.0, 0], "b2": [0.0, 0]}
    for w in win_ids:
        tb = src.build(w, tr_rows)
        vb = src.build(w, va_rows)
        if tb is None or vb is None:
            continue
        tr_a = _expand(tb)
        va_a = _expand(vb)
        r = baselines.evaluate(tr_a, va_a, vb.site_tok.numpy())
        n = va_a.size
        b_bits["b0"][0] += 1.0 * n; b_bits["b0"][1] += n
        b_bits["b1"][0] += r["b1_bits"] * n; b_bits["b1"][1] += n
        b_bits["b2"][0] += r["b2_bits"] * n; b_bits["b2"][1] += n
    base = {k: v[0] / v[1] for k, v in b_bits.items()}
    print(f"\n  B0 uniform            {base['b0']:.4f} bits/SNP")
    print(f"  B1 per-site marginal  {base['b1']:.4f} bits/SNP   <- the bar")
    print(f"  B2 upstream Markov    {base['b2']:.4f} bits/SNP   "
          f"(headroom {base['b1']-base['b2']:+.4f})")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading Carbon on {device} ({dtype}) …")
    model, tokenizer = load_carbon_variant_cache(args.model_path, device=device,
                                                 dtype=dtype, backend="efficient")
    src.verify(tokenizer)          # analytic tokenization == the real tokenizer
    print("  tokenization verified against the tokenizer")

    tr_dev = [b.to(device) for b in tr_batches]
    va_dev = [b.to(device) for b in va_batches]

    results = {"config": vars(args), "baselines": base,
               "n_sites": n_sites, "upstream_frac": float(up_frac),
               "site_stats": src.stats}

    t0 = time.time()
    zs = loss.evaluate(model, va_dev, "cache", args.hap_chunk)
    print(f"\n  Carbon zero-shot (cache)  {zs['all_bits']:.4f} bits/SNP  "
          f"[up {zs['up_bits']:.4f} | no-up {zs['noup_bits']:.4f}]  "
          f"({time.time()-t0:.0f}s)")
    results["zeroshot_cache"] = zs

    if args.exact:
        exact, _ = load_carbon_local(args.model_path, device=device, dtype=dtype)
        ze = loss.evaluate(exact, va_dev, "exact", args.hap_chunk)
        print(f"  Carbon zero-shot (exact)  {ze['all_bits']:.4f} bits/SNP  "
              f"[up {ze['up_bits']:.4f} | no-up {ze['noup_bits']:.4f}]")
        results["zeroshot_exact"] = ze
        del exact
        torch.cuda.empty_cache()

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    if args.epochs:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        order = np.arange(len(tr_dev))
        for epoch in range(args.epochs):
            model.train()
            rng.shuffle(order)
            run_bits, run_n = 0.0, 0.0
            for step, i in enumerate(order):
                opt.zero_grad(set_to_none=True)
                bits, counts = loss.window_bits(model, tr_dev[i], "cache",
                                                args.hap_chunk)
                w = counts.unsqueeze(1).expand_as(bits)
                l = (bits * w).sum() / w.sum()
                l.backward()
                opt.step()
                run_bits += float((bits * w).sum()); run_n += float(w.sum())
                if step % 50 == 0 and step:
                    print(f"    epoch {epoch} step {step}/{len(order)}  "
                          f"train {run_bits/run_n:.4f} bits/SNP")
            ft = loss.evaluate(model, va_dev, "cache", args.hap_chunk)
            print(f"  epoch {epoch}: train {run_bits/run_n:.4f} | "
                  f"val {ft['all_bits']:.4f} bits/SNP  "
                  f"[up {ft['up_bits']:.4f} | no-up {ft['noup_bits']:.4f}]")
            results[f"ft_epoch{epoch}"] = ft

    # ── Mechanism control ─────────────────────────────────────────────────────
    if args.permute:
        perm = build_batches(src, win_ids, va_rows, shuffle_sites=True,
                             seed=args.seed)
        pm = loss.evaluate(model, [b.to(device) for b in perm], "cache",
                           args.hap_chunk)
        print(f"\n  LD-destroyed control      {pm['all_bits']:.4f} bits/SNP  "
              f"(vs {zs['all_bits']:.4f} intact; a model using haplotype "
              f"context should get worse)")
        results["permuted"] = pm

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {args.output}")
    return 0


def _expand(batch) -> np.ndarray:
    """Deduplicated haplotypes -> the per-accession (n, S) 0/1 matrix."""
    return np.repeat(batch.hap_allele.numpy().astype(np.int64),
                     batch.hap_count.numpy().astype(int), axis=0)


if __name__ == "__main__":
    sys.exit(main())

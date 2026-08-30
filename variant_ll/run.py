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
import math
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
    p.add_argument("--vcf-engine", choices=["auto", "polars", "pysam"], default="auto",
                   help="VCF reader. 'auto' takes the vectorized polars path for a "
                        "plain .vcf and pysam for a bgzipped one; polars is ~100x "
                        "faster and is what the arabidopsis build needs.")
    p.add_argument("--windows", type=int, default=200,
                   help="Number of windows to sample (the chromosome is the frame).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--hap-chunk", type=int, default=128,
                   help="Haplotypes per forward; 0 = the window's whole set. Bounds "
                        "peak memory in both eval and training (training goes "
                        "through loss.window_backward, which frees each chunk before "
                        "the next). Costs one repeated reference forward per chunk.")
    p.add_argument("--schedule", choices=["constant", "cosine"], default="cosine",
                   help="Post-warmup LR shape. Constant LR bounces near the end of "
                        "training here (5e-5 went 0.760 -> 0.878 over 200 steps).")
    p.add_argument("--warmup", type=int, default=100,
                   help="Linear LR warmup steps; the first windows are otherwise a "
                        "large step from a model that starts above 1 bit/SNP.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Also evaluate on val every N optimizer steps (0 = only at "
                        "epoch ends).")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--ckpt-reference", action="store_true",
                   help="Gradient-checkpoint the frozen-reference forward. Exact, "
                        "~1.4x slower, and the difference between fitting and OOM "
                        "once the window is a few thousand tokens (BENCHMARK.md 8a).")
    p.add_argument("--exact", action="store_true",
                   help="Also score the exact full-forward control.")
    p.add_argument("--permute", action="store_true",
                   help="Also score with LD destroyed (the mechanism control).")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="Write results JSON here.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    hap_chunk = args.hap_chunk or None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    vcf_path, fasta_path = _resolve(args.dataset)

    # ── Windows ───────────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"Loading {args.dataset} VCF (chrom {args.chrom}) …")
    src = WindowSource(vcf_path, fasta_path, args.half_window, args.buffer,
                       args.chrom, engine=args.vcf_engine)
    print(f"  {len(src.samples)} accessions, {len(src)} windows "
          f"(hw={args.half_window}) in {time.time()-t0:.0f}s")

    tr_rows, va_rows, _ = genotype_split(src.samples, args.seed)
    rng = np.random.default_rng(args.seed)
    win_ids = np.sort(rng.choice(len(src), size=min(args.windows, len(src)),
                                 replace=False))

    tr_batches = build_batches(src, win_ids, tr_rows)
    va_batches = build_batches(src, win_ids, va_rows)
    n_sites = sum(b.n_site for b in va_batches)
    n_tokens = sum(len(b.tok_idx) for b in va_batches)
    n_tok = int(np.mean([len(b.ref_ids) for b in va_batches]))
    print(f"  train {len(tr_rows)} acc / val {len(va_rows)} acc | "
          f"{len(va_batches)} scoreable windows, {n_sites} sites in "
          f"{n_tokens} scored tokens, ~{n_tok} tokens/window")
    print(f"  site accounting: {src.stats}")
    up_frac = (np.average(
        np.concatenate([b.has_upstream.numpy() for b in va_batches]),
        weights=np.concatenate([b.tok_nsite.numpy() for b in va_batches]))
        if va_batches else 0.0)
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
    model.encoder.reference_checkpointing = args.ckpt_reference
    src.verify(tokenizer)          # analytic tokenization == the real tokenizer
    print("  tokenization verified against the tokenizer")

    tr_dev = [b.to(device) for b in tr_batches]
    va_dev = [b.to(device) for b in va_batches]

    results = {"config": vars(args), "baselines": base,
               "n_sites": n_sites, "upstream_frac": float(up_frac),
               "site_stats": src.stats}

    t0 = time.time()
    zs = loss.evaluate(model, va_dev, "cache", hap_chunk)
    print(f"\n  Carbon zero-shot (cache)  {zs['all_bits']:.4f} bits/SNP  "
          f"[up {zs['up_bits']:.4f} | no-up {zs['noup_bits']:.4f}]  "
          f"({time.time()-t0:.0f}s)")
    results["zeroshot_cache"] = zs

    if args.exact:
        exact, _ = load_carbon_local(args.model_path, device=device, dtype=dtype)
        ze = loss.evaluate(exact, va_dev, "exact", hap_chunk)
        print(f"  Carbon zero-shot (exact)  {ze['all_bits']:.4f} bits/SNP  "
              f"[up {ze['up_bits']:.4f} | no-up {ze['noup_bits']:.4f}]")
        results["zeroshot_exact"] = ze
        del exact
        torch.cuda.empty_cache()

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    history = []
    results["history"] = history
    if args.epochs:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        order = np.arange(len(tr_dev))
        total_steps = args.epochs * len(tr_dev)
        gstep = 0
        best = {"all_bits": float("inf")}

        def record(tag, train_bits):
            ft = loss.evaluate(model, va_dev, "cache", hap_chunk)
            row = {"tag": tag, "step": gstep, "train_bits": train_bits, **ft}
            history.append(row)
            flag = ""
            if ft["all_bits"] < best["all_bits"]:
                best.update(ft); best["step"] = gstep; flag = "  *best"
            print(f"  {tag}: train {train_bits:.4f} | val {ft['all_bits']:.4f} "
                  f"bits/SNP  [up {ft['up_bits']:.4f} | no-up {ft['noup_bits']:.4f}]"
                  f"  vs B1 {base['b1']:.4f} ({ft['all_bits']-base['b1']:+.4f}){flag}")
            model.train()
            return ft

        for epoch in range(args.epochs):
            model.train()
            rng.shuffle(order)
            run_bits, run_n = 0.0, 0.0
            win_bits, win_n = 0.0, 0.0          # since the last printed line
            for step, i in enumerate(order):
                # Warmup: the model starts above 1 bit/SNP against a 0.34 target, so
                # the first steps are a long way downhill and a cold full-LR AdamW
                # step on 500M params can wreck the checkpoint before it settles.
                if gstep < args.warmup:
                    scale = (gstep + 1) / args.warmup
                elif args.schedule == "cosine":
                    prog = (gstep - args.warmup) / max(1, total_steps - args.warmup)
                    scale = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
                else:
                    scale = 1.0
                for g in opt.param_groups:
                    g["lr"] = args.lr * scale
                opt.zero_grad(set_to_none=True)
                b_sum, n_sum = loss.window_backward(model, tr_dev[i], "cache",
                                                    hap_chunk)
                opt.step()
                gstep += 1
                run_bits += b_sum; run_n += n_sum
                win_bits += b_sum; win_n += n_sum
                if step % 100 == 0 and step:
                    print(f"    epoch {epoch} step {step}/{len(order)}  "
                          f"train {win_bits/win_n:.4f} bits/SNP (last 100)")
                    win_bits, win_n = 0.0, 0.0
                if args.eval_every and gstep % args.eval_every == 0:
                    record(f"step {gstep}", run_bits / run_n)
            results[f"ft_epoch{epoch}"] = record(f"epoch {epoch}", run_bits / run_n)

        results["best_val"] = best
        print(f"\n  best val {best['all_bits']:.4f} bits/SNP at step {best['step']}"
              f"  (B1 {base['b1']:.4f}, B2 {base['b2']:.4f})")

    # ── Mechanism control ─────────────────────────────────────────────────────
    # Compare against the *current* model's intact score, not the zero-shot one:
    # after fine-tuning those are different models, and pairing a fine-tuned
    # permuted number with a zero-shot intact number reads as a large degradation
    # that is really just the fine-tuning.
    if args.permute:
        perm = build_batches(src, win_ids, va_rows, shuffle_sites=True,
                             seed=args.seed)
        pm = loss.evaluate(model, [b.to(device) for b in perm], "cache",
                           hap_chunk)
        intact = results[f"ft_epoch{args.epochs - 1}"] if args.epochs else zs
        stage = f"after {args.epochs} epoch(s)" if args.epochs else "zero-shot"
        print(f"\n  LD-destroyed control      {pm['all_bits']:.4f} bits/SNP  "
              f"(vs {intact['all_bits']:.4f} intact, same model, {stage}; "
              f"a model using haplotype context should get worse)")
        results["permuted"] = pm
        results["permuted_vs"] = stage

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

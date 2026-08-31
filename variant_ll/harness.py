"""
variant_ll/harness.py
---------------------
The shared run loop for the allele log-likelihood benchmark, in the shape
``training/common/harness.py`` uses: a thin ``run.py`` parses args and hands them
here, and this does everything shared — build windows, fit the baselines, load
the model, fine-tune, evaluate, and write the run record.

Why this is one loop and not several scripts. The three experiments that answer
BENCHMARK.md's questions differ only in *which accessions get scored* and *which
backend runs*:

  * held-out benchmark   --eval-accessions heldout  --backend cache
  * capacity probe       --eval-accessions insample --backend cache
  * approximation cost   --eval-accessions insample --backend exact

Everything else — window selection, dedup, baselines, the optimizer schedule, the
diagnostic slices — is identical, and it has to *stay* identical or the numbers
stop being comparable. Two of those were originally throwaway scripts whose
numbers were only comparable by luck.

The unit is bits per SNP on the scored accessions, against B1 (per-site marginal)
and B2 (nearest-upstream-site Markov). See BENCHMARK.md for the objective.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from crop_embed import MetricLogger, metrics_path_for
from training.common import artifacts, run_record
from variant_ll import baselines, loss
from variant_ll.data import WindowSource


# ── CLI ───────────────────────────────────────────────────────────────────────

def add_common_args(p: argparse.ArgumentParser) -> None:
    """Every knob that changes a number. Mirrors training/common/harness.py."""
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="arabidopsis", help="registry name")
    g.add_argument("--chrom", type=int, nargs="*", default=[4],
                   help="chromosomes to draw windows from (PLAN.md picks chr4: "
                        "smallest and effectively N-free)")
    g.add_argument("--half-window", type=int, default=500)
    g.add_argument("--buffer", type=int, default=0)
    g.add_argument("--windows", type=int, default=200,
                   help="windows sampled from the chromosome (the sampling frame)")
    g.add_argument("--vcf-engine", choices=["auto", "polars", "pysam"], default="auto",
                   help="'auto' takes the vectorized polars path for a plain .vcf "
                        "and pysam for a bgzipped one; polars is ~100x faster and "
                        "is what any large VCF needs.")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("evaluation")
    g.add_argument("--eval-accessions", choices=["heldout", "insample"],
                   default="heldout",
                   help="'heldout' scores accessions the model never trained on — "
                        "the actual benchmark. 'insample' scores the training "
                        "accessions themselves: pure capacity, no generalisation, "
                        "and the baselines are refit on those same accessions so "
                        "the comparison stays honest.")
    g.add_argument("--eval-split", choices=["val", "test"], default="val",
                   help="which held-out partition to score (tune on val, touch "
                        "test once). Ignored when --eval-accessions insample.")
    g.add_argument("--eval-every", type=int, default=0,
                   help="also evaluate every N optimizer steps (0 = epoch ends only)")
    g.add_argument("--eval-epochs", type=int, default=1,
                   help="evaluate at the end of every Nth epoch (the last is always "
                        "scored). On a small window set an epoch is cheap but an "
                        "eval is not — it forwards every fit and eval accession "
                        "over every window — so at high epoch counts a per-epoch "
                        "eval costs more than the training it is measuring.")
    g.add_argument("--no-eval-train", action="store_true",
                   help="skip scoring the training accessions at each eval. They "
                        "are scored by default: the train-vs-eval gap is what "
                        "separates 'overfitting' from 'converged to the baseline', "
                        "and the running mean printed during an epoch cannot "
                        "answer that (it lags, being averaged over changing weights).")
    g.add_argument("--baselines-only", action="store_true",
                   help="score B0/B1/B2 on this window set and stop, before any "
                        "model is loaded. Costs no GPU and ~a minute, and it "
                        "establishes the bar on the exact windows and the exact "
                        "split the model run will use — so the comparison is "
                        "against a number that was fixed in advance.")

    g = p.add_argument_group("model")
    g.add_argument("--model-path", default="HuggingFaceBio/Carbon-500M")
    g.add_argument("--backend", choices=["cache", "exact"], default="cache",
                   help="'cache' is the variant cache under test; 'exact' is the "
                        "full causal forward it approximates (BENCHMARK.md §7's "
                        "ceiling). Both score identical candidates.")
    g.add_argument("--arch", choices=["full", "lora"], default="full",
                   help="'full' fine-tunes every weight (VariantCacheCarbonForCausalLM). "
                        "'lora' trains variant-only LoRA adapters over a frozen base "
                        "(VariantLoRACarbonForCausalLM): the reference stream becomes "
                        "a constant of the window, so it is computed once per window "
                        "for the whole run instead of once per haplotype chunk per "
                        "step. Fewer trainable parameters, so expect it to fit less "
                        "well at a fixed window count — the point is throughput.")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=float, default=32.0)
    g.add_argument("--lora-dropout", type=float, default=0.0)
    g.add_argument("--lora-targets", nargs="*", default=None,
                   help="projections to adapt; default q_proj o_proj gate_proj "
                        "up_proj down_proj — the ones applied ONLY to the variant "
                        "stream. Adding k_proj/v_proj adapts the variant keys but "
                        "not the reference keys, which changes what the log-sum-exp "
                        "merge is combining; deliberate, not free.")
    g.add_argument("--train-base", action="store_true",
                   help="with --arch lora, train the base weights alongside the "
                        "adapters. Strictly MORE expressive than --arch full: the "
                        "base is shared by the reference and variant streams so it "
                        "moves both together, while the adapters move the variant "
                        "stream alone, and neither alone spans the pair. Costs the "
                        "reference cache (the reference stream stops being constant) "
                        "and forces fp32.")
    g.add_argument("--no-ref-cache", action="store_true",
                   help="with --arch lora, recompute the frozen reference stream "
                        "instead of caching it. Only useful for measuring what the "
                        "cache saves.")
    g.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                   help="fp32 by default and deliberately: bf16 caps this "
                        "benchmark at the baseline and looks like a converged null "
                        "result (BENCHMARK.md §8c). Treat any bf16 number as a "
                        "lower bound.")
    g.add_argument("--ckpt-reference", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="gradient-checkpoint the frozen-reference forward: exact, "
                        "~1.4x slower, and the difference between fitting and OOM "
                        "on a long window (BENCHMARK.md §8a)")
    g.add_argument("--hap-chunk", type=int, default=32,
                   help="haplotypes per forward; 0 = the window's whole set. Bounds "
                        "peak memory in training too (loss.window_backward frees "
                        "each chunk). Note logits are (N, C, vocab~152k), so fp32 "
                        "wants 32 here; the exact backend wants 8.")

    g = p.add_argument_group("optimization")
    g.add_argument("--epochs", type=int, default=0,
                   help="0 = zero-shot evaluation only")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    g.add_argument("--warmup", type=int, default=200)
    g.add_argument("--accum-windows", type=int, default=8,
                   help="windows accumulated into one optimizer step. One window "
                        "is ~1 kb of genome, so its sites are heavily correlated "
                        "and a batch of one is a very noisy gradient. The group is "
                        "normalized by its TOTAL scored-site weight, so the step "
                        "minimizes mean bits/SNP — the reported metric — rather "
                        "than the mean over per-window means.")
    g.add_argument("--grad-clip", type=float, default=0.0,
                   help="0 = off")

    g = p.add_argument_group("controls (BENCHMARK.md §7)")
    g.add_argument("--permute", action="store_true",
                   help="also score with each site's alleles permuted across "
                        "accessions, destroying LD while preserving every site's "
                        "marginal. NB this also puts the input off-distribution, "
                        "so degradation is weaker evidence of LD use than it looks.")
    g.add_argument("--zeroshot-exact", action="store_true",
                   help="additionally score the exact full forward before training")

    g = p.add_argument_group("output")
    g.add_argument("--output", default=None,
                   help="checkpoint path; the JSONL metric sidecar and run.json go "
                        "beside it. Default: an auto-named dir under $SVAR_SCRATCH/runs/.")
    g.add_argument("--save-model", action="store_true",
                   help="write the fine-tuned weights (~2 GB in fp32): "
                        "model_best.pt on every new best eval, and model.pt at the "
                        "end. Off by default so a sweep does not fill scratch, but "
                        "turn it on for any run whose number you may want to "
                        "re-verify — a run that is killed or that diverges after "
                        "its peak otherwise leaves no weights behind at all.")
    g.add_argument("--out-root", default=None, help="override runs/ root")
    g.add_argument("--strict", action="store_true",
                   help="content-hash artifacts into the run record")
    g.add_argument("--device", default=None)
    g.add_argument("--wandb-project", default=None)
    g.add_argument("--wandb-name", default=None)


# ── Window/accession assembly ─────────────────────────────────────────────────

@dataclass
class WindowSet:
    """Windows plus the accession rows scored on each side of the comparison."""
    source: WindowSource
    win_ids: np.ndarray
    fit_rows: np.ndarray        # accessions the model trains on; baselines fit here
    eval_rows: np.ndarray       # accessions scored
    split_path: Path | None = None
    train_batches: list = field(default_factory=list)
    eval_batches: list = field(default_factory=list)

    @property
    def insample(self) -> bool:
        return len(self.fit_rows) == len(self.eval_rows) and \
            bool(np.array_equal(self.fit_rows, self.eval_rows))


def build_windows(args) -> WindowSet:
    """Resolve the dataset, sample windows, and build both sides' batches."""
    from training.common.datasets import get_dataset
    spec = get_dataset(args.dataset)

    t0 = time.time()
    print(f"Loading {args.dataset} VCF (chrom {args.chrom}) …")
    src = WindowSource(spec.vcf_path, spec.fasta_path, args.half_window,
                       args.buffer, args.chrom, engine=args.vcf_engine)
    print(f"  {len(src.samples)} accessions, {len(src)} windows "
          f"(hw={args.half_window}) in {time.time() - t0:.0f}s")

    # The split every model in the repo shares (training/common/splits.py), keyed
    # by sample ID and committed, so a variant_ll held-out accession is held out
    # for the phenotype models too — no cross-model leakage, and these numbers can
    # be set beside phenotype results later. `indices` resolves IDs against the
    # VCF's column order and raises if any is absent.
    #
    # It partitions *phenotyped* samples, so on a panel where some accessions have
    # no phenotype at all it covers less than the full VCF: arabidopsis 1,041 of
    # 1,135 (729/156/156), rice all 383 (269/57/57). Those 94 are simply not
    # scored — a slightly smaller and not-quite-random panel, which is the price of
    # a shared split.
    from training.common.splits import get_or_build_split, default_split_path
    split = get_or_build_split(args.dataset, seed=args.seed)
    tr_rows = split.indices("train", src.samples)
    va_rows = split.indices("val", src.samples)
    te_rows = split.indices("test", src.samples)
    split_path = default_split_path(args.dataset, args.seed)
    n_cov = len(tr_rows) + len(va_rows) + len(te_rows)
    if n_cov < len(src.samples):
        print(f"  split covers {n_cov}/{len(src.samples)} genotyped accessions "
              f"({len(src.samples) - n_cov} have no phenotype and are not scored)")
    if args.eval_accessions == "insample":
        eval_rows = tr_rows
    else:
        eval_rows = va_rows if args.eval_split == "val" else te_rows

    rng = np.random.default_rng(args.seed)
    win_ids = np.sort(rng.choice(len(src), size=min(args.windows, len(src)),
                                 replace=False))

    ws = WindowSet(source=src, win_ids=win_ids, fit_rows=tr_rows,
                   eval_rows=eval_rows, split_path=split_path)
    ws.train_batches = build_batches(src, win_ids, tr_rows)
    ws.eval_batches = (ws.train_batches if ws.insample
                       else build_batches(src, win_ids, eval_rows))

    n_sites = sum(b.n_site for b in ws.eval_batches)
    n_tok = sum(len(b.tok_idx) for b in ws.eval_batches)
    tokens = int(np.mean([len(b.ref_ids) for b in ws.eval_batches])) if ws.eval_batches else 0
    mode = "in-sample" if ws.insample else f"held-out ({args.eval_split})"
    print(f"  fit on {len(tr_rows)} acc | scoring {len(eval_rows)} acc [{mode}]")
    print(f"  {len(ws.eval_batches)} scoreable windows, {n_sites} sites in "
          f"{n_tok} tokens, ~{tokens} tokens/window")
    print(f"  site accounting: {src.stats}")
    print(f"  sites with upstream context: {100 * upstream_frac(ws.eval_batches):.1f}%")
    return ws


def build_batches(src: WindowSource, win_ids, rows, shuffle_sites=False, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for w in win_ids:
        b = src.build(w, rows, shuffle_sites=shuffle_sites, rng=rng)
        if b is not None:
            out.append(b)
    return out


def site_upstream_flags(batch) -> np.ndarray:
    """(S,) per-*site* upstream flag, from the per-token one.

    ``has_upstream`` is a property of the token (sites sharing a token are
    predicted simultaneously and give each other no context), but the baselines
    are per-site, so slicing them the same way the model is sliced means
    expanding by the token's site count.
    """
    return np.repeat(batch.has_upstream.numpy(), batch.tok_nsite.numpy())


def upstream_frac(batches) -> float:
    if not batches:
        return 0.0
    return float(np.average(
        np.concatenate([b.has_upstream.numpy() for b in batches]),
        weights=np.concatenate([b.tok_nsite.numpy() for b in batches])))


# ── Baselines, sliced the same way the model is ───────────────────────────────

def score_baselines(ws: WindowSet) -> dict:
    """B0/B1/B2 in bits/SNP, overall and split by upstream context.

    The slices matter more than the headline: BENCHMARK.md's argument is that a
    model *cannot* beat B1 on no-upstream sites (its prediction there is identical
    for every accession), so any real win has to show up in the ``up`` slice. That
    is only checkable if the baselines are sliced identically, which is why this
    returns per-slice numbers rather than one scalar.
    """
    acc = {k: [0.0, 0.0] for k in
           ("b0_all", "b1_all", "b2_all", "b1_up", "b2_up", "b1_noup", "b2_noup",
            "b0_up", "b0_noup")}

    for w in ws.win_ids:
        tb = ws.source.build(w, ws.fit_rows)
        eb = tb if ws.insample else ws.source.build(w, ws.eval_rows)
        if tb is None or eb is None:
            continue
        tr_a, ev_a = expand_alleles(tb), expand_alleles(eb)
        r = baselines.evaluate(tr_a, ev_a, eb.site_tok.numpy())
        up = site_upstream_flags(eb)                      # (S,)
        for tag, bits in (("b1", r["b1_per_call"]), ("b2", r["b2_per_call"])):
            _accum(acc, f"{tag}_all", bits)
            _accum(acc, f"{tag}_up", bits[:, up])
            _accum(acc, f"{tag}_noup", bits[:, ~up])
        ones = np.ones_like(r["b1_per_call"])
        _accum(acc, "b0_all", ones)
        _accum(acc, "b0_up", ones[:, up])
        _accum(acc, "b0_noup", ones[:, ~up])

    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}


def _accum(acc: dict, key: str, bits: np.ndarray) -> None:
    acc[key][0] += float(bits.sum())
    acc[key][1] += int(bits.size)


def expand_alleles(batch) -> np.ndarray:
    """Deduplicated haplotypes -> the per-accession (n, S) 0/1 matrix."""
    return np.repeat(batch.hap_allele.numpy().astype(np.int64),
                     batch.hap_count.numpy().astype(int), axis=0)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(args, device, dtype):
    """The variant-cache model, or the plain causal LM for the exact backend."""
    from CARBON_modules import (load_carbon_local, load_carbon_variant_cache,
                                load_carbon_variant_lora)
    if getattr(args, "arch", "full") == "lora":
        if args.backend != "cache":
            raise ValueError("--arch lora requires --backend cache")
        # bf16 is safe for the base here in a way it is not for full fine-tuning:
        # the base is frozen, and the adapters stay fp32 (see LoRADelta), so the
        # AdamW-moments-in-bf16 failure that caps full fine-tuning cannot occur.
        model, tokenizer = load_carbon_variant_lora(
            args.model_path, device=device, base_dtype=dtype,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            targets=tuple(args.lora_targets) if args.lora_targets else None,
            freeze_base=not args.train_base)
        if args.train_base:
            # The reference pass needs its graph back, and that graph is what
            # forced checkpointing on the full model in the first place.
            model.encoder.reference_checkpointing = args.ckpt_reference
        summ = model.parameter_summary()
        print(f"  LoRA r={args.lora_r} alpha={args.lora_alpha:g} on "
              f"{', '.join(model.lora_config.targets)}")
        print(f"  trainable {summ['trainable']:,} / {summ['total']:,} "
              f"({100 * summ['fraction']:.2f}%)")
        return model, tokenizer
    if args.backend == "cache":
        model, tokenizer = load_carbon_variant_cache(
            args.model_path, device=device, dtype=dtype, backend="efficient")
        model.encoder.reference_checkpointing = args.ckpt_reference
    else:
        model, tokenizer = load_carbon_local(args.model_path, device=device,
                                             dtype=dtype)
    return model, tokenizer


def lr_scale(step: int, total: int, args, warmup: int) -> float:
    """Linear warmup then constant or cosine, over *optimizer* steps.

    Warmup is not optional here: the model starts above 1 bit/SNP against a ~0.36
    target, so the first steps are a long way downhill and a cold full-LR AdamW
    step on 500M parameters can wreck the checkpoint before it settles.
    """
    if step < warmup:
        return (step + 1) / warmup
    if args.schedule == "cosine":
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return 1.0


# ── The run ───────────────────────────────────────────────────────────────────

def run(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    hap_chunk = args.hap_chunk or None

    ws = build_windows(args)
    if not ws.eval_batches:
        print("No scoreable windows — nothing to do.")
        return {"status": "empty"}

    base = score_baselines(ws)
    print(f"\n  B0 uniform            {base['b0_all']:.4f} bits/SNP")
    print(f"  B1 per-site marginal  {base['b1_all']:.4f} bits/SNP   <- the bar   "
          f"[up {base['b1_up']:.4f} | no-up {base['b1_noup']:.4f}]")
    print(f"  B2 upstream Markov    {base['b2_all']:.4f} bits/SNP   "
          f"(headroom {base['b1_all'] - base['b2_all']:+.4f})   "
          f"[up {base['b2_up']:.4f} | no-up {base['b2_noup']:.4f}]")

    if args.baselines_only:
        return baselines_only(args, ws, base)

    print(f"\nLoading Carbon on {device} ({dtype}, backend={args.backend}) …")
    model, tokenizer = load_model(args, device, dtype)
    ws.source.verify(tokenizer)
    print("  tokenization verified against the tokenizer")

    tr_dev = [b.to(device) for b in ws.train_batches]
    ev_dev = tr_dev if ws.insample else [b.to(device) for b in ws.eval_batches]

    # Only meaningful with a frozen base — ReferenceCache re-checks that itself
    # and raises rather than serving a stale stream.
    ref_cache = None
    if (getattr(args, "arch", "full") == "lora"
            and not args.no_ref_cache and not args.train_base):
        from CARBON_modules import ReferenceCache
        ref_cache = ReferenceCache(model)

    out_path, run_dir = resolve_output(args)
    logger = MetricLogger(metrics_path_for(out_path),
                          wandb_project=args.wandb_project,
                          wandb_name=args.wandb_name, config=vars(args))
    history: list[dict] = []
    results = {"config": vars(args), "baselines": base,
               "n_sites": sum(b.n_site for b in ws.eval_batches),
               "n_tokens": sum(len(b.tok_idx) for b in ws.eval_batches),
               "n_windows": len(ws.eval_batches),
               "n_fit_accessions": int(len(ws.fit_rows)),
               "n_eval_accessions": int(len(ws.eval_rows)),
               "upstream_frac": upstream_frac(ws.eval_batches),
               "site_stats": dict(ws.source.stats), "history": history}

    def score(tag: str, step: int, train_running: float | None = None) -> dict:
        ev = loss.evaluate(model, ev_dev, args.backend, hap_chunk, ref_cache)
        row = {"tag": tag, "step": step,
               "eval_bits": ev["all_bits"], "eval_up": ev["up_bits"],
               "eval_noup": ev["noup_bits"]}
        if train_running is not None:
            row["train_running"] = train_running
        if not args.no_eval_train and not ws.insample:
            # With the *final* weights, unlike the running mean, so the train-vs-eval
            # gap is a real generalisation measurement rather than a lag artifact.
            tr = loss.evaluate(model, tr_dev, args.backend, hap_chunk, ref_cache)
            row.update(train_bits=tr["all_bits"], train_up=tr["up_bits"],
                       gap=ev["all_bits"] - tr["all_bits"])
        history.append(row)
        logger.log(row)
        return row

    t0 = time.time()
    zs = score("zeroshot", 0)
    print(f"\n  zero-shot ({args.backend})  {zs['eval_bits']:.4f} bits/SNP  "
          f"[up {zs['eval_up']:.4f} | no-up {zs['eval_noup']:.4f}]  "
          f"({time.time() - t0:.0f}s)")
    results["zeroshot"] = zs

    if args.zeroshot_exact and args.backend == "cache":
        from CARBON_modules import load_carbon_local
        exact, _ = load_carbon_local(args.model_path, device=device, dtype=dtype)
        ze = loss.evaluate(exact, ev_dev, "exact", min(hap_chunk or 8, 8))
        print(f"  zero-shot (exact)     {ze['all_bits']:.4f} bits/SNP  "
              f"[up {ze['up_bits']:.4f} | no-up {ze['noup_bits']:.4f}]")
        results["zeroshot_exact"] = ze
        del exact
        torch.cuda.empty_cache()

    best = dict(zs)
    if args.epochs:
        best = fine_tune(model, tr_dev, args, score, base, results,
                         ckpt_path=out_path.with_name("model_best.pt"),
                         ref_cache=ref_cache)

    if args.permute:
        results["permuted"] = permutation_control(model, ws, args, hap_chunk,
                                                  best, results, ref_cache)

    results["best"] = best
    if ref_cache is not None:
        st = ref_cache.stats()
        results["ref_cache"] = st
        # misses should equal the window count exactly: one reference pass per
        # window for the whole run. Anything more means it is being recomputed.
        print(f"\n  reference cache: {st['entries']} windows, {st['mb']:.0f} MB, "
              f"{st['hits']:,} hits / {st['misses']:,} misses "
              f"({100 * st['hit_rate']:.1f}% hit rate)")
    finalize(args, model, results, base, ws, out_path, run_dir)
    logger.close()
    return results


def save_checkpoint(model, args, results, best: dict, path: Path) -> None:
    """Write a checkpoint atomically.

    Via a temp file + rename because this is called *during* training, on every
    new best: a run killed mid-write would otherwise leave a truncated file where
    the previous best used to be, which is exactly the checkpoint you want when a
    run is killed. Rename is atomic within a filesystem, so the old best survives
    until the new one is complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # For LoRA only the adapters changed; writing 2 GB of frozen base on every
    # new best would dominate the step it is saving.
    sd = (model.checkpoint_state_dict()
          if hasattr(model, "checkpoint_state_dict") else model.state_dict())
    torch.save({"model_state_dict": sd, "args": vars(args),
                "best": dict(best),
                "results": {k: v for k, v in results.items() if k != "history"}}, tmp)
    tmp.replace(path)


def baselines_only(args, ws: WindowSet, base: dict) -> dict:
    """The bar for this window set, recorded without loading the model.

    Same WindowSet, same shared split, same per-slice accounting as a full run —
    so the B1 a model is later compared against is a number that existed before
    the model did, not one produced in the same process that produced the model's.
    """
    out_path, run_dir = resolve_output(args)
    results = {"config": vars(args), "baselines": base,
               "n_sites": sum(b.n_site for b in ws.eval_batches),
               "n_tokens": sum(len(b.tok_idx) for b in ws.eval_batches),
               "n_windows": len(ws.eval_batches),
               "n_fit_accessions": int(len(ws.fit_rows)),
               "n_eval_accessions": int(len(ws.eval_rows)),
               "upstream_frac": upstream_frac(ws.eval_batches),
               "site_stats": dict(ws.source.stats), "history": []}
    artifacts.save_json(run_dir / "results.json", results)

    phase = "test" if (args.eval_split == "test" and
                       args.eval_accessions == "heldout") else "val"
    rec = run_record.build(
        dataset=args.dataset, features="variant_ll", model="baselines",
        seed=args.seed, traits=[],
        hyperparams={"half_window": args.half_window, "windows": args.windows,
                     "chrom": args.chrom, "eval_accessions": args.eval_accessions},
        metrics={phase: {"mean": {"b1_bits": base["b1_all"],
                                  "b2_bits": base["b2_all"],
                                  "b1_bits_up": base["b1_up"],
                                  "b2_bits_up": base["b2_up"]}}},
        half_window=args.half_window,
        split_path=str(ws.split_path) if ws.split_path else None,
        strict=args.strict,
        extra={"n_sites": results["n_sites"], "n_windows": results["n_windows"],
               "upstream_frac": results["upstream_frac"],
               "site_stats": results["site_stats"],
               "eval_accessions": args.eval_accessions,
               "eval_split": args.eval_split})
    run_record.write(rec, run_dir)
    print(f"\n[{args.dataset}/variant_ll/baselines] run_id={rec.run_id}\n  -> {run_dir}")
    return results


def fine_tune(model, tr_dev, args, score, base, results,
              ckpt_path: Path | None = None, ref_cache=None) -> dict:
    # trainable_parameters() on the LoRA model is the adapters only; handing
    # AdamW model.parameters() there would build moments for 500M frozen weights.
    params = (model.trainable_parameters()
              if hasattr(model, "trainable_parameters") else list(model.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(tr_dev))
    accum = max(1, args.accum_windows)
    steps_per_epoch = math.ceil(len(tr_dev) / accum)
    total = args.epochs * steps_per_epoch
    # --warmup is in optimizer steps, and accumulation cuts those by `accum`, so a
    # warmup tuned for one-window steps can swallow a whole short run. Clamp it.
    warmup = min(args.warmup, max(1, total // 10))
    hap_chunk = args.hap_chunk or None
    gstep = 0
    best = {"eval_bits": float("inf")}
    print(f"  accumulating {accum} window(s)/step -> {steps_per_epoch} steps/epoch, "
          f"{total} total; warmup {warmup}"
          + (f" (clamped from {args.warmup})" if warmup != args.warmup else ""))

    def maybe_best(row):
        if row["eval_bits"] < best["eval_bits"]:
            best.clear(); best.update(row)
            # Written here, not at the end: the previous 40-window run peaked at
            # epoch 119, then diverged, and was killed at 146 — the best weights
            # had never been written anywhere and the number was unreproducible.
            if args.save_model and ckpt_path is not None:
                save_checkpoint(model, args, results, best, ckpt_path)
                return "  *best (saved)"
            return "  *best"
        return ""

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(order)
        run_bits = run_n = 0.0
        win_bits = win_n = 0.0
        for step, gi in enumerate(range(0, len(order), accum)):
            group = [tr_dev[i] for i in order[gi:gi + accum]]
            # The normalizer is the group's total scored-site weight, known before
            # any forward, so every site in the group counts once no matter which
            # window it came from.
            denom = loss.group_weight(group)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_scale(gstep, total, args, warmup)
            opt.zero_grad(set_to_none=True)
            for b in group:
                b_sum, n_sum = loss.window_backward(model, b, args.backend,
                                                    hap_chunk, denom=denom,
                                                    ref_cache=ref_cache)
                run_bits += b_sum; run_n += n_sum
                win_bits += b_sum; win_n += n_sum
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            gstep += 1
            if step % 100 == 0 and step:
                print(f"    epoch {epoch} step {step}/{steps_per_epoch}  "
                      f"train {win_bits / win_n:.4f} bits/SNP (last 100)", flush=True)
                win_bits = win_n = 0.0
            if args.eval_every and gstep % args.eval_every == 0:
                row = score(f"step {gstep}", gstep, run_bits / run_n)
                print(f"  {row['tag']}: eval {row['eval_bits']:.4f} vs B1 "
                      f"{base['b1_all']:.4f} ({row['eval_bits'] - base['b1_all']:+.4f})"
                      f"{maybe_best(row)}", flush=True)
                model.train()
        if (epoch + 1) % max(1, args.eval_epochs) and epoch != args.epochs - 1:
            print(f"  epoch {epoch}: train {run_bits / run_n:.4f} bits/SNP", flush=True)
            continue
        row = score(f"epoch {epoch}", gstep, run_bits / run_n)
        extra = (f" | train {row['train_bits']:.4f} (gap {row['gap']:+.4f})"
                 if "train_bits" in row else "")
        print(f"  epoch {epoch}: eval {row['eval_bits']:.4f} bits/SNP "
              f"[up {row['eval_up']:.4f} | no-up {row['eval_noup']:.4f}]{extra}  "
              f"vs B1 {base['b1_all']:.4f} "
              f"({row['eval_bits'] - base['b1_all']:+.4f}){maybe_best(row)}", flush=True)
        model.train()

    results["final_epoch"] = history_last(results)
    print(f"\n  best eval {best['eval_bits']:.4f} bits/SNP at step {best['step']}"
          f"  (B1 {base['b1_all']:.4f}, B2 {base['b2_all']:.4f})")
    return best


def history_last(results) -> dict:
    h = results.get("history") or []
    return h[-1] if h else {}


def permutation_control(model, ws: WindowSet, args, hap_chunk, best, results,
                        ref_cache=None) -> dict:
    """Score with LD destroyed but every site's marginal preserved.

    Read this with care. Permuting alleles across accessions also produces
    haplotypes that do not occur in the population, so the inputs go
    off-distribution. Degradation is therefore consistent both with 'the model is
    using the individual's haplotype' and with 'the model memorised which
    haplotypes exist' — it is weaker evidence than BENCHMARK.md §7 assumes. The
    ``up`` vs ``no-up`` slices against B1 are the sharper test.
    """
    device = next(model.parameters()).device
    perm = build_batches(ws.source, ws.win_ids, ws.eval_rows,
                         shuffle_sites=True, seed=args.seed)
    pm = loss.evaluate(model, [b.to(device) for b in perm], args.backend,
                       hap_chunk, ref_cache)
    stage = f"after {args.epochs} epoch(s)" if args.epochs else "zero-shot"
    intact = best.get("eval_bits", float("nan"))
    print(f"\n  LD-destroyed control  {pm['all_bits']:.4f} bits/SNP  "
          f"(vs {intact:.4f} intact, same model, {stage})")
    return {**pm, "vs_intact": intact, "stage": stage}


# ── Output ────────────────────────────────────────────────────────────────────

def resolve_output(args) -> tuple[Path, Path]:
    """(checkpoint path, run dir). Auto-named under $SVAR_SCRATCH/runs/ by default."""
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out, out.parent
    model_tag = ("baselines_" + args.eval_accessions
                 if getattr(args, "baselines_only", False)
                 else f"{args.backend}_{args.precision}_{args.eval_accessions}")
    rd = artifacts.run_dir(args.dataset, "variant_ll", model_tag,
                           run_id=f"hw{args.half_window}_w{args.windows}"
                                  f"_e{args.epochs}_lr{args.lr:g}_s{args.seed}",
                           root=args.out_root)
    return rd / "model.pt", rd


def finalize(args, model, results, base, ws, out_path: Path, run_dir: Path) -> None:
    best = results.get("best", {})
    perm = results.get("permuted")

    if args.save_model:
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                    "results": {k: v for k, v in results.items() if k != "history"}},
                   out_path)
        print(f"  saved final weights → {out_path}")
        bp = out_path.with_name("model_best.pt")
        if bp.exists():
            print(f"  best weights (eval {best.get('eval_bits', float('nan')):.4f}) "
                  f"→ {bp}")

    artifacts.save_json(run_dir / "results.json", results)

    # metrics.mean is what run_record._flatten lifts into the central manifest, so
    # the headline bits and both baselines land in one comparable row per run.
    mean = {"bits_per_snp": best.get("eval_bits"),
            "bits_up": best.get("eval_up"), "bits_noup": best.get("eval_noup"),
            "b1_bits": base["b1_all"], "b2_bits": base["b2_all"],
            "b1_bits_up": base["b1_up"], "b2_bits_up": base["b2_up"],
            "margin_vs_b1": (best.get("eval_bits", float("nan")) - base["b1_all"]),
            "zeroshot_bits": results.get("zeroshot", {}).get("eval_bits")}
    if "train_bits" in best:
        mean["train_bits"] = best["train_bits"]
        mean["gap"] = best.get("gap")
    if perm:
        mean["permuted_bits"] = perm["all_bits"]

    phase = "test" if (args.eval_split == "test" and
                       args.eval_accessions == "heldout") else "val"
    rec = run_record.build(
        dataset=args.dataset, features="variant_ll",
        model=(f"carbon_{args.backend}_{args.precision}"
               + ("_lora" if getattr(args, "arch", "full") == "lora" else "")
               + ("_base" if getattr(args, "train_base", False) else "")),
        seed=args.seed, traits=[],
        hyperparams={"half_window": args.half_window, "windows": args.windows,
                     "chrom": args.chrom, "epochs": args.epochs, "lr": args.lr,
                     "schedule": args.schedule, "warmup": args.warmup,
                     "weight_decay": args.weight_decay,
                     "hap_chunk": args.hap_chunk,
                     "ckpt_reference": args.ckpt_reference,
                     "eval_accessions": args.eval_accessions},
        metrics={phase: {"mean": mean}},
        half_window=args.half_window,
        split_path=str(ws.split_path) if ws.split_path else None,
        strict=args.strict,
        extra={"n_sites": results["n_sites"], "n_windows": results["n_windows"],
               "upstream_frac": results["upstream_frac"],
               "site_stats": results["site_stats"],
               "backend": args.backend, "precision": args.precision,
               "eval_accessions": args.eval_accessions,
               "eval_split": args.eval_split})
    run_record.write(rec, run_dir)
    print(f"\n[{args.dataset}/variant_ll/{args.backend}] run_id={rec.run_id}"
          f"  eval {best.get('eval_bits', float('nan')):.4f} bits/SNP"
          f"  (B1 {base['b1_all']:.4f})\n  → {run_dir}")

"""
Go/no-go benchmark: is fine-tuning through the variant cache tractable at
*expression-prediction* scale?

The variant_ll numbers (BENCHMARK.md §8) were all measured at cs≈8. A cis
expression window is the other regime: T up to 8192 tokens (±24 kb around the
TSS, Carbon's position ceiling) and the FULL SNP set cached — at 1001G density
that is cs in the hundreds to low thousands. The cross branch saves an
(h, N·cs, T) probability tensor per layer for backward, so grad memory scales
N·cs·T and the variant_ll envelope says nothing about whether this fits.

What this measures, per (T, cs, N) grid point, in the frozen-base LoRA recipe
(reference under no_grad — the default expression training path):

  * reference forward time (no grad) — paid once per gene per pass
  * one training step through the variant branch: forward + mean-pool over the
    recomputed positions + linear head + MSE + backward, with
    `encoder.variant_checkpointing` off and on
  * peak CUDA memory for each; OOMs are caught and reported as such

Plus a correctness gate for the new flag: adapter/head gradients with
checkpointing on vs off must agree (small config, fp32).

Run (idle A100 recommended; ~10 min for the default grid):

    python -m model_dev.bench_expression_cache
    python -m model_dev.bench_expression_cache --grid-t 8192 --grid-cs 256,1024,2448 --grid-n 8,32,128
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

DNA_LO, DNA_N = 151672, 4096  # 6-mer block (verified in variant_ll/BENCHMARK.md)


def _synth_window(T: int, cs: int, N: int, device) -> dict:
    g = torch.Generator(device="cpu").manual_seed(0)
    ref_ids = DNA_LO + torch.randint(DNA_N, (T,), generator=g)
    # sorted unique variant positions, avoiding position 0 (the <dna> marker slot)
    pos = torch.randperm(T - 1, generator=g)[:cs].sort().values + 1
    hap_ids = DNA_LO + torch.randint(DNA_N, (N, cs), generator=g)
    return dict(ref_ids=ref_ids.to(device), pos=pos.to(device),
                hap_ids=hap_ids.to(device))


def _one_step(model, head, w, ref_layers, use_ckpt: bool) -> torch.Tensor:
    model.encoder.variant_checkpointing = use_ckpt
    out = model(w["ref_ids"], variant_positions=w["pos"],
                variant_input_ids=w["hap_ids"], output_logits=False,
                reference_layer_inputs=ref_layers)
    pooled = out.last_hidden_state.float().mean(dim=1)     # (N, hidden)
    pred = head(pooled).squeeze(-1)                        # (N,)
    loss = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()
    return loss


def _timed_step(model, head, w, ref_layers, use_ckpt: bool, reps: int = 3):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(reps + 1):  # rep 0 is warmup
        for p in model.parameters():
            p.grad = None
        for p in head.parameters():
            p.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _one_step(model, head, w, ref_layers, use_ckpt)
        torch.cuda.synchronize()
        if i > 0:
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    return sorted(times)[len(times) // 2], peak


def grad_check(model, head, device) -> float:
    """Max relative difference of trainable-parameter grads, ckpt on vs off."""
    w = _synth_window(T=512, cs=32, N=4, device=device)
    ref_layers = model.encode_reference(w["ref_ids"])
    grads = {}
    for use_ckpt in (False, True):
        for p in list(model.parameters()) + list(head.parameters()):
            p.grad = None
        _one_step(model, head, w, ref_layers, use_ckpt)
        grads[use_ckpt] = [p.grad.clone() for p in
                           list(model.trainable_parameters()) + list(head.parameters())
                           if p.grad is not None]
    worst = 0.0
    for a, b in zip(grads[False], grads[True]):
        denom = a.abs().max().item()
        if denom > 0:
            worst = max(worst, (a - b).abs().max().item() / denom)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-t", default="3334,8192")
    ap.add_argument("--grid-cs", default="64,256,1024,2448")
    ap.add_argument("--grid-n", default="8,32,128")
    ap.add_argument("--base-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--skip-no-ckpt-above-cs", type=int, default=256,
                    help="don't even attempt ckpt-off above this cs (known OOM)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from CARBON_modules import load_carbon_variant_lora

    device = "cuda"
    base_dtype = torch.bfloat16 if args.base_dtype == "bf16" else torch.float32
    model, _ = load_carbon_variant_lora(device=device, base_dtype=base_dtype)
    hidden = model.config.hidden_size
    head = torch.nn.Linear(hidden, 1).to(device=device, dtype=torch.float32)

    # correctness gate for the new variant_checkpointing flag (fp32 to make the
    # comparison about the mechanism, not bf16 reduction order)
    m32, _ = load_carbon_variant_lora(device=device, base_dtype=torch.float32)
    err = grad_check(m32, torch.nn.Linear(hidden, 1).to(device), device)
    print(f"[gate] variant_checkpointing grad agreement: max rel err = {err:.3e}")
    del m32
    torch.cuda.empty_cache()
    if err > 1e-5:
        print("FAIL: checkpointed gradients disagree — do not trust the timings.")
        return 1

    rows = []
    for T in [int(x) for x in args.grid_t.split(",")]:
        # reference forward, no grad — once per gene per pass in this recipe
        w0 = _synth_window(T, 8, 1, device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ref_layers = model.encode_reference(w0["ref_ids"])
        torch.cuda.synchronize()
        t_ref = time.perf_counter() - t0
        ref_gib = sum(t.numel() * t.element_size() for t in ref_layers) / 2**30
        print(f"\nT={T}: reference forward {t_ref*1e3:.0f} ms (no grad), "
              f"stream {ref_gib:.2f} GiB resident")
        del ref_layers

        for cs in [int(x) for x in args.grid_cs.split(",")]:
            if cs >= T:
                continue
            w = _synth_window(T, cs, max(int(x) for x in args.grid_n.split(",")),
                              device)
            ref_layers = model.encode_reference(w["ref_ids"])
            for N in [int(x) for x in args.grid_n.split(",")]:
                wN = dict(w, hap_ids=w["hap_ids"][:N])
                for use_ckpt in (False, True):
                    if not use_ckpt and cs > args.skip_no_ckpt_above_cs:
                        rows.append(dict(T=T, cs=cs, N=N, ckpt=use_ckpt,
                                         status="skipped (known OOM regime)"))
                        continue
                    try:
                        t_step, peak = _timed_step(model, head, wN, ref_layers,
                                                   use_ckpt)
                        rows.append(dict(T=T, cs=cs, N=N, ckpt=use_ckpt,
                                         status="ok", step_ms=t_step * 1e3,
                                         peak_gib=peak))
                        print(f"  cs={cs:5d} N={N:4d} ckpt={int(use_ckpt)}: "
                              f"{t_step*1e3:7.0f} ms/step  peak {peak:6.1f} GiB")
                    except torch.cuda.OutOfMemoryError:
                        rows.append(dict(T=T, cs=cs, N=N, ckpt=use_ckpt,
                                         status="OOM"))
                        print(f"  cs={cs:5d} N={N:4d} ckpt={int(use_ckpt)}: OOM")
                        torch.cuda.empty_cache()
            del ref_layers
            torch.cuda.empty_cache()

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(dict(base_dtype=args.base_dtype, grad_check_rel_err=err,
                           rows=rows), fh, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

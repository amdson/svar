"""
training/sweeps/variant_ll.py
-----------------------------
The allele log-likelihood benchmark's standing experiment matrix, as one grid.

Every number quoted in BENCHMARK.md §8c/§8d comes from a block below, so a claim
in that file is reproducible with one command rather than a remembered
invocation. Run:

    python -m training.sweep --config training/sweeps/variant_ll.py --list
    python -m training.sweep --config training/sweeps/variant_ll.py --gpus 0,2

The driver is resumable — completed points are skipped — and every point appends
one row to ``$SVAR_SCRATCH/runs/index.jsonl``, so afterwards:

    from training.common.run_record import load_runs
    df = load_runs()
    df[df.features == "variant_ll"][["model", "val.bits_per_snp", "val.b1_bits",
                                     "val.margin_vs_b1", "val.gap"]]

Arabidopsis chr4 throughout (PLAN.md: smallest chromosome, effectively N-free).
"""

DATA = {"dataset": "arabidopsis", "chrom": 4, "half_window": 500, "seed": 42}

# ── 1. Precision. The single most consequential setting, and the least obvious ──
# bf16 converges to ~the baseline and looks like a legitimate null result; fp32
# goes well below it. In-sample (20 windows, 75 passes each) so this is pure
# fitting capacity with no generalisation term to confound it. BENCHMARK.md §8c.
precision = {
    "runner": "variant_ll",
    "name": "precision",
    "gpu": True,
    "fixed": {**DATA, "windows": 20, "epochs": 75, "lr": "1e-4",
              "eval_accessions": "insample", "warmup": 50, "hap_chunk": 32,
              "accum_windows": 4},
    "grid": {"precision": ["fp32", "bf16"]},
}

# ── 2. What the frozen-reference approximation costs ──────────────────────────
# Identical recipe, cache vs the exact full forward it approximates. The gap is
# BENCHMARK.md §7's control. The exact backend needs a smaller chunk: it runs a
# full T-token forward per haplotype with no checkpointing.
backend_cache = {
    "runner": "variant_ll",
    "name": "backend_cache",
    "gpu": True,
    "fixed": {**DATA, "windows": 20, "epochs": 75, "lr": "1e-4",
              "precision": "fp32", "eval_accessions": "insample",
              "warmup": 50, "backend": "cache", "hap_chunk": 32,
              "accum_windows": 4},
    "grid": {},
}
backend_exact = {
    "runner": "variant_ll",
    "name": "backend_exact",
    "gpu": True,
    "fixed": {**DATA, "windows": 20, "epochs": 75, "lr": "1e-4",
              "precision": "fp32", "eval_accessions": "insample",
              "warmup": 50, "backend": "exact", "hap_chunk": 8,
              "accum_windows": 4},
    "grid": {},
}

# ── 3. The benchmark itself — held-out accessions ─────────────────────────────
# Site count is the binding constraint: at a fixed ~16k step budget, 200 windows
# reaches the baseline and 400 does not. Sweeping windows x epochs at constant
# budget separates "needs more passes" from "needs fewer sites".
heldout = {
    "runner": "variant_ll",
    "name": "heldout",
    "gpu": True,
    "fixed": {**DATA, "precision": "fp32", "lr": "1e-4",
              "eval_accessions": "heldout", "eval_split": "val",
              "hap_chunk": 32, "permute": True, "eval_every": 250,
              "accum_windows": 8},
    "grid": {"windows": [200, 400], "epochs": [40, 80]},
}

# ── 4. Learning rate, on the cheap in-sample setting ──────────────────────────
lr_probe = {
    "runner": "variant_ll",
    "name": "lr",
    "gpu": True,
    "fixed": {**DATA, "windows": 20, "epochs": 75, "precision": "fp32",
              "eval_accessions": "insample", "warmup": 50,
              "accum_windows": 4},
    "grid": {"lr": ["3e-5", "1e-4", "3e-4"]},
}

# ── 5. Negative control (BENCHMARK.md §7) ─────────────────────────────────────
# At hw=500 on ARRAY data (rice) almost every window holds one SNP, so nearly
# every site lacks upstream context and the expected result is "ties B1". If it
# does not tie, suspect a bug.
rice_null = {
    "runner": "variant_ll",
    "name": "rice_null",
    "gpu": True,
    "fixed": {"dataset": "rice", "chrom": 1, "half_window": 500, "seed": 42,
              "windows": 200, "epochs": 40, "lr": "1e-4", "precision": "fp32",
              "eval_accessions": "heldout", "permute": True, "hap_chunk": 32,
              "accum_windows": 8},
    "grid": {},
}

# Start with the cheap in-sample blocks; they answer the mechanism questions in
# minutes. The held-out block is hours per point.
SWEEP = [precision, backend_cache, backend_exact, lr_probe]

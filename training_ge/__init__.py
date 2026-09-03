"""
training_ge — gene expression prediction through the Carbon variant cache.

Distinct from `training/` (trait prediction): the unit here is a (gene, line)
pair, the target is a per-gene-normalized expression deviation, and the model
is the dual-stream LoRA variant cache — frozen bf16 base serving both the
reference stream and the variant recompute, fp32 adapters on the variant
stream only, a small fp32 head on the pooled (mutant − reference) hidden-state
deltas at the mutated positions.

First dataset: SIEVE (Brachypodium sodium-azide mutant lines). Sanity ladder
and data-level gates: model_dev/sieve_signal_gate.py (gate 0, passed: ~11%
learnable ceiling) and model_dev/sieve_dumb_baseline.py (pooled-feature floor:
R² ≈ 0.002).

  data.py  -> SieveWindowSource: per-gene batches (reference window + each
              line's mutations patched in + z targets)
  run.py   -> fine-tune + evaluate CLI (gates 1-3)
"""

"""
variant_ll
----------
Allele log-likelihood benchmark for the Carbon variant cache: can a model
fine-tuned through the cache predict which allele a held-out individual carries
at each SNP, better than the naive per-site allele-frequency baseline?

See BENCHMARK.md for the objective and PLAN.md for the run sequence.

  run.py / harness.py -> the runner; every experiment is this one loop with
                         different flags (see run.py's docstring)
  data.WindowSource   -> per-window batches (frozen reference + haplotypes)
  baselines.evaluate  -> B0 / B1 / B2 in bits per SNP
  loss.window_bits    -> the objective, cache or exact backend
  loss.evaluate       -> bits/SNP sliced by upstream context

The scored unit is a *token*, not a site: a 6-mer holding m segregating sites is
scored jointly over its 2**m possible values, which keeps the sum an exact
chain-rule factorization of P(haplotype | reference). See BENCHMARK.md §5.
"""
from variant_ll.data import WindowBatch, WindowSource

__all__ = ["WindowSource", "WindowBatch"]

"""
variant_ll
----------
Allele log-likelihood benchmark for the Carbon variant cache: can a model
fine-tuned through the cache predict which allele a held-out individual carries
at each SNP, better than the naive per-site allele-frequency baseline?

See BENCHMARK.md for the objective and PLAN.md for the run sequence.

  data.WindowSource   -> per-window batches (frozen reference + haplotypes)
  baselines.evaluate  -> B0 / B1 / B2 in bits per SNP
  loss.window_bits    -> the objective, cache or exact backend
  loss.evaluate       -> bits/SNP sliced by upstream context
"""
from variant_ll.data import WindowBatch, WindowSource, genotype_split

__all__ = ["WindowSource", "WindowBatch", "genotype_split"]

"""
pretrained_eval
----------------
Pretrained-checkpoint evaluation harness: autoregressive log-likelihood /
perplexity of a Carbon causal LM on the rice dataset, reported both over entire
sequences (every DNA token in a window) and over individual SNP tokens.

  carbon_ar_nll -> per-batch summed NLL + token counts (seq + SNP, one forward)
  evaluate      -> token-weighted mean NLL + perplexity over a window loader

CLI: ``python pretrained_eval/eval.py --limit 50``
"""
from .loss import carbon_ar_nll, collect_per_window, evaluate

__all__ = ["carbon_ar_nll", "collect_per_window", "evaluate"]

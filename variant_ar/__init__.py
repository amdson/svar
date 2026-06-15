"""
variant_ar
----------
Autoregressive SNP-token log-likelihood loss for Carbon — a testing harness for
the variant-cache work. Feeds variant haplotype windows (VCF + reference) through
the causal LM and scores next-token prediction restricted to the SNP tokens.

  snp_ar_nll          -> (sum_nll, n_tokens) for a batch of windows
  evaluate            -> token-weighted mean NLL / perplexity over a window loader

Variant-aware objective used for training/eval through the variant cache
(next-token prediction from each SNP position; see train.py):

  snp_next_token_nll  -> (sum_nll, n_tokens) for one ref/alt window
  evaluate_next_token -> token-weighted mean NLL / perplexity over a ref/alt loader
"""
from .loss import (evaluate, evaluate_next_token, snp_ar_nll,
                   snp_next_token_nll)

__all__ = ["snp_ar_nll", "evaluate", "snp_next_token_nll",
           "evaluate_next_token"]

"""
CARBON_modules
--------------
Loaders for the Carbon genome language model (HuggingFaceBio/Carbon-*, a
decoder-only causal LM over DNA). Mirrors the surface of `DNABERT2_modules` and
`PlantCAD_modules`.

  load_carbon       -> (AutoModelForCausalLM, tokenizer)  for embeddings
  load_carbon_lm    -> (AutoModelForCausalLM, tokenizer)  for causal-LM scoring
  load_carbon_local -> (CarbonForCausalLM, tokenizer)     local, hackable copy

`carbon_layers.py` is a self-contained re-implementation of Carbon's Llama
backbone (analogue of `DNABERT2_modules/bert_layers.py`); use `load_carbon_local`
to materialize the checkpoint in it.
"""
from .loader import (load_carbon, load_carbon_lm, load_carbon_local,
                     load_carbon_variant_cache)
from .carbon_layers import CarbonForCausalLM, CarbonModel
from .variant_cache_layers import (VariantCacheCarbonEncoder,
                                   VariantCacheCarbonForCausalLM)

__all__ = ["load_carbon", "load_carbon_lm", "load_carbon_local",
           "load_carbon_variant_cache", "CarbonForCausalLM", "CarbonModel",
           "VariantCacheCarbonEncoder", "VariantCacheCarbonForCausalLM"]

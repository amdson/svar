"""
CARBON_modules
--------------
Loaders for the Carbon genome language model (HuggingFaceBio/Carbon-*, a
decoder-only causal LM over DNA). Mirrors the surface of `DNABERT2_modules` and
`PlantCAD_modules`.

  load_carbon    -> (AutoModelForCausalLM, tokenizer)  for embeddings
  load_carbon_lm -> (AutoModelForCausalLM, tokenizer)  for causal-LM scoring
"""
from .loader import load_carbon, load_carbon_lm

__all__ = ["load_carbon", "load_carbon_lm"]

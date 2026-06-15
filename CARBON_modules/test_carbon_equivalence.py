"""Numerical equivalence check: the local `CarbonForCausalLM` vs the stock
HuggingFace `AutoModelForCausalLM` (Llama) that Carbon really ships as.

Both load the *same* checkpoint; the local copy vendors the Llama math as a
flat, hackable stack (see `carbon_layers.py`). They must therefore produce the
same logits and hidden states up to fp32 eager-attention numerics. Run:

    python -m CARBON_modules.test_carbon_equivalence

Downloads the 500M checkpoint on first run. Compares on CPU in fp32 with
`attn_implementation="eager"` on the reference so the two go through identical
attention math. Exits 0 on success, 1 on mismatch.
"""

import sys

import torch

from .loader import load_carbon_local


def main() -> int:
    repo_id = "HuggingFaceBio/Carbon-500M"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading local CarbonForCausalLM on {device} ({dtype})...")
    local_model, tokenizer = load_carbon_local(repo_id, device=device, dtype=dtype)

    print("Loading reference AutoModelForCausalLM (eager)...")
    from transformers import AutoModelForCausalLM
    ref_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    # A couple of short DNA windows of differing length, so the padding mask is
    # exercised. The Carbon tokenizer routes DNA via a literal "<dna>" marker and
    # needs add_special_tokens=False (see CARBON_modules/loader.py docstring).
    seqs = ["<dna>" + "ACGTACGTACGTACGTACGT",
            "<dna>" + "TTGGCCAATTGGCCAA"]
    enc = tokenizer(seqs, return_tensors="pt", padding=True,
                    add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out_local = local_model(input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        out_ref = ref_model(input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

    mask = attention_mask.bool().unsqueeze(-1)

    def _masked_max_diff(a, b):
        diff = (a - b).abs()
        diff = diff[mask.expand_as(diff)]
        return diff.max().item(), diff.mean().item()

    logit_max, logit_mean = _masked_max_diff(out_local.logits, out_ref.logits)
    hs_max, hs_mean = _masked_max_diff(out_local.hidden_states[-1],
                                       out_ref.hidden_states[-1])
    ref_scale = out_ref.logits[mask.expand_as(out_ref.logits)].abs().mean().item()

    print(f"logits : max_abs_diff={logit_max:.4e}  mean_abs_diff={logit_mean:.4e}")
    print(f"hidden : max_abs_diff={hs_max:.4e}  mean_abs_diff={hs_mean:.4e}")
    print(f"ref_logit_scale={ref_scale:.4e}")
    print(f"hidden_states tuple len: local={len(out_local.hidden_states)} "
          f"ref={len(out_ref.hidden_states)}")

    tol = 1e-3 * max(ref_scale, 1.0)
    if logit_max > tol or hs_max > 1e-3:
        print(f"FAIL: difference exceeds tolerance (logit tol={tol:.4e}).")
        return 1
    print("PASS: local Carbon matches the HuggingFace reference within fp32 tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

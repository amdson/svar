"""
Testing framework: Carbon variant-cache forward vs the normal full forward.

Three checks, in increasing strlength of claim:

  1. efficient vs bruteforce
     The log-sum-exp merge (`VariantCacheCarbonEncoder.forward`) must equal the
     full-sequence oracle (`forward_bruteforce`) up to fp32 numerics, for the
     *same* substituted variants. Validates the cross/self attention split.

  2. identity probe: variants == reference  (cache vs normal forward)
     When the substituted tokens equal the reference tokens, freezing the
     non-variant positions changes nothing, so the cache must reproduce the
     normal `CarbonForCausalLM` forward *exactly* at the variant positions.
     Validates RoPE-at-parametrized-positions, causal masking, and the
     pre-norm/GQA layer plumbing — independent of any approximation.

  3. approximation error: real substitutions  (cache vs normal forward)
     With actual allele substitutions the cache is an *approximation* (it freezes
     the downstream context). This is the scientific quantity of interest, so we
     only report cosine similarity / relative error rather than asserting it's
     tiny.

Run (downloads the 500M checkpoint on first use; CPU fp32 is fine):

    python -m model_dev.test_carbon_variant_cache
"""

import sys

import torch


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main() -> int:
    from CARBON_modules import (load_carbon_local, load_carbon_variant_cache)

    repo_id = "HuggingFaceBio/Carbon-500M"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading models on {device} ({dtype})...")
    full_model, tokenizer = load_carbon_local(repo_id, device=device, dtype=dtype)
    vc_model, _ = load_carbon_variant_cache(repo_id, device=device, dtype=dtype,
                                            backend="efficient")
    vc_model_bf, _ = load_carbon_variant_cache(repo_id, device=device,
                                               dtype=dtype, backend="bruteforce")

    # One reference window. DNA routes through the tokenizer's "<dna>" marker
    # with add_special_tokens=False (see CARBON_modules/loader.py).
    seq = "<dna>" + "ACGTACGTTTGGCCAA" * 12  # ~32 tokens after 6-mer packing
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)  # (1, ref)
    ref_len = input_ids.shape[1]

    # Variant positions spread across the window. Position 0 is included on
    # purpose: its cross branch (reference keys at positions < 0) is empty, which
    # exercises the finfo.min / log-sum-exp NaN-guard for an all-masked branch.
    variant_positions = torch.tensor(
        [0, ref_len // 3, 2 * ref_len // 3, ref_len - 1], device=device)
    print(f"ref_len={ref_len}  variant_positions={variant_positions.tolist()}")

    with torch.no_grad():
        # --- Check 1: efficient vs bruteforce (identity-substituted) ---
        out_eff = vc_model(input_ids, variant_positions)
        out_bf = vc_model_bf(input_ids, variant_positions)
        d1 = (out_eff.last_hidden_state - out_bf.last_hidden_state).abs().max().item()
        print(f"\n[1] efficient vs bruteforce : max_abs_diff={d1:.3e}  "
              f"cos={_cos(out_eff.last_hidden_state, out_bf.last_hidden_state):.8f}")

        # --- Check 2: identity probe vs normal full forward ---
        full = full_model(input_ids, output_hidden_states=True)
        full_hidden = full.hidden_states[-1][0]                # (ref, hidden)
        full_at_var = full_hidden.index_select(0, variant_positions)  # (cs, hidden)
        vc_at_var = out_eff.last_hidden_state[0]               # (cs, hidden)
        d2 = (full_at_var - vc_at_var).abs().max().item()
        scale = full_at_var.abs().mean().item()
        print(f"[2] identity probe vs full  : max_abs_diff={d2:.3e}  "
              f"hidden_scale={scale:.3e}  cos={_cos(full_at_var, vc_at_var):.8f}")

        # logits too
        full_logits_at_var = full.logits[0].index_select(0, variant_positions)
        d2l = (full_logits_at_var - out_eff.logits[0]).abs().max().item()
        print(f"    identity probe logits   : max_abs_diff={d2l:.3e}")

        # --- Check 3: real substitutions (report approximation, don't assert) ---
        # Substitute a different 6-mer DNA token at each variant position.
        dna_lo = 151669  # dna_start_id from dna_config.json
        sub_ids = input_ids[0].index_select(0, variant_positions).clone()
        sub_ids = dna_lo + ((sub_ids - dna_lo + 7) % 4096)  # nudge to another DNA token
        out_sub = vc_model(input_ids, variant_positions, variant_input_ids=sub_ids)

        # True forward of the actually-substituted sequence.
        sub_seq_ids = input_ids.clone()
        sub_seq_ids[0, variant_positions] = sub_ids
        full_sub = full_model(sub_seq_ids, output_hidden_states=True)
        full_sub_at_var = full_sub.hidden_states[-1][0].index_select(
            0, variant_positions)
        d3 = (full_sub_at_var - out_sub.last_hidden_state[0]).abs().max().item()
        print(f"[3] real-substitution approx: max_abs_diff={d3:.3e}  "
              f"cos={_cos(full_sub_at_var, out_sub.last_hidden_state[0]):.6f}  "
              f"(approximation — informational)")

    tol = 1e-3 * max(scale, 1.0)
    ok = (d1 < 1e-3) and (d2 < tol) and (d2l < 1e-2 * max(
        full_logits_at_var.abs().mean().item(), 1.0))
    if not ok:
        print("\nFAIL: efficient/bruteforce or identity probe exceeded tolerance.")
        return 1
    print("\nPASS: variant cache reproduces the normal forward (identity) and "
          "efficient==bruteforce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

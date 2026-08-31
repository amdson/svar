"""Checks for `VariantLoRACarbonForCausalLM` — the properties its design claims.

    python -m CARBON_modules.test_variant_lora

The class exists to make two things true by construction rather than by
convention, and both are cheap to falsify, so they are asserted here:

  1. **The adapters are variant-only.** Randomising every adapter must not move
     `encode_reference` by a single bit. If it does, the reference stream is no
     longer a constant of the window and `ReferenceCache` is silently serving
     stale activations.
  2. **A fresh model is exactly the base model.** `lora_B` is zero-initialised,
     so the LoRA class must reproduce `VariantCacheCarbonForCausalLM` bit for
     bit before training. Otherwise a zero-shot number is not comparable across
     the two classes and every later delta is confounded.

Plus the things that would quietly produce a wrong number: that the reference
cache is exact, that only adapters get gradients, that the stale-cache guard
actually fires, and that the bruteforce backend is refused rather than silently
scoring the base model. Exits 0 on success, 1 on any failure.
"""

import sys

import torch

from .loader import load_carbon_variant_cache, load_carbon_variant_lora
from .variant_lora_layers import ALL_TARGETS, ReferenceCache


def _fixture(device, vocab=155776, ref_len=48, n_hap=6, n_var=5, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ref_ids = torch.randint(151672, 155767, (ref_len,), generator=g).to(device)
    var_pos = torch.sort(torch.randperm(ref_len - 2, generator=g)[:n_var] + 1).values.to(device)
    hap_ids = torch.randint(151672, 155767, (n_hap, n_var), generator=g).to(device)
    return ref_ids, var_pos, hap_ids


def _randomise_adapters(model, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = torch.randn(p.shape, generator=g).to(p.device, p.dtype) * 0.05


def main() -> int:
    repo = "HuggingFaceBio/Carbon-500M"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print(f"Loading both classes on {device} ({dtype}) …")
    # Adapt every projection, k/v included, so the test exercises the widest
    # surface — a variant-only violation is most likely to show up in k/v, which
    # are the projections that also run on the reference stream.
    lora, _ = load_carbon_variant_lora(repo, device=device, base_dtype=dtype,
                                       r=8, alpha=16.0, targets=ALL_TARGETS)
    cache, _ = load_carbon_variant_cache(repo, device=device, dtype=dtype,
                                         backend="efficient")
    ref_ids, var_pos, hap_ids = _fixture(device)

    print("\n1. a fresh model is exactly the base model")
    with torch.no_grad():
        a = lora(ref_ids, variant_positions=var_pos, variant_input_ids=hap_ids).logits
        b = cache(ref_ids, variant_positions=var_pos, variant_input_ids=hap_ids).logits
    check("zero-init LoRA == variant cache", torch.equal(a, b),
          f"max|d| = {(a - b).abs().max().item():.3e}")

    print("\n2. adapters are variant-only")
    with torch.no_grad():
        ref_before = lora.encode_reference(ref_ids)
    _randomise_adapters(lora)
    with torch.no_grad():
        ref_after = lora.encode_reference(ref_ids)
        after = lora(ref_ids, variant_positions=var_pos,
                     variant_input_ids=hap_ids).logits
    check("encode_reference unmoved by adapters",
          all(torch.equal(x, y) for x, y in zip(ref_before, ref_after)))
    check("variant logits DO move (adapters are wired up)",
          not torch.allclose(after, a),
          f"max|d| = {(after - a).abs().max().item():.3e}")

    print("\n3. ReferenceCache is exact")
    rc = ReferenceCache(lora)
    with torch.no_grad():
        cached = lora(ref_ids, variant_positions=var_pos, variant_input_ids=hap_ids,
                      reference_layer_inputs=rc.get("w0", ref_ids)).logits
        again = lora(ref_ids, variant_positions=var_pos, variant_input_ids=hap_ids,
                     reference_layer_inputs=rc.get("w0", ref_ids)).logits
    check("cached reference == recomputed", torch.equal(cached, after),
          f"max|d| = {(cached - after).abs().max().item():.3e}")
    check("second get is a hit", rc.stats()["hits"] == 1, str(rc.stats()))
    check("repeat use is stable", torch.equal(cached, again))

    print("\n4. only adapters train")
    summ = lora.parameter_summary()
    names = {n for n, p in lora.named_parameters() if p.requires_grad}
    check("every trainable param is a lora param",
          bool(names) and all("lora_" in n for n in names),
          f"{summ['trainable']:,}/{summ['total']:,} = {100*summ['fraction']:.2f}%")
    lora.zero_grad(set_to_none=True)
    lora(ref_ids, variant_positions=var_pos,
         variant_input_ids=hap_ids).logits.square().mean().backward()
    got = [n for n, p in lora.named_parameters() if p.grad is not None]
    check("gradients reach the adapters",
          bool(got) and all("lora_" in n for n in got),
          f"{len(got)} tensors with grad")
    check("lora params are fp32 under an fp32 base",
          all(p.dtype == torch.float32
              for n, p in lora.named_parameters() if "lora_" in n))

    print("\n5. guards")
    with torch.no_grad():
        lora.encoder.layers[0].self_attn.q_proj.weight.add_(1.0)
    try:
        rc.get("w1", ref_ids)
        check("stale-cache guard fires when base moves", False)
    except RuntimeError as e:
        check("stale-cache guard fires when base moves", "stale" in str(e))
    try:
        lora.set_backend("bruteforce")
        check("bruteforce backend refused", False)
    except ValueError:
        check("bruteforce backend refused", True)

    print("\n6. bf16 base keeps fp32 adapters")
    lora_bf, _ = load_carbon_variant_lora(repo, device=device,
                                          base_dtype=torch.bfloat16, r=8)
    dts = {("lora" if "lora_" in n else "base"): p.dtype
           for n, p in lora_bf.named_parameters()}
    check("base bf16, adapters fp32",
          dts.get("base") == torch.bfloat16 and dts.get("lora") == torch.float32,
          str(dts))

    print("\n7. trainable base (--train-base)")
    tb, _ = load_carbon_variant_lora(repo, device=device, base_dtype=dtype,
                                     r=8, freeze_base=False)
    names = {n for n, p in tb.named_parameters() if p.requires_grad}
    check("base and adapters both trainable",
          any("lora_" in n for n in names) and any("lora_" not in n for n in names),
          f"{tb.parameter_summary()['trainable']:,} trainable")
    tb.zero_grad(set_to_none=True)
    tb(ref_ids, variant_positions=var_pos,
       variant_input_ids=hap_ids).logits.square().mean().backward()
    grads = {n for n, p in tb.named_parameters() if p.grad is not None}
    check("gradients reach the base too",
          any("lora_" not in n for n in grads),
          f"{len(grads)} tensors with grad")
    # The point of the mode: the reference stream is no longer a constant, which
    # is exactly what a frozen base bought. Gradient must reach a base weight
    # that is used ONLY on the reference stream path plus the shared projections.
    q0 = tb.encoder.layers[0].self_attn.q_proj.weight
    check("shared base weight gets gradient", q0.grad is not None
          and bool((q0.grad != 0).any()))
    try:
        ReferenceCache(tb)
        check("ReferenceCache refuses a trainable base", False)
    except ValueError as e:
        check("ReferenceCache refuses a trainable base", "frozen base" in str(e))
    try:
        tb.cast_base(torch.bfloat16)
        check("bf16 refused while base is trainable", False)
    except ValueError as e:
        check("bf16 refused while base is trainable", "rounded away" in str(e))
    check("checkpoint saves the full model when base trains",
          len(tb.checkpoint_state_dict()) > len(tb.lora_state_dict()),
          f"{len(tb.checkpoint_state_dict())} vs {len(tb.lora_state_dict())} tensors")
    lora.freeze_base(True)
    check("checkpoint saves adapters only when base is frozen",
          set(lora.checkpoint_state_dict()) == set(lora.lora_state_dict()))

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

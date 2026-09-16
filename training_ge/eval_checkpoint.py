"""
Re-evaluate a saved training_ge checkpoint (LoRA + head) on ALL val
accessions for the same genes it trained on — inference only. Exists to
shrink the SE of a cache number that was trained/evaluated with
--max-lines subsampling (e.g. n=1,971 val pairs -> ~10k), so it can be
compared on identical pairs to the elastic-net controls.

    python -m training_ge.eval_checkpoint --ckpt $SVAR_SCRATCH/runs/ge_ath_double_resid.pt \
        --kinship-residual --enet-residual
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--hw", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kinship-residual", action="store_true")
    ap.add_argument("--enet-residual", action="store_true")
    ap.add_argument("--rows-per-call", type=int, default=64)
    ap.add_argument("--splits", default="val", help="comma list of acc splits to score")
    ap.add_argument("--replicate-subsample", type=int, default=None,
                    help="reproduce the TRAINING run's val subsample: set "
                         "max_lines to this and replay its rng sequence (one "
                         "throwaway build pass for train batches, then the val "
                         "pass). Must return the training-time val number.")
    ap.add_argument("--z-cache", default=None,
                    help=".npy to load/save the (residualized) z_all so enet #1 "
                         "is not refit on every evaluation")
    args = ap.parse_args()

    from CARBON_modules import load_carbon_variant_lora
    from training_ge.ath_data import ArabidopsisWindowSource

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    device = "cuda"
    model, tok = load_carbon_variant_lora(
        device=device, base_dtype=torch.bfloat16, r=a["lora_r"],
        alpha=a["lora_alpha"], dropout=a["lora_dropout"])
    missing, unexpected = model.load_state_dict(ck["lora"], strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if "lora_" in k], "adapter weights missing"
    model.eval()
    model.encoder.variant_checkpointing = False
    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    head.load_state_dict(ck["head"])
    print(f"checkpoint epoch {ck['epoch']} (saved val {ck['val']})")

    import os
    src = ArabidopsisWindowSource(
        tok, half_window=args.hw,
        max_lines=args.replicate_subsample or 700, seed=args.seed,
        kinship_residual=args.kinship_residual)
    gene_ix = src.sample_genes(args.n_genes, split="train")  # same set as training
    if args.enet_residual:
        if args.z_cache and os.path.exists(args.z_cache):
            src.z_all = np.load(args.z_cache)
            print(f"loaded residualized z from {args.z_cache}")
        else:
            src.subtract_enet(gene_ix)
            if args.z_cache:
                np.save(args.z_cache, src.z_all)
    if args.replicate_subsample:
        # run.py built train batches first (one rng.choice per gene), then
        # val batches; replay the first pass so the val subsample matches.
        for gi in gene_ix:
            src.build(int(gi))
    want = {s: set(src.eco[src.acc_split == s]) for s in args.splits.split(",")}

    preds = {s: [] for s in want}
    targs = {s: [] for s in want}
    with torch.no_grad():
        for gi in gene_ix:
            b = src.build(int(gi))
            if b is None:
                continue
            rows = [i for i, l in enumerate(b.lines)
                    if any(l in w for w in want.values())]
            if not rows:
                continue
            rows_t = torch.tensor(rows)
            outs = []
            for chunk in rows_t.split(args.rows_per_call):
                hap = torch.cat([b.hap_ids[:1], b.hap_ids[1:][chunk]]).to(device)
                out = model(b.ref_ids.to(device),
                            variant_positions=b.cache_idx.to(device),
                            variant_input_ids=hap, output_logits=False)
                h = out.last_hidden_state.float()
                delta = h[1:] - h[0:1]
                m = b.own_mask[chunk].to(device).unsqueeze(-1).float()
                outs.append((delta * m).sum(1) / m.sum(1).clamp(min=1))
            p = head(torch.cat(outs)).squeeze(-1).cpu().numpy()
            for k, i in enumerate(rows):
                for s, w in want.items():
                    if b.lines[i] in w:
                        preds[s].append(p[k]); targs[s].append(float(b.z[i]))
    for s in want:
        P, T = np.array(preds[s]), np.array(targs[s])
        r = np.corrcoef(P, T)[0, 1]
        se = 1 / np.sqrt(len(T))
        print(f"[{s}] n={len(T):,}  pooled pearson={r:+.4f}  (SE~{se:.3f}, "
              f"{r/se:.1f} sigma)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

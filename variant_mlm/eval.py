"""
variant_mlm/eval.py
-------------------
Evaluate a DNABERT2 model's *variant* MLM loss: over every unique window that
contains an alt allele, mask the variant token(s) and measure how well the model
predicts them. Reports mean cross-entropy, perplexity, and top-1 accuracy.

This is the benign-ness diagnostic for the variant-cache approximation: run it
on the exact model to get the baseline, and (once the cache MLM model is wired
up) on the cache model — if the two losses match, the approximation is benign
for the quantity the model actually computes.

Masking is deterministic here (every variant token is masked with [MASK], no
random/keep corruption) so the number is reproducible.

Example
-------
    python variant_mlm/eval.py --half-window 500 --max-length 2048 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH
from DNABERT2_modules import load_dnabert2_mlm, load_dnabert2_variant_cache_mlm
from variant_mlm.data import (
    VariantMLMCollator,
    build_variant_window_dataset,
    make_loader,
)

DEFAULT_VCF_PATH = str(Path(__file__).resolve().parents[1] / "sativas413_msu7_final.vcf")

parser = argparse.ArgumentParser(description="Evaluate variant-MLM loss for DNABERT2.")
parser.add_argument("--model-path", type=str, default="zhihan1996/DNABERT-2-117M",
                    help="HF repo or local dir for the MLM model (encoder + head).")
parser.add_argument("--backend", choices=["exact", "cache", "cache-bruteforce"],
                    default="exact",
                    help="exact: standard BertForMaskedLM. cache: variant-cache "
                         "encoder (efficient LSE). cache-bruteforce: variant-cache "
                         "oracle (full-sequence, clamps context). The cache backends "
                         "measure variant-MLM loss through the approximation.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Optional train.py state dict to load on top of --model-path.")
parser.add_argument("--max-length", type=int, default=2048)
parser.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
parser.add_argument("--fasta-path", type=str, default=FASTA_PATH)
parser.add_argument("--half-window", type=int, default=500,
                    help="Half window size; full window is 2*half_window bp.")
parser.add_argument("--buffer", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--attn-impl", choices=["flash", "triton", "torch"], default=None,
                    help="Override the model's attention impl. Default (flash) needs "
                         "CUDA + flash-attn; use 'torch' to run on CPU / without it.")
parser.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N variant windows (smoke test).")
parser.add_argument("--device", type=str, default=None)
args = parser.parse_args()

device = torch.device(
    args.device if args.device is not None
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Device: {device}  Model: {args.model_path}  half_window={args.half_window}")

# ── Data ──────────────────────────────────────────────────────────────────────
print("Building variant window dataset …")
dataset, indices = build_variant_window_dataset(
    args.vcf_path, args.fasta_path, half_window=args.half_window, buffer=args.buffer
)
if args.limit is not None:
    indices = indices[: args.limit]
print(f"  {len(indices):,} variant windows (of {len(dataset.unique_fingerprints):,} unique)")

# ── Model ─────────────────────────────────────────────────────────────────────
overrides = {"attn_impl": args.attn_impl} if args.attn_impl is not None else None
if args.backend == "exact":
    model, tokenizer = load_dnabert2_mlm(args.model_path, device=device,
                                         config_overrides=overrides)
else:
    cache_backend = "bruteforce" if args.backend == "cache-bruteforce" else "efficient"
    model, tokenizer = load_dnabert2_variant_cache_mlm(
        args.model_path, device=device, backend=cache_backend,
        config_overrides=overrides)
print(f"Backend: {args.backend}")
if args.checkpoint is not None:
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    print(f"  loaded checkpoint weights from {args.checkpoint}")
model.eval()

# Deterministic, full masking: every variant token -> [MASK].
collator = VariantMLMCollator(
    tokenizer, max_length=args.max_length,
    select_prob=1.0, mask_replace_prob=1.0, random_replace_prob=0.0,
)
loader = make_loader(dataset, indices, args.batch_size, shuffle=False)

# ── Eval loop ─────────────────────────────────────────────────────────────────
total_loss = 0.0
total_correct = 0
total_targets = 0
with torch.no_grad():
    for batch in tqdm(loader, desc="variant MLM", unit="batch"):
        enc = collator(batch)
        if enc["n_targets"] == 0:
            continue
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = enc["labels"].to(device)

        out = model(input_ids, attention_mask=attention_mask, labels=labels,
                    return_dict=True)

        tmask = labels > 0
        n = int(tmask.sum())
        preds = out.logits.argmax(dim=-1)
        total_correct += int((preds[tmask] == labels[tmask]).sum())
        total_loss += float(out.loss) * n  # loss is mean over this batch's targets
        total_targets += n

if total_targets == 0:
    raise SystemExit("No variant tokens were evaluated — check windowing / data paths.")

mean_loss = total_loss / total_targets
ppl = float(torch.exp(torch.tensor(mean_loss)))
acc = total_correct / total_targets
print("\n── variant MLM ──")
print(f"  windows         : {len(indices):,}")
print(f"  masked tokens   : {total_targets:,}")
print(f"  cross-entropy   : {mean_loss:.4f}")
print(f"  perplexity      : {ppl:.4f}")
print(f"  top-1 accuracy  : {acc:.4f}")

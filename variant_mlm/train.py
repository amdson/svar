"""
variant_mlm/train.py
--------------------
Fine-tune a DNABERT2 model on the *variant* MLM objective: mask the BPE tokens
overlapping alt positions and train the model to predict them. Use this to adapt
the model to the variant-prediction task before downstream ML (and, once the
cache MLM model exists, to make the model "aware" of the cache approximation by
training through it).

Windows come from the embed_windows machinery with ``--half-window`` as a
parameter; only windows containing an alt allele are used. Train/val is a
deterministic split over those windows.

Shared-weight caution (from the design discussion): training the full backbone
on variant tokens alone can degrade the general encoder the frozen reference
relies on. ``--freeze-backbone`` trains only the MLM head as a cheap, drift-free
option; full fine-tuning is the default for the standalone (non-cache) model.

Example
-------
    python variant_mlm/train.py --output checkpoints/variant_mlm/dnabert2.pt \\
        --half-window 500 --epochs 3 --batch-size 16 --lr 5e-5
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH
from crop_embed.logging_utils import MetricLogger, metrics_path_for
from DNABERT2_modules import load_dnabert2_mlm, load_dnabert2_variant_cache_mlm
from variant_mlm.data import (
    VariantMLMCollator,
    build_variant_window_dataset,
    make_loader,
    split_indices,
)

DEFAULT_VCF_PATH = str(Path(__file__).resolve().parents[1] / "sativas413_msu7_final.vcf")

parser = argparse.ArgumentParser(description="Fine-tune DNABERT2 on variant-MLM loss.")
# Model
parser.add_argument("--model-path", type=str, default="zhihan1996/DNABERT-2-117M")
parser.add_argument("--max-length", type=int, default=2048)
parser.add_argument("--backend", choices=["exact", "cache", "cache-bruteforce"],
                    default="exact",
                    help="exact: standard BertForMaskedLM. cache / cache-bruteforce: "
                         "train through the variant-cache approximation.")
parser.add_argument("--freeze-backbone", action="store_true",
                    help="Train only the MLM head; keeps the encoder fixed so the "
                         "frozen-reference encoder doesn't drift.")
parser.add_argument("--attn-impl", choices=["flash", "triton", "torch"], default=None,
                    help="Override the model's attention impl. Default (flash) needs "
                         "CUDA + flash-attn; use 'torch' to run on CPU / without it.")
# Data / windowing
parser.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
parser.add_argument("--fasta-path", type=str, default=FASTA_PATH)
parser.add_argument("--half-window", type=int, default=500,
                    help="Half window size; full window is 2*half_window bp.")
parser.add_argument("--buffer", type=int, default=0)
parser.add_argument("--val-frac", type=float, default=0.1)
parser.add_argument("--limit", type=int, default=None,
                    help="Use only the first N variant windows (smoke test / dev).")
# Masking
parser.add_argument("--select-prob", type=float, default=1.0,
                    help="Probability each variant token is a prediction target.")
parser.add_argument("--mask-replace-prob", type=float, default=0.8)
parser.add_argument("--random-replace-prob", type=float, default=0.1)
# Optimization
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
parser.add_argument("--log-every", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
# wandb / IO
parser.add_argument("--wandb-project", type=str, default=None)
parser.add_argument("--wandb-name", type=str, default=None)
parser.add_argument("--output", type=str, required=True,
                    help="Path to write the fine-tuned model.pt.")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_amp_dtype = {"bf16": torch.bfloat16}.get(args.precision)


def autocast_ctx():
    if _amp_dtype is None:
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_amp_dtype)


# ── Data ──────────────────────────────────────────────────────────────────────
print(f"Device: {device}  half_window={args.half_window}")
print("Building variant window dataset …")
dataset, indices = build_variant_window_dataset(
    args.vcf_path, args.fasta_path, half_window=args.half_window, buffer=args.buffer
)
if args.limit is not None:
    indices = indices[: args.limit]
train_idx, val_idx = split_indices(indices, args.val_frac, args.seed)
print(f"  {len(indices):,} variant windows  |  {len(train_idx):,} train / {len(val_idx):,} val")

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

if args.freeze_backbone:
    # exact model nests the encoder under `.bert`; the cache model exposes
    # `.embeddings` + `.encoder` directly. Freeze everything but the MLM head.
    backbone = (model.bert.parameters() if args.backend == "exact"
                else (*model.embeddings.parameters(), *model.encoder.parameters()))
    for p in backbone:
        p.requires_grad_(False)

train_params = [p for p in model.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in train_params)
print(f"Trainable params: {n_train:,}"
      + ("  (MLM head only)" if args.freeze_backbone else "  (full model)"))

opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

# Masking generator is seeded so corruption is reproducible across runs.
mask_gen = torch.Generator().manual_seed(args.seed)
train_collator = VariantMLMCollator(
    tokenizer, max_length=args.max_length, select_prob=args.select_prob,
    mask_replace_prob=args.mask_replace_prob,
    random_replace_prob=args.random_replace_prob, generator=mask_gen,
)
# Validation: deterministic full masking (matches eval.py).
val_collator = VariantMLMCollator(
    tokenizer, max_length=args.max_length,
    select_prob=1.0, mask_replace_prob=1.0, random_replace_prob=0.0,
)

train_loader = make_loader(dataset, train_idx, args.batch_size, shuffle=True)
val_loader = make_loader(dataset, val_idx, args.batch_size, shuffle=False)

logger = MetricLogger(
    metrics_path_for(args.output),
    wandb_project=args.wandb_project, wandb_name=args.wandb_name, config=vars(args),
)
print(f"Logging metrics to {logger.metrics_path}")


@torch.no_grad()
def evaluate() -> dict:
    model.eval()
    total_loss, total_correct, total_targets = 0.0, 0, 0
    for batch in val_loader:
        enc = val_collator(batch)
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
        total_loss += float(out.loss) * n
        total_targets += n
    mean = total_loss / max(total_targets, 1)
    return {
        "val/loss": mean,
        "val/ppl": float(torch.exp(torch.tensor(mean))),
        "val/acc": total_correct / max(total_targets, 1),
    }


# ── Train ─────────────────────────────────────────────────────────────────────
best_val = float("inf")
step = 0
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"epoch {epoch}", unit="batch"):
        enc = train_collator(batch)
        if enc["n_targets"] == 0:
            continue
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = enc["labels"].to(device)

        opt.zero_grad()
        with autocast_ctx():
            out = model(input_ids, attention_mask=attention_mask, labels=labels,
                        return_dict=True)
        out.loss.backward()
        opt.step()

        if step % args.log_every == 0:
            logger.log({"epoch": epoch, "step": step, "train/loss": float(out.loss)})
            print(f"epoch {epoch} step {step}  train_loss={float(out.loss):.4f}")
        step += 1

    val_m = evaluate()
    print(f"epoch {epoch}  [val] loss={val_m['val/loss']:.4f}  "
          f"ppl={val_m['val/ppl']:.3f}  acc={val_m['val/acc']:.4f}")
    logger.log({"epoch": epoch, "step": step, **val_m})

    if val_m["val/loss"] < best_val:
        best_val = val_m["val/loss"]
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_path": args.model_path,
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_m,
        }, out_path)
        print(f"  saved best model (val_loss={best_val:.4f}) -> {out_path}")

logger.close()
print(f"\nDone. Best val loss: {best_val:.4f}")

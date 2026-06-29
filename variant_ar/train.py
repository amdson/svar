"""
variant_ar/train.py
-------------------
Fine-tune a Carbon variant-cache model on the autoregressive SNP objective:
freeze the reference-genome window, substitute the alt alleles at the SNP
positions, recompute just those positions through the variant cache, and
maximize the likelihood of the actual next token at each SNP position. Evaluate
the same loss on a held-out window split.

Objective rationale (see `variant_ar/loss.py`): in a causal LM the SNP only
influences positions >= itself, so the variant-aware signal is next-token
prediction *from* each SNP position — which is exactly what the cache emits
(logits at the recomputed variant positions). Training through the cache adapts
the model so its frozen-reference approximation is good for the quantity it
actually computes.

``--backend`` picks the model:
  exact           : full CarbonForCausalLM (ground-truth forward of the alt seq).
  cache           : variant cache, efficient log-sum-exp merge (the real target).
  cache-bruteforce: variant cache, full-sequence oracle (slow; for validation).
All three are scored on the identical objective, so their losses are comparable
(exact = the number the cache is approximating).

Train/val is a deterministic split over the variant windows (same `split_indices`
the variant-MLM trainer uses); the train_pipeline splits are sample/phenotype
level and don't apply to this per-window objective.

Example
-------
    python variant_ar/train.py --output checkpoints/variant_ar/carbon.pt \\
        --backend cache --half-window 500 --epochs 3 --batch-size 4 --lr 1e-5
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH
from crop_embed import MetricLogger, metrics_path_for
from CARBON_modules import load_carbon_local, load_carbon_variant_cache
from variant_mlm.data import build_variant_window_dataset, split_indices
from variant_ar.loss import evaluate_next_token, snp_next_token_nll

DEFAULT_VCF_PATH = str(Path(__file__).resolve().parents[1] / "sativas413_msu7_final.vcf")


# ── Ref/alt window data ───────────────────────────────────────────────────────
class RefAltWindows(Dataset):
    """Yield both the reference-genome window (no alts) and the variant haplotype
    (alts substituted) for each variant window, by calling the base dataset's
    `extract_sequence` with and without the fingerprint's alt set."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        fp = self.base.unique_fingerprints[self.indices[i]]
        chrom, start, end, _ = fp
        return {
            "ref_sequence": self.base.extract_sequence((chrom, start, end, frozenset())),
            "alt_sequence": self.base.extract_sequence(fp),
            "fingerprint": fp,
        }


def _collate(items):
    return {
        "ref_sequences": [x["ref_sequence"] for x in items],
        "alt_sequences": [x["alt_sequence"] for x in items],
        "fingerprints": [x["fingerprint"] for x in items],
    }


def make_ar_loader(base, indices, batch_size, shuffle):
    # num_workers=0: the base dataset holds pyfaidx FASTA handles that aren't
    # fork-safe (same constraint as the embedder / variant-MLM loaders).
    return DataLoader(RefAltWindows(base, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, collate_fn=_collate)


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune Carbon variant cache on the AR SNP objective.")
    p.add_argument("--model-path", type=str, default="HuggingFaceBio/Carbon-500M")
    p.add_argument("--backend", choices=["exact", "cache", "cache-bruteforce"],
                   default="cache")
    p.add_argument("--max-length", type=int, default=2048)
    # Data / windowing
    p.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
    p.add_argument("--fasta-path", type=str, default=str(FASTA_PATH))
    p.add_argument("--half-window", type=int, default=500,
                   help="Half window size; full window is 2*half_window bp.")
    p.add_argument("--buffer", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N variant windows (smoke test / dev).")
    # Optimization
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Windows per optimizer step. Keep modest for the cache "
                        "backends — each window holds a full reference forward graph.")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, required=True,
                   help="Path to write the best fine-tuned model.pt.")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="If set, also mirror metrics to this wandb project.")
    p.add_argument("--wandb-name", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None

    def autocast_ctx():
        if amp_dtype is None:
            from contextlib import nullcontext
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    # ── Metric logging (local JSONL sidecar next to --output; optional wandb) ───
    # Mirrors train_pipeline/train_head.py so AR runs are tracked identically.
    # Log-likelihood on SNP tokens is just the negation of the next-token NLL.
    logger = MetricLogger(
        metrics_path_for(args.output),
        wandb_project=args.wandb_project, wandb_name=args.wandb_name,
        config=vars(args),
    )
    print(f"Logging metrics to {logger.metrics_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Device: {device}  backend={args.backend}  half_window={args.half_window}")
    print("Building variant window dataset …")
    dataset, indices = build_variant_window_dataset(
        args.vcf_path, args.fasta_path, half_window=args.half_window, buffer=args.buffer)
    if args.limit is not None:
        indices = indices[: args.limit]
    train_idx, val_idx = split_indices(indices, args.val_frac, args.seed)
    print(f"  {len(indices):,} variant windows  |  "
          f"{len(train_idx):,} train / {len(val_idx):,} val")

    train_loader = make_ar_loader(dataset, train_idx, args.batch_size, shuffle=True)
    val_loader = make_ar_loader(dataset, val_idx, args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    if args.backend == "exact":
        model, tokenizer = load_carbon_local(args.model_path, device=device, dtype=dtype)
    else:
        cache_backend = "bruteforce" if args.backend == "cache-bruteforce" else "efficient"
        model, tokenizer = load_carbon_variant_cache(
            args.model_path, device=device, dtype=dtype, backend=cache_backend)
    n_params = sum(pm.numel() for pm in model.parameters() if pm.requires_grad)
    print(f"Trainable params: {n_params:,} (full model)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # ── Train ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", unit="batch"):
            opt.zero_grad()
            losses, n_tok = [], 0
            for rs, asq, fp in zip(batch["ref_sequences"], batch["alt_sequences"],
                                   batch["fingerprints"]):
                with autocast_ctx():
                    nll, n = snp_next_token_nll(model, tokenizer, rs, asq, fp,
                                                args.backend, args.max_length, device)
                if n > 0:
                    losses.append(nll)
                    n_tok += n
            if n_tok == 0:
                continue
            loss = torch.stack(losses).sum() / n_tok  # token-weighted mean
            loss.backward()
            opt.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}  train_loss={loss.item():.4f}")
                logger.log({"epoch": epoch, "step": step,
                            "train/nll": loss.item(), "train/ll": -loss.item()})
            step += 1

        val = evaluate_next_token(model, tokenizer, val_loader, backend=args.backend,
                                  max_length=args.max_length, device=device)
        print(f"epoch {epoch}  [val] nll={val['mean_nll']:.4f}  "
              f"ppl={val['perplexity']:.3f}  (n_tokens={val['n_tokens']:,})")
        logger.log({"epoch": epoch, "step": step,
                    "val/nll": val["mean_nll"], "val/ll": -val["mean_nll"],
                    "val/perplexity": val["perplexity"], "val/n_tokens": val["n_tokens"]})

        if val["mean_nll"] < best_val:
            best_val = val["mean_nll"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_path": args.model_path,
                "backend": args.backend,
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val,
            }, out_path)
            print(f"  saved best model (val_nll={best_val:.4f}) -> {out_path}")

    print(f"\nDone. Best val nll: {best_val:.4f}")
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
training_ge/run.py — fine-tune the dual-stream LoRA variant cache on SIEVE
expression deviations.

The sanity ladder (gates from the go/no-go discussion):

  gate 1  in-sample smoke: --n-genes 100 --val-frac 0. Train loss must drop
          meaningfully below predict-zero. Catches wiring/precision bugs.
  gate 2  held-out lines at seen genes: --holdout lines
  gate 3  held-out gene families: --holdout family (train on 'train' families,
          evaluate on 'val'). Compare against the dumb-baseline floor
          (R² ≈ 0.002) and the gate-0 ceiling (~11%).
  control --permute shuffles z across lines within each gene; generalization
          must collapse.

Model: frozen bf16 Carbon base, fp32 LoRA on the variant stream (regularized:
small r, dropout, weight decay), linear fp32 head on the mean over the line's
own mutated positions of (mutant − reference) final hidden states. Row 0 of
every batch is the identity haplotype, so the reference representation comes
through the same code path bit-for-bit.

    python -m training_ge.run --n-genes 200 --val-frac 0.2 \
        --output $SVAR_SCRATCH/runs/ge_sieve_gate1.pt
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

# direct submodule import: the crop_embed package __init__ imports polars,
# which this env doesn't have (vcf_polars is the data agent's new path)
from crop_embed.logging_utils import MetricLogger, metrics_path_for


def pooled_delta(model, batch, device) -> torch.Tensor:
    """(N, hidden) fp32: mean over own-mutation positions of (mutant − ref)."""
    out = model(batch.ref_ids.to(device),
                variant_positions=batch.cache_idx.to(device),
                variant_input_ids=batch.hap_ids.to(device),
                output_logits=False)
    h = out.last_hidden_state.float()          # (N+1, C, H)
    delta = h[1:] - h[0:1]                     # row 0 = reference
    m = batch.own_mask.to(device).unsqueeze(-1).float()
    return (delta * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def evaluate(model, head, batches, device, tag, logger=None, step=None):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in batches:
            p = head(pooled_delta(model, batch, device)).squeeze(-1)
            preds.append(p.cpu())
            targs.append(batch.z)
    model.train()
    if not preds:
        return {}
    p = torch.cat(preds).numpy()
    t = torch.cat(targs).numpy()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = 1 - ((t - p) ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 else 0.0
    row = {f"{tag}/r2": float(r2), f"{tag}/pearson": r,
           f"{tag}/n_pairs": int(len(t))}
    if logger is not None:
        logger.log({"step": step, **row})
    print(f"  [{tag}] n={len(t)}  R2={r2:+.4f}  pearson={r:+.4f}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--hw", type=int, default=4000)
    ap.add_argument("--max-lines", type=int, default=64)
    ap.add_argument("--max-ac", type=int, default=5,
                    help="drop SNVs shared by more lines than this (stock "
                         "heterogeneity, not induced mutations)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--accum-genes", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of sampled genes held out (holdout=genes)")
    ap.add_argument("--holdout", choices=["genes", "lines", "family"],
                    default="genes")
    ap.add_argument("--permute", action="store_true",
                    help="shuffle z across lines within each gene (control)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    from CARBON_modules import load_carbon_variant_lora
    from training_ge.data import SieveWindowSource

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, tokenizer = load_carbon_variant_lora(
        device=device, base_dtype=torch.bfloat16, r=args.lora_r,
        alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.train()
    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)

    source = SieveWindowSource(tokenizer, half_window=args.hw,
                               max_lines=args.max_lines, seed=args.seed,
                               max_ac=args.max_ac)
    rng = np.random.default_rng(args.seed)

    if args.holdout == "family":
        train_ix = source.sample_genes(args.n_genes, split="train")
        # cap val: ~1.5k genes ≈ 16k pairs -> pearson SE ~0.008, plenty
        val_ix = source.sample_genes(min(max(args.n_genes // 4, 50), 1500),
                                     split="val")
    else:
        ix = source.sample_genes(args.n_genes)
        n_val = int(len(ix) * args.val_frac)
        val_ix, train_ix = ix[:n_val], ix[n_val:]
        # holdout=lines: same genes both sides; lines partitioned inside the loop
        if args.holdout == "lines":
            train_ix = val_ix = ix
    print(f"{len(train_ix)} train genes, {len(val_ix)} val genes "
          f"(holdout={args.holdout}, hw={args.hw}, permute={args.permute})")

    line_split = None
    if args.holdout == "lines":
        lines = sorted(source.snv)
        test = set(rng.choice(lines, size=len(lines) // 5, replace=False))
        line_split = test

    def filter_batch(batch, want_val: bool):
        if line_split is None:
            return batch
        keep = [i for i, l in enumerate(batch.lines)
                if (l in line_split) == want_val]
        if not keep:
            return None
        k = torch.tensor(keep)
        batch.hap_ids = torch.cat([batch.hap_ids[:1], batch.hap_ids[1:][k]])
        batch.own_mask, batch.z = batch.own_mask[k], batch.z[k]
        batch.lines = [batch.lines[i] for i in keep]
        return batch

    # Materialize batches ONCE — a GeneBatch is ~11 KB, so even the full train
    # family split (~16k genes) is a few hundred MB of host RAM, while
    # rebuilding (fasta fetch + tokenize) every epoch dominated early runs'
    # wall clock. Line filtering and the permutation control are applied here,
    # so --permute is a *fixed* mislabeled dataset, the cleanest control.
    import time
    t0 = time.perf_counter()
    train_batches = [b for b in (filter_batch(x, want_val=False)
                                 for x in source.iter_batches(train_ix)) if b]
    val_batches = [b for b in (filter_batch(x, want_val=True)
                               for x in source.iter_batches(val_ix)) if b] \
        if len(val_ix) else []
    if args.permute:
        g = torch.Generator().manual_seed(args.seed)
        for b in train_batches:
            if len(b.z) > 1:
                b.z = b.z[torch.randperm(len(b.z), generator=g)]
    n_tr = sum(len(b.z) for b in train_batches)
    n_va = sum(len(b.z) for b in val_batches)
    print(f"built {len(train_batches)} train batches ({n_tr:,} pairs), "
          f"{len(val_batches)} val batches ({n_va:,} pairs) in "
          f"{time.perf_counter() - t0:.0f}s; skips {source.skip_counts}")

    params = [{"params": model.trainable_parameters(), "lr": args.lr},
              {"params": head.parameters(), "lr": args.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    n_train = sum(p.numel() for g in params for p in g["params"])
    print(f"trainable parameters: {n_train:,}")

    logger = MetricLogger(metrics_path_for(args.output),
                          wandb_project=args.wandb_project, config=vars(args))
    step = 0
    best_val = -math.inf
    zero_r2_note = "predict-zero baseline is R2=0 by construction"
    print(zero_r2_note)
    train_probe = train_batches[:max(len(val_batches), 40)]
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_batches))
        losses = []
        for g0 in range(0, len(order), args.accum_genes):
            group = [train_batches[i] for i in order[g0:g0 + args.accum_genes]]
            group_n = sum(len(b.z) for b in group)
            opt.zero_grad(set_to_none=True)
            for b in group:
                pred = head(pooled_delta(model, b, device)).squeeze(-1)
                loss = ((pred - b.z.to(device)) ** 2).sum() / group_n
                loss.backward()
                losses.append(loss.item())
            opt.step()
            step += 1
            if step % 20 == 0:
                logger.log({"step": step, "epoch": epoch,
                            "train/mse": float(np.sum(losses[-len(group):]))})
        evaluate(model, head, train_probe, device, "train", logger, step)
        if val_batches:
            val_row = evaluate(model, head, val_batches, device, "val",
                               logger, step)
            key = "val/pearson"
            if val_row.get(key, -math.inf) > best_val:
                best_val = val_row[key]
                torch.save({"lora": model.checkpoint_state_dict(),
                            "head": head.state_dict(), "args": vars(args),
                            "epoch": epoch, "val": val_row}, args.output)
        print(f"epoch {epoch}: mean group loss "
              f"{np.mean(losses) * args.accum_genes:.4f}")
    logger.close()
    print(f"done; best {best_val:+.4f} (val pearson) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

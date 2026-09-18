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

HAP_CHUNK = None  # set from --hap-chunk in main()


def _delta_rows(model, batch, device, rows: torch.Tensor) -> torch.Tensor:
    """pooled (mutant − ref) delta for a subset of haplotype rows (row 0 = ref
    is always prepended so every chunk carries its own reference baseline)."""
    hap = torch.cat([batch.hap_ids[:1], batch.hap_ids[1:][rows]]).to(device)
    out = model(batch.ref_ids.to(device),
                variant_positions=batch.cache_idx.to(device),
                variant_input_ids=hap, output_logits=False)
    h = out.last_hidden_state.float()          # (n+1, C, H)
    delta = h[1:] - h[0:1]                     # row 0 = reference
    m = batch.own_mask[rows].to(device).unsqueeze(-1).float()
    return (delta * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def pooled_delta(model, batch, device, hap_chunk=None, no_grad=False) -> torch.Tensor:
    """(N, hidden) fp32: mean over own-mutation positions of (mutant − ref).
    hap_chunk splits the N rows into chunks (memory at exact-forward cs=T);
    no_grad runs the encoder without a graph (head-only training)."""
    N = batch.hap_ids.shape[0] - 1
    idx = torch.arange(N)
    chunks = [idx] if not hap_chunk else idx.split(hap_chunk)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    outs = []
    with ctx:
        for rows in chunks:
            outs.append(_delta_rows(model, batch, device, rows))
    return torch.cat(outs)


def to_exact(batch):
    """Recompute EVERY window position through the cache: cache_idx = all of
    0..T-1, so the cross branch is fully masked and the self branch is a
    plain causal full forward of each haplotype's complete sequence — the
    bruteforce/exact-forward control, sharing the LoRA, readout and target
    code with the cache runs. Cost scales with rows x T² (no amortization)."""
    T = batch.ref_ids.shape[0]
    R = batch.hap_ids.shape[0]
    full = batch.ref_ids.unsqueeze(0).repeat(R, 1)
    full[:, batch.cache_idx] = batch.hap_ids
    own = torch.zeros(R - 1, T, dtype=torch.bool)
    own[:, batch.cache_idx] = batch.own_mask
    batch.hap_ids, batch.own_mask = full, own
    batch.cache_idx = torch.arange(T)
    return batch


def _sample_rows(batch, k: int, rng):
    """A shallow copy of `batch` holding a random k of its haplotype rows
    (row 0, the reference, is kept). Fresh draw per gene visit."""
    import copy
    N = batch.hap_ids.shape[0] - 1
    if N <= k:
        return batch
    keep = torch.from_numpy(np.sort(rng.choice(N, k, replace=False)))
    b = copy.copy(batch)
    b.hap_ids = torch.cat([batch.hap_ids[:1], batch.hap_ids[1:][keep]])
    b.own_mask, b.z = batch.own_mask[keep], batch.z[keep]
    b.lines = [batch.lines[i] for i in keep.tolist()]
    return b


class GeneHead(torch.nn.Module):
    """Linear readout with an optional per-gene ridge component.

    pred = feat . (w0 + w_g) + b0 + b_g.  w_g / b_g live in embedding tables
    indexed by gene id and start at zero, so an unseen gene (family holdout,
    or index -1) falls back to the shared head exactly. This is the fair
    counterpart to a PrediXcan-style per-gene elastic net: the same per-gene
    capacity, but over cache features instead of genotypes. Decay w_g
    separately (--gene-wd): 1024 weights per gene vs ~530 train accessions.
    """

    def __init__(self, hidden: int, gene_ids=None):
        super().__init__()
        self.shared = torch.nn.Linear(hidden, 1)
        self.gene_index = {g: i for i, g in enumerate(gene_ids or [])}
        n = max(len(self.gene_index), 1)
        self.w_g = torch.nn.Embedding(n, hidden)
        self.b_g = torch.nn.Embedding(n, 1)
        torch.nn.init.zeros_(self.w_g.weight)
        torch.nn.init.zeros_(self.b_g.weight)

    def forward(self, feat, gene_id=None):
        out = self.shared(feat)
        i = self.gene_index.get(gene_id, -1) if self.gene_index else -1
        if i >= 0:
            idx = torch.tensor(i, device=feat.device)
            out = out + feat @ self.w_g(idx).unsqueeze(-1) + self.b_g(idx)
        return out

    def gene_parameters(self):
        return [self.w_g.weight, self.b_g.weight]


def evaluate(model, head, batches, device, tag, logger=None, step=None):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in batches:
            p = head(pooled_delta(model, batch, device, HAP_CHUNK),
                     batch.gene_id).squeeze(-1)
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
    ap.add_argument("--dataset", choices=["sieve", "ath"], default="sieve")
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--hw", type=int, default=4000)
    ap.add_argument("--max-lines", type=int, default=None,
                    help="rows per gene batch (default: 64 sieve, 700 ath)")
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
    ap.add_argument("--holdout", choices=["genes", "lines", "family",
                                          "accessions"],
                    default="genes",
                    help="'accessions' (ath only): train rows = acc_split "
                         "train, val rows = acc_split val, same genes")
    ap.add_argument("--split-key", default="acc_split",
                    choices=["acc_split", "kin_split"],
                    help="ath accession partition: random (acc_split) or "
                         "admixture-group holdout (kin_split)")
    ap.add_argument("--kinship-residual", action="store_true",
                    help="ath only: train/eval on z minus the train-fitted "
                         "GBLUP prediction (relatedness-orthogonal target)")
    ap.add_argument("--enet-residual", action="store_true",
                    help="ath only: additionally subtract a per-gene cis "
                         "elastic net (double residual — signal beyond both "
                         "relatedness and linear cis effects)")
    ap.add_argument("--exact", action="store_true",
                    help="exact-forward control: recompute every window "
                         "position (cache_idx = all), i.e. a full forward per "
                         "haplotype; cost ~ rows x T^2")
    ap.add_argument("--hap-chunk", type=int, default=None,
                    help="rows per encoder call (default: 16 with --exact)")
    ap.add_argument("--rows-per-step", type=int, default=None,
                    help="stochastic minibatching over accessions: each gene "
                         "visit uses a fresh random subset of this many rows "
                         "(all rows across epochs; eval always scores all). "
                         "Unlike --hap-chunk this changes the objective's "
                         "sampling, not its memory staging.")
    ap.add_argument("--per-gene-head", action="store_true",
                    help="add a per-gene weight vector + bias to the linear "
                         "head (GeneHead); unseen genes use the shared head")
    ap.add_argument("--gene-wd", type=float, default=1.0,
                    help="AdamW weight decay on the per-gene head tables")
    ap.add_argument("--head-only", action="store_true",
                    help="freeze the adapters: pretrained Carbon features + "
                         "fitted head only (the zero-shot-style row)")
    ap.add_argument("--variant-ckpt", action="store_true",
                    help="checkpoint the variant branch (needed at ath-scale "
                         "cs; on by default for --dataset ath)")
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

    if args.dataset == "ath":
        from training_ge.ath_data import ArabidopsisWindowSource
        source = ArabidopsisWindowSource(
            tokenizer, half_window=args.hw, max_lines=args.max_lines or 700,
            seed=args.seed, kinship_residual=args.kinship_residual,
            split_key=args.split_key)
        model.encoder.variant_checkpointing = True  # cs ~100+ per window
    else:
        source = SieveWindowSource(tokenizer, half_window=args.hw,
                                   max_lines=args.max_lines or 64,
                                   seed=args.seed, max_ac=args.max_ac)
        model.encoder.variant_checkpointing = args.variant_ckpt
    rng = np.random.default_rng(args.seed)

    if args.holdout == "accessions":
        if args.dataset != "ath":
            raise SystemExit("--holdout accessions requires --dataset ath")
        # same genes both sides; rows partitioned by the committed acc_split
        train_ix = val_ix = source.sample_genes(args.n_genes, split="train")
    elif args.holdout == "family":
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
    if args.enet_residual:
        if args.dataset != "ath":
            raise SystemExit("--enet-residual requires --dataset ath")
        source.subtract_enet(np.unique(np.concatenate([train_ix, val_ix])))
    print(f"{len(train_ix)} train genes, {len(val_ix)} val genes "
          f"(holdout={args.holdout}, hw={args.hw}, permute={args.permute})")

    line_split = None
    if args.holdout == "lines":
        lines = sorted(source.snv)
        test = set(rng.choice(lines, size=len(lines) // 5, replace=False))
        line_split = test
    excluded = set()
    if args.holdout == "accessions":
        # committed split: val rows held out, test rows excluded entirely
        line_split = set(source.eco[source.acc_split == "val"])
        excluded = set(source.eco[np.isin(source.acc_split,
                                          ["test", "excluded"])])

    def filter_batch(batch, want_val: bool):
        if line_split is None:
            return batch
        keep = [i for i, l in enumerate(batch.lines)
                if (l in line_split) == want_val and l not in excluded]
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
    global HAP_CHUNK
    HAP_CHUNK = args.hap_chunk or (16 if args.exact else None)
    if args.exact:
        for b in train_batches + val_batches:
            to_exact(b)
        model.encoder.variant_checkpointing = True
        print(f"--exact: cache_idx = all {train_batches[0].ref_ids.shape[0]} "
              f"positions; hap_chunk={HAP_CHUNK}")
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

    gene_ids = sorted({b.gene_id for b in train_batches}) \
        if args.per_gene_head else None
    head = GeneHead(model.config.hidden_size, gene_ids).to(device)
    params = [{"params": head.shared.parameters(), "lr": args.head_lr}]
    if args.per_gene_head:
        params.append({"params": head.gene_parameters(), "lr": args.head_lr,
                       "weight_decay": args.gene_wd})
        print(f"per-gene head over {len(gene_ids)} genes (gene-wd {args.gene_wd})")
    if not args.head_only:
        params.insert(0, {"params": model.trainable_parameters(), "lr": args.lr})
    else:
        for p_ in model.trainable_parameters():
            p_.requires_grad_(False)
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
            if args.rows_per_step:
                group = [_sample_rows(b, args.rows_per_step, rng) for b in group]
                group_n = sum(len(b.z) for b in group)
            for b in group:
                pred = head(pooled_delta(model, b, device, HAP_CHUNK,
                                        no_grad=args.head_only),
                            b.gene_id).squeeze(-1)
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
                            "head": head.state_dict(),
                            "head_gene_ids": gene_ids, "args": vars(args),
                            "epoch": epoch, "val": val_row}, args.output)
        print(f"epoch {epoch}: mean group loss "
              f"{np.mean(losses) * args.accum_genes:.4f}")
    logger.close()
    print(f"done; best {best_val:+.4f} (val pearson) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

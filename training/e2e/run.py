"""
training/e2e/run.py  (was train_pipeline/train_end2end.py)
----------------------------------------------------------
Train the DNA encoder (embedder) and head jointly. The genome is partitioned into
~20k windows, so a batch of ~10 samples references tens of thousands of unique
fingerprint windows. We use manual activation checkpointing: embed once WITHOUT a
graph to get per-fingerprint gradients, then recompute the encoder in chunks to
accumulate its parameter gradients.

The two-pass gradient machinery, sanity check, resume logic, and backbone loading
are unchanged from the original train_end2end.py — only the data layer is swapped
onto training/common (pick a dataset by name → shared 70/15/15 split), plus a
held-out test evaluation and a lightweight run record.

v1 runs the encoder in eval() so the pass-1 and pass-2 forwards are bit-identical
without RNG bookkeeping. Validate with --sanity-check before trusting it.

    python -m training.e2e.run --dataset rice --output trained_e2e/linear/model.pt \\
        --epochs 3 --chunk-size 64 --batch-size 32 \\
        --head-checkpoint trained_heads/linear_sum/model.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from crop_embed.models.fp_head_model import (
    MLPModel, LinearModel, FPSumHeadModel, FPRefDeltaSumHeadModel, _LearnedStandardizer,
)
from crop_embed import (
    DEFAULT_MODEL_PATHS, build_window_embedder, MetricLogger, metrics_path_for,
    ActivationTracker,
)
from crop_embed.dataset import SampleDataset
from crop_embed.train import masked_mse, _compute_metrics

from training.common import features as feat, metrics as cmetrics, run_record
from training.common.datasets import get_dataset
from training.common.splits import default_split_path, get_or_build_split

# Module globals populated by main(); the chunked-embedding helpers below read them
# (args.chunk_size, autocast_ctx, dataset) exactly as the original flat script did.
args: argparse.Namespace = None            # type: ignore[assignment]
dataset = None
autocast_ctx = nullcontext

DIAG_STEPS = 5   # fire first_batch_diagnostics on this many leading steps


# ── Chunked embedding helpers (the activation-checkpointing core) ──────────────
def _chunk_starts(n, desc):
    total = (n + args.chunk_size - 1) // args.chunk_size
    return tqdm(range(0, n, args.chunk_size), desc=desc, total=total,
                unit="chunk", leave=False)


def embed_chunked_nograd(encoder, sequences, fingerprints, desc="embed (no grad)"):
    """Embed all windows under no_grad in --chunk-size groups → (n, D), detached."""
    outs = []
    with torch.no_grad():
        for i in _chunk_starts(len(sequences), desc):
            with autocast_ctx():
                outs.append(encoder(sequences[i:i + args.chunk_size],
                                    fingerprints[i:i + args.chunk_size]))
    return torch.cat(outs, dim=0)


def accumulate_encoder_grads(encoder, sequences, fingerprints, grad,
                             desc="grad accum (pass 2)"):
    """PASS 2: recompute each chunk WITH grad and backprop the cached dL/dE slice."""
    start = 0
    for i in _chunk_starts(len(sequences), desc):
        with autocast_ctx():
            e_c = encoder(sequences[i:i + args.chunk_size],
                          fingerprints[i:i + args.chunk_size])
        n = e_c.shape[0]
        e_c.backward(gradient=grad[start:start + n].to(e_c.dtype))
        start += n


def extend_with_references(sequences, fingerprints):
    """Append each window's variant-free reference fingerprint to the batch's window
    set when absent, and return a LOCAL reference index mapping every (extended) row
    to the row of its reference. `inverse` stays valid (appended rows go last)."""
    seqs = list(sequences)
    fps = list(fingerprints)
    fp_to_local = {fp: i for i, fp in enumerate(fps)}
    for fp in fingerprints:
        ref_fp = (fp[0], fp[1], fp[2], ())
        if ref_fp not in fp_to_local:
            fp_to_local[ref_fp] = len(fps)
            fps.append(ref_fp)
            seqs.append(dataset.extract_sequence(ref_fp))
    ref_local = torch.tensor(
        [fp_to_local[(fp[0], fp[1], fp[2], ())] for fp in fps], dtype=torch.long)
    return seqs, fps, ref_local


def predict_nograd(encoder, head, ds, sample_indices):
    """Full forward with no graph: chunked-embed the batch's windows, then head."""
    sequences, fingerprints, inverse = ds.gather_batch(sample_indices.cpu())
    ref_local = None
    if isinstance(head, FPRefDeltaSumHeadModel):
        sequences, fingerprints, ref_local = extend_with_references(sequences, fingerprints)
        ref_local = ref_local.to(next(head.parameters()).device)
    E = embed_chunked_nograd(encoder, sequences, fingerprints)
    dev = next(head.parameters()).device
    with torch.no_grad(), autocast_ctx():
        if ref_local is None:
            return head(E, inverse.to(dev))
        return head(E, inverse.to(dev), ref_index=ref_local)


def _grad_global_norm(params):
    sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return (torch.stack(sq).sum().sqrt().item() if sq else float("nan"))


@torch.no_grad()
def first_batch_diagnostics(step, encoder, head, E, inv, pred, loss, ref_index=None):
    """Probe the first few steps, before opt.step(), to watch the head's operating
    point drift as the encoder moves (see original design notes)."""
    table = E.detach() if ref_index is None else (E.detach() - E.detach()[ref_index])
    summed = F.embedding_bag(inv, table, mode=head.pool)
    normed = head.norm(summed)
    print(f"\n── diagnostics @ step {step} (pre-update) ──")
    print(f"  loss (pre-update)        : {loss.item():.4f}")
    print(f"  per-window E             : mean={E.mean():+.4e}  std={E.std():.4e}")
    print(f"  summed (head.norm input) : mean={summed.mean():+.4e}  std={summed.std():.4e}  "
          f"|.|mean={summed.abs().mean():.4e}")
    if isinstance(head.norm, _LearnedStandardizer):
        scale = head.norm.log_scale.exp()
        resid = summed - head.norm.mean
        print(f"  standardizer.mean        : mean={head.norm.mean.mean():+.4e}  "
              f"std={head.norm.mean.std():.4e}")
        print(f"  standardizer.scale=e^ls  : mean={scale.mean():.4e}  "
              f"min={scale.min():.4e}  max={scale.max():.4e}")
        print(f"  resid = summed - mean    : |.|mean={resid.abs().mean():.4e}  "
              f"std={resid.std():.4e}   <-- frozen-mean cancellation; grows as encoder drifts")
    else:
        print(f"  standardizer             : {type(head.norm).__name__} (normalize disabled)")
    print(f"  head.norm(summed) output : mean={normed.mean():+.4e}  std={normed.std():.4e}  "
          f"  <-- should be ~0 mean / ~1 std")
    print(f"  pred                     : mean={pred.mean():+.4e}  std={pred.std():.4e}  "
          f"|.|mean={pred.abs().mean():.4e}")
    print(f"  grad norms               : head={_grad_global_norm(head.parameters()):.4e}  "
          f"encoder={_grad_global_norm(encoder.parameters()):.4e}  "
          f"E.grad={E.grad.detach().float().norm().item():.4e}")
    print("──────────────────────────────────────────\n")


def run_sanity_check(encoder, head, ds, n_traits):
    """Validate two-pass encoder grads against a naive single-pass autograd reference
    (both in eval() so they're deterministic). Small synthetic problem."""
    encoder.eval()
    head.eval()
    device = next(head.parameters()).device
    n_win = min(args.sanity_windows, len(ds.unique_fingerprints))
    args.chunk_size = max(1, min(args.chunk_size, n_win // 4))
    print(f"sanity: {n_win} windows, chunk_size={args.chunk_size} "
          f"({-(-n_win // args.chunk_size)} chunks)")

    fps = list(ds.unique_fingerprints[:n_win])
    sequences = [ds.extract_sequence(fp) for fp in fps]

    gen = torch.Generator().manual_seed(0)
    B_fake, w = 4, max(2, n_win // 2)
    inv = torch.randint(0, n_win, (B_fake, w), generator=gen).to(device)
    targets = torch.randn(B_fake, n_traits, generator=gen).to(device)

    ref_local = ((torch.arange(n_win, device=device) + 1) % n_win
                 if isinstance(head, FPRefDeltaSumHeadModel) else None)

    def call_head(emb):
        return head(emb, inv) if ref_local is None else head(emb, inv, ref_index=ref_local)

    tracked = list(encoder.named_parameters()) + [("head." + n, p) for n, p in head.named_parameters()]

    def snapshot():
        return {n: p.grad.detach().clone() for n, p in tracked if p.grad is not None}

    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    with autocast_ctx():
        E_ref = encoder(sequences, fps)
        loss_ref = masked_mse(call_head(E_ref), targets)
    loss_ref.backward()
    ref = snapshot()

    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    E = embed_chunked_nograd(encoder, sequences, fps).requires_grad_(True)
    with autocast_ctx():
        loss_tp = masked_mse(call_head(E), targets)
    loss_tp.backward()
    accumulate_encoder_grads(encoder, sequences, fps, E.grad)
    tp = snapshot()

    tol = 1e-4 if args.precision == "fp32" else 5e-2
    worst_name, worst_diff = None, 0.0
    missing = []
    for name, g_ref in ref.items():
        if name not in tp:
            missing.append(name)
            continue
        diff = (g_ref - tp[name]).abs().max().item()
        if diff > worst_diff:
            worst_name, worst_diff = name, diff

    print(f"\n── sanity check ({len(ref)} grad tensors, tol={tol:.0e}) ──")
    print(f"  loss: reference={loss_ref.item():.6f}  two-pass={loss_tp.item():.6f}")
    print(f"  max |grad diff| = {worst_diff:.3e}  (param: {worst_name})")
    if missing:
        print(f"  WARNING: {len(missing)} params had grads in reference but not two-pass "
              f"(e.g. {missing[:3]})")
    ok = worst_diff < tol and not missing
    print(f"  {'PASSED' if ok else 'FAILED'}")
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train encoder + head end-to-end.")
    p.add_argument("--dataset", default="rice", help="registry dataset name (rice/soy/...).")
    # Embedder
    p.add_argument("--backend", choices=["dnabert2", "plantcad", "carbon"], default="dnabert2")
    p.add_argument("--model-path", type=str, default=None, help="HF repo or local dir. Defaults per --backend.")
    p.add_argument("--max-length", type=int, default=2048, help="Tokenizer truncation (PlantCAD caps 512).")
    # Genome windowing — must match the head/cache being fine-tuned.
    p.add_argument("--half-window", type=int, default=500)
    p.add_argument("--buffer", type=int, default=0)
    # Head
    p.add_argument("--head-checkpoint", type=str, default=None,
                   help="Head model.pt from training.emb_nn (arch from its head_config). Default e2e start.")
    p.add_argument("--init-head-from-scratch", action="store_true",
                   help="Build a fresh head from --head/--hidden-dim/... instead of loading a checkpoint.")
    p.add_argument("--head", choices=["linear", "mlp"], default="mlp", help="Fresh-init only.")
    p.add_argument("--hidden-dim", type=int, default=None, help="Fresh-init only.")
    p.add_argument("--n-layers", type=int, default=2, help="Fresh-init only.")
    p.add_argument("--dropout", type=float, default=0.0, help="Fresh-init only.")
    p.add_argument("--no-normalize", action="store_true", help="Disable standardizer. Fresh-init only.")
    p.add_argument("--subtract-reference", action="store_true", help="Fresh-init: FPRefDeltaSumHeadModel.")
    p.add_argument("--pool", choices=["sum", "mean"], default="sum", help="Fresh-init pooling mode.")
    # Optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 decay, inner head Linear weights only.")
    p.add_argument("--batch-size", type=int, default=8, help="Samples per step.")
    p.add_argument("--chunk-size", type=int, default=256, help="Unique windows the encoder forwards at once.")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--track-activations", action="store_true", help="Log head activation stats on --log-every steps.")
    p.add_argument("--seed", type=int, default=42)
    # wandb
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    # Modes / I/O
    p.add_argument("--sanity-check", action="store_true", help="Compare two-pass grads to a reference, then exit.")
    p.add_argument("--sanity-windows", type=int, default=64)
    p.add_argument("--output", type=str, default=None, help="Path to write model.pt (required unless --sanity-check).")
    p.add_argument("--ckpt-path", type=str, default=None, help="Resumable state (default <output>.train_state.pt).")
    p.add_argument("--no-resume", action="store_true", help="Ignore an existing --ckpt-path; start from epoch 0.")
    return p


def main() -> None:
    global args, dataset, autocast_ctx
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if not args.sanity_check and args.output is None:
        parser.error("--output is required unless --sanity-check is set.")
    if args.head_checkpoint and args.init_head_from_scratch:
        parser.error("--head-checkpoint and --init-head-from-scratch are mutually exclusive.")
    if not args.head_checkpoint and not args.init_head_from_scratch and not args.sanity_check:
        parser.error("provide --head-checkpoint <emb_nn model.pt> (default), or --init-head-from-scratch.")
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATHS[args.backend]
    if args.ckpt_path is None and args.output is not None:
        args.ckpt_path = str(Path(args.output).with_suffix(".train_state.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _amp_dtype = {"bf16": torch.bfloat16}.get(args.precision)
    autocast_ctx = ((lambda: torch.autocast(device_type=device.type, dtype=_amp_dtype))
                    if _amp_dtype is not None else nullcontext)
    if args.precision == "bf16":
        print("WARNING: bf16 not yet validated end-to-end with the manual two-pass backward. "
              "Prefer fp32 until `--sanity-check --precision bf16` passes.")

    split_path = str(default_split_path(args.dataset, args.seed))

    # ── Data (via training/common): window dataset + targets + 3-way split ────────
    spec = get_dataset(args.dataset)
    import pickle
    _dc = Path(".prepare_data_cache") / f"{args.dataset}_hw{args.half_window}_buf{args.buffer}.pkl"
    if _dc.exists():
        print(f"Loading cached window dataset from {_dc}")
        with open(_dc, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = feat.build_window_dataset(spec, args.half_window, args.buffer)
        _dc.parent.mkdir(parents=True, exist_ok=True)
        with open(_dc, "wb") as f:
            pickle.dump(dataset, f)
    Y_np, trait_cols = feat.scaled_targets(spec, dataset.samples)
    Y = torch.tensor(Y_np, dtype=torch.float32)
    n_traits = Y.shape[1]

    split = get_or_build_split(args.dataset, seed=args.seed)
    train_idx = torch.tensor(split.indices("train", dataset.samples), dtype=torch.long)
    val_idx = torch.tensor(split.indices("val", dataset.samples), dtype=torch.long)
    test_idx = torch.tensor(split.indices("test", dataset.samples), dtype=torch.long)
    train_dataset = SampleDataset(dataset, Y, train_idx)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    encoder = build_window_embedder(backend=args.backend, model_path=args.model_path,
                                    device=device, max_length=args.max_length)
    _fp0 = dataset.unique_fingerprints[0]
    with torch.no_grad():
        emb_dim = encoder([dataset.extract_sequence(_fp0)], [_fp0]).shape[1]
    print(f"emb_dim = {emb_dim}; {n_traits} traits")

    def build_head(head_config: dict) -> FPSumHeadModel:
        if head_config.get("dropout", 0.0) != 0.0:
            print(f"WARNING: overriding head dropout {head_config['dropout']} -> 0.0 (force-disabled in e2e).")
        head_config["dropout"] = 0.0
        if head_config["head"] == "linear":
            inner: nn.Module = LinearModel(emb_dim, n_traits)
        else:
            inner = MLPModel(emb_dim, n_traits, hidden_dim=head_config["hidden_dim"],
                             n_layers=head_config["n_layers"], dropout=head_config["dropout"])
        pool = head_config.get("pool", "sum")
        standardizer = head_config.get("standardizer", "perdim")
        if head_config.get("subtract_reference") or head_config.get("model_class") == "FPRefDeltaSumHeadModel":
            ref_index = FPRefDeltaSumHeadModel.build_ref_index(dataset.unique_fingerprints)
            return FPRefDeltaSumHeadModel(inner, emb_dim=emb_dim, ref_index=ref_index,
                                          normalize=head_config["normalize"], pool=pool,
                                          standardizer=standardizer).to(device)
        return FPSumHeadModel(inner, emb_dim=emb_dim, normalize=head_config["normalize"],
                              pool=pool, standardizer=standardizer).to(device)

    if args.head_checkpoint:
        print(f"Loading head from {args.head_checkpoint}")
        ckpt = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
        head_config = dict(ckpt["head_config"])
        if head_config["emb_dim"] != emb_dim:
            raise SystemExit(f"Head emb_dim ({head_config['emb_dim']}) != encoder emb_dim ({emb_dim}).")
        if head_config["n_traits"] != n_traits:
            raise SystemExit(f"Head n_traits ({head_config['n_traits']}) != current n_traits ({n_traits}).")
        head = build_head(head_config)
        head.load_state_dict(ckpt["head_state_dict"])
    else:
        head_config = {
            "head": args.head, "emb_dim": emb_dim, "n_traits": n_traits,
            "hidden_dim": args.hidden_dim, "n_layers": args.n_layers, "dropout": args.dropout,
            "normalize": not args.no_normalize, "subtract_reference": args.subtract_reference,
            "model_class": "FPRefDeltaSumHeadModel" if args.subtract_reference else "FPSumHeadModel",
            "pool": args.pool,
        }
        head = build_head(head_config)

    is_refdelta = isinstance(head, FPRefDeltaSumHeadModel)
    if is_refdelta:
        print("Reference-delta head: subtracting per-window reference embeddings each step.")

    if args.sanity_check:
        ok = run_sanity_check(encoder, head, dataset, n_traits)
        sys.exit(0 if ok else 1)

    # ── Optimizer (decay only on inner head Linear weights) ───────────────────────
    linear_weight_ids = {id(m.weight) for m in head.modules() if isinstance(m, nn.Linear)}
    decay, nodecay = [], []
    for p in list(encoder.parameters()) + list(head.parameters()):
        if not p.requires_grad:
            continue
        (decay if id(p) in linear_weight_ids else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}], lr=args.lr)

    logger = MetricLogger(metrics_path_for(args.output), wandb_project=args.wandb_project,
                          wandb_name=args.wandb_name, config=vars(args))
    print(f"Logging metrics to {logger.metrics_path}")

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"Encoder {n_enc:,} params (trainable) | head {n_head:,} params "
          f"| precision={args.precision} chunk_size={args.chunk_size}")

    tracker = ActivationTracker(head) if args.track_activations else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # ── Resumable training state ──────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt_path)

    def save_training_state(epoch, step):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch, "step": step,
            "head_state_dict": head.state_dict(),
            "embedder_state_dict": encoder.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        }
        tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)

    start_epoch, step = 0, 0
    if ckpt_path.exists() and not args.no_resume:
        print(f"Resuming from training state {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["embedder_state_dict"])
        head.load_state_dict(state["head_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["cpu_rng_state"])
        if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_epoch, step = state["epoch"], state["step"]
        print(f"  resumed at epoch {start_epoch} (step {step})")
    elif ckpt_path.exists():
        print(f"--no-resume: ignoring existing {ckpt_path}")

    print(f"\nTraining epochs {start_epoch}..{args.epochs} over {len(train_dataset)} samples …")
    for epoch in range(start_epoch, args.epochs):
        save_training_state(epoch, step)
        encoder.eval()
        head.train()
        for global_indices, targets in train_loader:
            targets = targets.to(device)
            sequences, fingerprints, inverse = dataset.gather_batch(global_indices.cpu())
            inv = inverse.to(device)

            ref_local = None
            if is_refdelta:
                sequences, fingerprints, ref_local = extend_with_references(sequences, fingerprints)
                ref_local = ref_local.to(device)

            E = embed_chunked_nograd(encoder, sequences, fingerprints,
                                     desc="embed (pass 1)").requires_grad_(True)
            opt.zero_grad()
            is_log = step % args.log_every == 0
            if tracker is not None and is_log:
                tracker.arm()
            with autocast_ctx():
                pred = head(E, inv) if ref_local is None else head(E, inv, ref_index=ref_local)
                loss = masked_mse(pred, targets)
            loss.backward()

            if step < DIAG_STEPS:
                first_batch_diagnostics(step, encoder, head, E, inv, pred, loss, ref_index=ref_local)

            accumulate_encoder_grads(encoder, sequences, fingerprints, E.grad)

            if step < DIAG_STEPS:
                first_batch_diagnostics(step, encoder, head, E, inv, pred, loss, ref_index=ref_local)

            opt.step()

            if is_log:
                train_m = _compute_metrics(pred.detach().float(), targets, trait_cols, "train")
                act_m = tracker.collect() if tracker is not None else {}
                grad_m = {
                    "grad/head_norm": _grad_global_norm(head.parameters()),
                    "grad/encoder_norm": _grad_global_norm(encoder.parameters()),
                    "grad/E_norm": E.grad.detach().float().norm().item(),
                }
                print(f"epoch {epoch:4d} step {step:6d}"
                      f"  loss={train_m.get('train/mean_mse', loss.item()):.4f}"
                      f"  pcc={train_m.get('train/mean_pcc', float('nan')):.3f}"
                      f"  |g|head={grad_m['grad/head_norm']:.2e} enc={grad_m['grad/encoder_norm']:.2e}")
                logger.log({"epoch": epoch, "step": step, **train_m, **act_m, **grad_m})
            step += 1

        head.eval()
        val_pred = predict_nograd(encoder, head, dataset, val_idx)
        val_m = _compute_metrics(val_pred.float(), Y[val_idx].to(device), trait_cols, "val")
        print(f"epoch {epoch:4d}  [val]  val_loss={val_m.get('val/mean_mse', float('nan')):.4f}"
              f"  val_pcc={val_m.get('val/mean_pcc', float('nan')):.3f}")
        logger.log({"epoch": epoch, "step": step, **val_m})

    # ── Final held-out TEST evaluation ────────────────────────────────────────────
    head.eval()
    val_pred = predict_nograd(encoder, head, dataset, val_idx).float().cpu().numpy()
    test_pred = predict_nograd(encoder, head, dataset, test_idx).float().cpu().numpy()
    val_metrics = cmetrics.evaluate(Y[val_idx].numpy(), val_pred, trait_cols)
    test_metrics = cmetrics.evaluate(Y[test_idx].numpy(), test_pred, trait_cols)
    tmn = test_metrics.get("mean", {})
    print(f"\n[test] mean pcc={tmn.get('pearson', float('nan')):.3f}  mse={tmn.get('mse', float('nan')):.4f}")

    # ── Save (unchanged format) + run record ──────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "embedder_state_dict": encoder.state_dict(),
        "head_config": head_config,
        "embedder_config": {"backend": args.backend, "model_path": args.model_path,
                            "max_length": args.max_length},
        "head_checkpoint": args.head_checkpoint,
        "split_path": split_path,
        "trait_cols": trait_cols,
        "sample_ids": dataset.samples,
        "args": vars(args),
    }, out_path)
    print(f"\nSaved to {out_path}")

    head_label = ("refdelta-" if is_refdelta else "") + head_config["head"]
    rec = run_record.build(
        dataset=args.dataset, features="e2e", model=head_label, seed=args.seed,
        traits=trait_cols,
        hyperparams={**head_config, "backend": args.backend, "lr": args.lr, "epochs": args.epochs,
                     "chunk_size": args.chunk_size, "precision": args.precision,
                     "batch_size": args.batch_size},
        metrics={"val": val_metrics, "test": test_metrics},
        backbone=args.backend, half_window=args.half_window,
        split_path=split_path)
    run_record.write(rec, out_path.parent)

    if ckpt_path.exists():
        ckpt_path.unlink()
    if tracker is not None:
        tracker.remove()
    logger.close()


if __name__ == "__main__":
    main()

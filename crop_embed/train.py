"""
crop_embed/train.py
-------------------
Generic training loop. The embedder is opaque to train() — it just has to be
an nn.Module whose forward maps a LongTensor[B] of sample indices to a
Tensor[B, n_windows, D]. Concrete implementations live in crop_embed.embedder:

  * BatchedWindowEmbedder — wraps a WindowEmbedder + dataset for end-to-end.
  * FixedWindowEmbedder   — precomputed cache; head-only training.

Multi-phase fine-tuning (e.g. head warmup → full e2e) is done by the caller
making multiple train() calls with different embedder instances.

Usage
-----
    from crop_embed import (
        UniqueWindowDataset, SampleDataset, SNPWindowPartitioner,
        WindowEmbedder, BatchedWindowEmbedder, FixedWindowEmbedder,
        LinearHead, train,
    )
    from DNABERT2_modules import load_dnabert2

    model, tokenizer = load_dnabert2()
    window_emb = WindowEmbedder(model, tokenizer, max_length=512)
    head       = LinearHead(emb_dim=768, n_traits=Y.shape[1])

    # Y: Tensor[n_samples, n_traits] aligned to dataset.samples order, NaN = missing.
    train_ds = SampleDataset(dataset, Y, train_indices)
    val_ds   = SampleDataset(dataset, Y, val_indices)

    # Phase 1: head warmup against precomputed embeddings.
    fixed = FixedWindowEmbedder.from_embedder(window_emb, dataset)
    train(fixed, head, train_ds, val_ds, lr=1e-3, steps=500)

    # Phase 2: full end-to-end.
    batched = BatchedWindowEmbedder(window_emb, dataset)
    train(batched, head, train_ds, val_ds, lr=1e-5, steps=500)
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crop_embed.dataset import SampleDataset

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def _log_wandb(payload: dict) -> None:
    """No-op unless wandb is installed AND `wandb.init` has been called by the caller."""
    if _HAS_WANDB and wandb.run is not None:
        wandb.log(payload)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over non-NaN target entries. Empty mask returns a zero with grad."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()


def _compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    trait_names: list[str] | None,
    prefix: str,
) -> dict:
    """
    Per-trait MSE and Pearson correlation, plus their means.
    NaN target entries are masked independently per trait.
    Returns a flat dict of wandb-ready keys under `prefix/`.
    Traits with fewer than 2 non-NaN samples are skipped.
    """
    pred   = pred.float()    # ensure fp32 regardless of training precision
    target = target.float()
    n_traits = pred.shape[1]
    names    = trait_names if trait_names is not None else [str(t) for t in range(n_traits)]

    mse_vals: dict[str, float] = {}
    pcc_vals: dict[str, float] = {}

    for t in range(n_traits):
        mask = ~torch.isnan(target[:, t])
        if mask.sum().item() < 2:
            continue
        p = pred[mask, t]
        y = target[mask, t]
        name = names[t]

        mse_vals[name] = ((p - y) ** 2).mean().item()

        p_c = p - p.mean()
        y_c = y - y.mean()
        pcc_vals[name] = ((p_c * y_c).sum() / (p_c.norm() * y_c.norm()).clamp(min=1e-8)).item()

    payload: dict = {}
    if mse_vals:
        payload[f"{prefix}/mean_mse"] = sum(mse_vals.values()) / len(mse_vals)
    if pcc_vals:
        payload[f"{prefix}/mean_pcc"] = sum(pcc_vals.values()) / len(pcc_vals)
    for name, v in mse_vals.items():
        payload[f"{prefix}/mse/{name}"] = v
    for name, v in pcc_vals.items():
        payload[f"{prefix}/pcc/{name}"] = v

    return payload


def _infinite(loader: DataLoader):
    """
    Yield batches forever, re-iterating the DataLoader at each epoch boundary.
    Unlike itertools.cycle, this re-invokes iter(loader) each pass so shuffle=True
    actually re-shuffles instead of replaying the first epoch's order.
    """
    while True:
        for batch in loader:
            yield batch


def _eval_split(
    head: nn.Module,
    embedder: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    autocast_ctx_factory,
    trait_names: list[str] | None,
) -> dict:
    """
    Run val_loader through the head and compute metrics on the full split.
    Collects predictions across batches so PCC is computed over the entire
    validation set, not averaged across batches.
    """
    all_preds   = []
    all_targets = []
    with torch.no_grad():
        with autocast_ctx_factory():
            for global_indices, batch_targets in val_loader:
                global_indices = global_indices.to(device)
                batch_targets  = batch_targets.to(device)
                windows = embedder(global_indices)
                all_preds.append(head(windows))
                all_targets.append(batch_targets)
    return _compute_metrics(
        torch.cat(all_preds), torch.cat(all_targets), trait_names, "val"
    )


def train(
    embedder: nn.Module,
    head: nn.Module,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset | None = None,
    *,
    loss_fn=masked_mse,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    steps: int = 1000,
    precision: str = "fp32",
    device: str | torch.device | None = None,
    log_every: int = 50,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 500,
    resume_from: str | None = None,
    phase: str | None = None,
    trait_names: list[str] | None = None,
) -> None:
    """
    Run gradient descent on whichever parameters of (embedder + head) currently
    have `requires_grad=True`. Frozen params are skipped (no optimizer state),
    so passing a FixedWindowEmbedder yields head-only training automatically.

    Parameters
    ----------
    embedder         : nn.Module mapping LongTensor[B] → Tensor[B, n_windows, D].
                       Typically BatchedWindowEmbedder (live) or FixedWindowEmbedder
                       (precomputed cache).
    head             : nn.Module mapping (B, n_windows, D) → (B, n_traits).
    train_dataset    : SampleDataset for training samples.
    val_dataset      : SampleDataset for validation samples. If provided, val
                       metrics are computed over the full split every `log_every`
                       steps so PCC is exact (not averaged across batches).
    loss_fn          : (pred, target) → scalar tensor. Default: masked_mse.
    weight_decay     : L2 weight decay applied ONLY to nn.Linear weight tensors
                       in `head`. Biases and all embedder parameters are left
                       at 0 decay (standard practice — bias decay hurts more
                       than it helps, and the embedder is typically either
                       frozen or fine-tuned with its own regularization).
    precision        : "fp32" | "bf16" | "fp16". fp16 uses GradScaler; bf16 doesn't.
    checkpoint_path  : if set, save {step, head, embedder, optimizer} every
                       `checkpoint_every` steps plus once at the end. Atomic
                       (writes to `.tmp` then renames). The embedder's full
                       state_dict is saved — for FixedWindowEmbedder that
                       includes the cache buffer.
    resume_from      : path to a checkpoint written by a previous run. Restores
                       head, embedder, and optimizer state, then resumes from
                       the step after the one that was saved.
    phase            : optional label written to the checkpoint (e.g. "phase1",
                       "phase2"). Used by callers to route resume files.
    trait_names      : human-readable names for each output trait. Used as
                       wandb metric suffixes. Falls back to "0", "1", … if None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    embedder.to(device).train()
    head.to(device).train()

    # Mixed precision
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    if amp_dtype is None:
        autocast_ctx_factory = nullcontext
    else:
        autocast_ctx_factory = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None

    # Apply weight decay only to head Linear weights (not biases, not embedder).
    head_linear_weight_ids = {
        id(m.weight) for m in head.modules() if isinstance(m, nn.Linear)
    }
    decay_params, nodecay_params = [], []
    for p in list(embedder.parameters()) + list(head.parameters()):
        if not p.requires_grad:
            continue
        if id(p) in head_linear_weight_ids:
            decay_params.append(p)
        else:
            nodecay_params.append(p)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if nodecay_params:
        groups.append({"params": nodecay_params, "weight_decay": 0.0})
    if not groups:
        raise ValueError("train(): no parameters have requires_grad=True.")
    opt = torch.optim.AdamW(groups, lr=lr)

    # Resume: restore model + optimizer state and fast-forward the step counter.
    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        head.load_state_dict(ckpt["head_state_dict"])
        embedder.load_state_dict(ckpt["embedder_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"  Resumed from {resume_from} (step {ckpt['step']} → continuing at {start_step})")

    def _save_checkpoint(step: int) -> None:
        if checkpoint_path is None:
            return
        d = os.path.dirname(checkpoint_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = checkpoint_path + ".tmp"
        payload = {
            "step":                 step,
            "head_state_dict":      head.state_dict(),
            "embedder_state_dict":  embedder.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }
        if phase is not None:
            payload["phase"] = phase
        torch.save(payload, tmp)
        os.replace(tmp, checkpoint_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_iter  = _infinite(train_loader)

    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size * 4, shuffle=False)
        if val_dataset is not None
        else None
    )

    for step in range(start_step, steps):
        global_indices, batch_targets = next(loader_iter)
        global_indices = global_indices.to(device)
        batch_targets  = batch_targets.to(device)

        with autocast_ctx_factory():
            windows = embedder(global_indices)              # (B, n_windows, D)
            y_hat   = head(windows)                         # (B, n_traits)
            loss    = loss_fn(y_hat, batch_targets)

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if step % log_every == 0:
            train_m = _compute_metrics(y_hat.detach(), batch_targets, trait_names, "train")
            log_line = (f"step {step:5d}"
                        f"  loss={train_m.get('train/mean_mse', loss.item()):.4f}"
                        f"  pcc={train_m.get('train/mean_pcc', float('nan')):.3f}")
            wandb_payload = {"step": step, **train_m}

            if val_loader is not None:
                head.eval()
                embedder.eval()
                val_m = _eval_split(head, embedder, val_loader, device,
                                    autocast_ctx_factory, trait_names)
                head.train()
                embedder.train()
                log_line += (f"  val_loss={val_m.get('val/mean_mse', float('nan')):.4f}"
                             f"  val_pcc={val_m.get('val/mean_pcc', float('nan')):.3f}")
                wandb_payload.update(val_m)

            print(log_line)
            _log_wandb(wandb_payload)

        if checkpoint_path and (step + 1) % checkpoint_every == 0:
            _save_checkpoint(step)

    if start_step < steps:
        _save_checkpoint(steps - 1)

"""
crop_embed/train.py
-------------------
Training loop and loss. Heads live in crop_embed.heads; embedders in
crop_embed.embedder. This module just glues them together with gradient
descent over whichever parameters currently have `requires_grad=True`.

Usage
-----
    from crop_embed import (
        UniqueWindowDataset, SNPWindowPartitioner,
        WindowEmbedder, LinearHead, train,
    )
    from DNABERT2_modules import load_dnabert2

    model, tokenizer = load_dnabert2()
    embedder = WindowEmbedder(model, tokenizer, max_length=512)
    head     = LinearHead(emb_dim=768, n_traits=Y.shape[1])

    # Y: Tensor[n_samples, n_traits] aligned to dataset.samples order, NaN = missing.
    train(embedder, head, dataset, Y, batch_size=32, steps=500)

For head-only training on precomputed embeddings, swap WindowEmbedder for
CachedWindowEmbedder — it has no parameters, so the optimizer updates only
the head.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn

from crop_embed.dataset import UniqueWindowDataset
from crop_embed.embedder import WindowEmbedder

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


def train(
    embedder: WindowEmbedder,
    head: nn.Module,
    dataset: UniqueWindowDataset,
    targets: torch.Tensor,
    *,
    loss_fn=masked_mse,
    batch_size: int = 32,
    lr: float = 1e-4,
    steps: int = 1000,
    precision: str = "fp32",
    device: str | torch.device | None = None,
    log_every: int = 50,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 500,
) -> None:
    """
    Run gradient descent on whichever parameters of (embedder + head) currently
    have `requires_grad=True`. Frozen params are skipped (no optimizer state).

    For two-phase fine-tuning, call this twice:
      1. Freeze the embedder (`for p in embedder.parameters(): p.requires_grad_(False)`),
         call train(...) — only the head updates.
      2. Unfreeze the embedder, call train(...) again — full end-to-end.
    Adam state for the head resets at the phase boundary; for most settings this
    costs only a few warmup steps.

    wandb logging is automatic: if the caller has run `wandb.init(...)`, train()
    logs loss / step / unique_windows every `log_every` steps. Otherwise no-op.

    Parameters
    ----------
    embedder         : WindowEmbedder — wraps a DNA encoder + tokenizer + pooling.
    head             : nn.Module mapping (B, n_windows, D) → (B, n_traits).
    dataset          : UniqueWindowDataset; rows of `dataset.samples` define sample order.
    targets          : Tensor[n_samples, n_traits] aligned to `dataset.samples` order.
                       NaN entries are masked out of the loss.
    loss_fn          : (pred, target) → scalar tensor. Default: masked_mse.
    precision        : "fp32" | "bf16" | "fp16". fp16 uses GradScaler; bf16 doesn't.
    checkpoint_path  : if set, save {step, head, embedder, optimizer} every
                       `checkpoint_every` steps plus once at the end. Atomic
                       (writes to `.tmp` then renames).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    embedder.to(device).train()
    head.to(device).train()
    targets = targets.to(device)

    # Mixed precision
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    if amp_dtype is None:
        autocast_ctx_factory = nullcontext
    else:
        autocast_ctx_factory = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None

    # Only optimize params that are currently trainable. Lets the caller freeze
    # the embedder by setting requires_grad=False externally.
    params = [
        p for p in list(embedder.parameters()) + list(head.parameters())
        if p.requires_grad
    ]
    if not params:
        raise ValueError("train(): no parameters have requires_grad=True.")
    opt = torch.optim.AdamW(params, lr=lr)

    def _save_checkpoint(step: int) -> None:
        if checkpoint_path is None:
            return
        d = os.path.dirname(checkpoint_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = checkpoint_path + ".tmp"
        torch.save({
            "step":                 step,
            "head_state_dict":      head.state_dict(),
            "embedder_state_dict":  embedder.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }, tmp)
        os.replace(tmp, checkpoint_path)

    n_samples = len(dataset.samples)
    for step in range(steps):
        idx = torch.randint(n_samples, (batch_size,))
        sequences, fingerprints, inverse = dataset.gather_batch(idx)

        with autocast_ctx_factory():
            emb     = embedder(sequences, fingerprints)     # (N_unique, D)
            windows = emb[inverse.to(device)]               # (B, n_windows, D)
            y_hat   = head(windows)                         # (B, n_traits)
            loss    = loss_fn(y_hat, targets[idx])

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if step % log_every == 0:
            print(f"step {step:5d}  loss={loss.item():.4f}  "
                  f"unique_windows={len(sequences):,}")
            _log_wandb({
                "loss":           loss.item(),
                "step":           step,
                "unique_windows": len(sequences),
            })

        if checkpoint_path and (step + 1) % checkpoint_every == 0:
            _save_checkpoint(step)

    _save_checkpoint(steps - 1)

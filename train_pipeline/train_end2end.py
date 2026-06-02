"""
train_end2end.py
-----------------
Train the DNA encoder (embedder) and head jointly. The genome is partitioned into
~20k windows, so a batch of ~10 samples references tens of thousands of unique
fingerprint windows. Building the full encoder autograd graph over all of those
windows at once OOMs by orders of magnitude (a transformer caches per-layer
activations for every window it forwards under grad), so we use manual activation
checkpointing: embed once WITHOUT a graph to get per-fingerprint gradients, then
recompute the encoder in chunks to accumulate its parameter gradients.

The output is a saved ckpt with the head + embedder state_dicts and the metadata
needed to reconstruct both for inference.

v1 (this file) runs the encoder in eval() — dropout off — so the pass-1 and pass-2
forwards are bit-identical without any RNG bookkeeping. Validate it with
--sanity-check before trusting it (compares the two-pass encoder grads against a
naive single-pass autograd reference). Encoder dropout + per-chunk RNG handling is
a later addition; see the design notes below.
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.fp_head_model import (
    MLPModel, LinearModel, FPSumHeadModel,
)
from crop_embed import (
    DEFAULT_MODEL_PATHS, build_window_embedder, MetricLogger, metrics_path_for,
)
from crop_embed.data.loading import prepare_data
from crop_embed.train import masked_mse, _compute_metrics

"""
DESIGN NOTES (full spec; v1 implements the eval() path)

INITIALIZATION
1. Load train/val datasets + split via prepare_data (hardcoded SPLIT_PATH for now).
2. Load the trainable embedder (WindowEmbedder around DNABERT-2 / PlantCAD).
3. Build the head: FPSumHeadModel wrapping LinearModel or MLPModel (no gather head
   yet). No standardizer warm-start in e2e — the fingerprint embeddings move every
   step, so fixed init stats would be stale.
4. Optimizer (AdamW over encoder + head params; weight decay only on inner Linear
   weights), loss = masked_mse, MetricLogger, epoch/batch params.
   - --precision {fp32, bf16}: default fp32. Avoid fp16+GradScaler — loss scaling
     interacts awkwardly with the two manual backward passes.
   - --chunk-size: # of unique fingerprint windows the encoder forwards at once.
     The one knob trading recompute memory (pass 2) against step time; tune per
     GPU. Applies to EVERY embedding pass — pass 1, pass 2, and val.

TRAIN LOOP (per batch of sample indices from a shuffled DataLoader)
1. sequences, fingerprints, inverse = dataset.gather_batch(sample_indices)
   `inverse` is (B, n_windows) of *local* indices into the batch's unique
   fingerprint set; E is the (n_unique_batch, D) embedding tensor those index into.
2. PASS 1 (no encoder graph): chunk the unique fingerprints; embed each chunk under
   no_grad into E_detached. E = E_detached.requires_grad_(True). opt.zero_grad();
   loss = masked_mse(head(E, inverse), targets); loss.backward() — populates head
   grads AND E.grad (= dL/dE), no encoder grads yet.
3. PASS 2 (recompute, same chunk boundaries): for each chunk, e_c = encoder(chunk)
   WITH grad; e_c.backward(gradient=E.grad[chunk]) to accumulate encoder grads.
   Peak mem = E (tiny) + head graph + one chunk's encoder graph.
   v1 keeps the encoder in eval() so pass-2 reproduces pass-1 exactly with no RNG
   work. With dropout ON you MUST capture/restore per-chunk RNG (or use
   torch.utils.checkpoint, which handles it for free — same --chunk-size knob),
   else dL/dE (realization A) is paired with dE/dθ (realization B) → wrong grads.
4. opt.step()  # updates head (pass-1 grads) + encoder (pass-2 grads) together.

VALIDATION / LOGGING (mirror train_head.py)
- Every --log-every steps: log train metrics from the current batch.
- Once per epoch: eval the full val split with encoder + head in eval()/no_grad,
  embedding via the same --chunk-size pass. Save state_dicts + provenance at end.
"""

SPLIT_PATH = "splits/sativas413_seed42.pt"

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train encoder + head end-to-end.")

# Embedder
parser.add_argument("--backend", choices=["dnabert2", "plantcad"], default="dnabert2")
parser.add_argument("--model-path", type=str, default=None,
                    help="HF repo or local dir. Defaults per --backend.")
parser.add_argument("--max-length", type=int, default=2048,
                    help="Tokenizer truncation length (PlantCAD caps at 512).")

# Head
parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
parser.add_argument("--hidden-dim", type=int, default=None)
parser.add_argument("--n-layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--no-normalize", action="store_true",
                    help="Disable the FPSumHeadModel standardizer.")

# Optimization
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 decay, applied only to inner head Linear weights.")
parser.add_argument("--batch-size", type=int, default=8,
                    help="Samples per step. Small — each carries ~n_windows windows.")
parser.add_argument("--chunk-size", type=int, default=256,
                    help="Unique windows the encoder forwards at once (memory knob). "
                         "Applies to pass 1, pass 2, and val. Tune per GPU.")
parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 for debugging; bf16 for speed/memory once stable.")
parser.add_argument("--log-every", type=int, default=20,
                    help="Log train metrics every N optimizer steps.")
parser.add_argument("--seed", type=int, default=42)

# wandb
parser.add_argument("--wandb-project", type=str, default=None)
parser.add_argument("--wandb-name", type=str, default=None)

# Modes / I/O
parser.add_argument("--sanity-check", action="store_true",
                    help="Compare two-pass encoder grads against a naive single-pass "
                         "autograd reference on a small synthetic window set, then "
                         "exit. Correctness is window-count-independent, so this stays "
                         "tiny regardless of --batch-size.")
parser.add_argument("--sanity-windows", type=int, default=64,
                    help="# of unique windows to test in --sanity-check. Kept small so "
                         "the naive full-graph reference fits; chunk size is shrunk to "
                         "force several chunks.")
parser.add_argument("--output", type=str, default=None,
                    help="Path to write model.pt. Required unless --sanity-check.")

args = parser.parse_args()
torch.manual_seed(args.seed)

if not args.sanity_check and args.output is None:
    parser.error("--output is required unless --sanity-check is set.")
if args.model_path is None:
    args.model_path = DEFAULT_MODEL_PATHS[args.backend]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bf16 autocast or fp32 no-op. No GradScaler (fp16 is intentionally unsupported).
_amp_dtype = {"bf16": torch.bfloat16}.get(args.precision)
autocast_ctx = (
    (lambda: torch.autocast(device_type=device.type, dtype=_amp_dtype))
    if _amp_dtype is not None else nullcontext
)
if args.precision == "bf16":
    print("WARNING: bf16 is not yet validated end-to-end with the manual two-pass "
          "backward (the cached dL/dE is cast to bf16 in pass 2). Stick with fp32 "
          "until `--sanity-check --precision bf16` passes within tolerance.")

# ── Chunked embedding helpers (the activation-checkpointing core) ──────────────

def embed_chunked_nograd(encoder, sequences, fingerprints):
    """Embed all windows under no_grad in --chunk-size groups → (n, D), detached."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(sequences), args.chunk_size):
            with autocast_ctx():
                outs.append(encoder(sequences[i:i + args.chunk_size],
                                    fingerprints[i:i + args.chunk_size]))
    return torch.cat(outs, dim=0)

def accumulate_encoder_grads(encoder, sequences, fingerprints, grad):
    """PASS 2: recompute each chunk WITH grad and backprop the cached dL/dE slice."""
    start = 0
    for i in range(0, len(sequences), args.chunk_size):
        with autocast_ctx():
            e_c = encoder(sequences[i:i + args.chunk_size],
                          fingerprints[i:i + args.chunk_size])
        n = e_c.shape[0]
        e_c.backward(gradient=grad[start:start + n].to(e_c.dtype))
        start += n


def predict_nograd(encoder, head, dataset, sample_indices):
    """Full forward with no graph: chunked-embed the batch's windows, then head."""
    sequences, fingerprints, inverse = dataset.gather_batch(sample_indices.cpu())
    E = embed_chunked_nograd(encoder, sequences, fingerprints)
    with torch.no_grad(), autocast_ctx():
        return head(E, inverse.to(device))


# ── Sanity check: two-pass grads vs. naive single-pass reference ──────────────

def run_sanity_check(encoder, head, dataset, n_traits):
    """
    Validate the two-pass gradient machinery against a naive single-pass autograd
    reference (embed all windows WITH grad → head → backward), both in eval() so
    they're deterministic.

    Crucially this runs on a SMALL synthetic problem — the first --sanity-windows
    unique fingerprints — NOT a real batch. A real sample spans ~20k windows, so
    the naive reference's full graph would itself OOM (the very thing the two-pass
    avoids). Correctness of the chunked recompute is independent of the window
    count; it only needs >1 chunk, so we shrink the window set and the chunk size
    to keep the reference tiny while still exercising the chunk boundaries.
    """
    encoder.eval()
    head.eval()
    n_win = min(args.sanity_windows, len(dataset.unique_fingerprints))
    # Force several chunks even if --chunk-size is large; helpers read args.chunk_size.
    args.chunk_size = max(1, min(args.chunk_size, n_win // 4))
    print(f"sanity: {n_win} windows, chunk_size={args.chunk_size} "
          f"({-(-n_win // args.chunk_size)} chunks)")

    fps = list(dataset.unique_fingerprints[:n_win])
    sequences = [dataset.extract_sequence(fp) for fp in fps]

    # Synthetic batch: a few fake samples each summing a random subset of windows.
    gen = torch.Generator().manual_seed(0)
    B_fake, w = 4, max(2, n_win // 2)
    inv = torch.randint(0, n_win, (B_fake, w), generator=gen).to(device)
    targets = torch.randn(B_fake, n_traits, generator=gen).to(device)

    tracked = list(encoder.named_parameters()) + [("head." + n, p) for n, p in head.named_parameters()]

    def snapshot():
        return {n: p.grad.detach().clone() for n, p in tracked if p.grad is not None}

    # Naive reference: one forward over all n_win windows, with grad (fits — n_win small).
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    with autocast_ctx():
        E_ref = encoder(sequences, fps)
        loss_ref = masked_mse(head(E_ref, inv), targets)
    loss_ref.backward()
    ref = snapshot()

    # Manual two-pass.
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    E = embed_chunked_nograd(encoder, sequences, fps).requires_grad_(True)
    with autocast_ctx():
        loss_tp = masked_mse(head(E, inv), targets)
    loss_tp.backward()
    accumulate_encoder_grads(encoder, sequences, fps, E.grad)
    tp = snapshot()

    # Compare.
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


# ── Build everything ──────────────────────────────────────────────────────────

data = prepare_data(split_path=SPLIT_PATH)
dataset    = data["dataset"]
Y          = data["Y"]
trait_cols = data["trait_cols"]
train_idx  = data["train_idx"]
val_idx    = data["val_idx"]
train_dataset = data["train_dataset"]
n_traits = Y.shape[1]

encoder = build_window_embedder(
    backend=args.backend, model_path=args.model_path,
    device=device, max_length=args.max_length,
)

# Probe emb_dim with a tiny forward (backend-agnostic — PlantCAD RC-averages, etc.).
_fp0 = dataset.unique_fingerprints[0]
with torch.no_grad():
    emb_dim = encoder([dataset.extract_sequence(_fp0)], [_fp0]).shape[1]
print(f"emb_dim = {emb_dim}; {n_traits} traits")

if args.head == "linear":
    inner: nn.Module = LinearModel(emb_dim, n_traits)
else:
    inner = MLPModel(emb_dim, n_traits, hidden_dim=args.hidden_dim,
                     n_layers=args.n_layers, dropout=args.dropout)
# No standardizer warm-start in e2e — fingerprint embeddings shift every step.
head = FPSumHeadModel(inner, emb_dim=emb_dim, normalize=not args.no_normalize).to(device)

# ── Sanity-check mode: run on a small synthetic problem and exit ──────────────

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
     {"params": nodecay, "weight_decay": 0.0}],
    lr=args.lr,
)

logger = MetricLogger(
    metrics_path_for(args.output),
    wandb_project=args.wandb_project, wandb_name=args.wandb_name, config=vars(args),
)
print(f"Logging metrics to {logger.metrics_path}")

n_enc = sum(p.numel() for p in encoder.parameters())
n_head = sum(p.numel() for p in head.parameters())
print(f"Encoder {n_enc:,} params (trainable) | head {n_head:,} params "
      f"| precision={args.precision} chunk_size={args.chunk_size}")

# ── Train ─────────────────────────────────────────────────────────────────────

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(f"\nTraining {args.epochs} epochs over {len(train_dataset)} samples …")
step = 0
for epoch in range(args.epochs):
    # v1: encoder stays in eval() (dropout off) so the two passes match without
    # RNG bookkeeping. The head trains normally.
    encoder.eval()
    head.train()
    for global_indices, targets in train_loader:
        targets = targets.to(device)
        sequences, fingerprints, inverse = dataset.gather_batch(global_indices.cpu())
        inv = inverse.to(device)

        # PASS 1 — no encoder graph; get head grads + E.grad (= dL/dE).
        E = embed_chunked_nograd(encoder, sequences, fingerprints).requires_grad_(True)
        opt.zero_grad()
        with autocast_ctx():
            pred = head(E, inv)
            loss = masked_mse(pred, targets)
        loss.backward()

        # PASS 2 — recompute chunks WITH grad to accumulate encoder grads.
        accumulate_encoder_grads(encoder, sequences, fingerprints, E.grad)
        opt.step()

        if step % args.log_every == 0:
            train_m = _compute_metrics(pred.detach().float(), targets, trait_cols, "train")
            print(f"epoch {epoch:4d} step {step:6d}"
                  f"  loss={train_m.get('train/mean_mse', loss.item()):.4f}"
                  f"  pcc={train_m.get('train/mean_pcc', float('nan')):.3f}")
            logger.log({"epoch": epoch, "step": step, **train_m})
        step += 1

    # End-of-epoch validation over the full split (eval/no_grad, chunked embed).
    head.eval()
    val_pred = predict_nograd(encoder, head, dataset, val_idx)
    val_m = _compute_metrics(val_pred.float(), Y[val_idx].to(device), trait_cols, "val")
    print(f"epoch {epoch:4d}  [val]"
          f"  val_loss={val_m.get('val/mean_mse', float('nan')):.4f}"
          f"  val_pcc={val_m.get('val/mean_pcc', float('nan')):.3f}")
    logger.log({"epoch": epoch, "step": step, **val_m})

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "head_state_dict": head.state_dict(),
    "embedder_state_dict": encoder.state_dict(),
    "head_config": {
        "head": args.head, "emb_dim": emb_dim, "n_traits": n_traits,
        "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
        "dropout": args.dropout, "normalize": not args.no_normalize,
    },
    "embedder_config": {
        "backend": args.backend, "model_path": args.model_path,
        "max_length": args.max_length,
    },
    "split_path": SPLIT_PATH,
    "trait_cols": trait_cols,
    "sample_ids": dataset.samples,
    "args": vars(args),
}, out_path)
print(f"\nSaved to {out_path}")

logger.close()

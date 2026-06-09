"""
train_end2end.py
-----------------
Train the DNA encoder (embedder) and head jointly. The genome is partitioned into
~20k windows, so a batch of ~10 samples references tens of thousands of unique
fingerprint windows. We use manual activation
checkpointing: embed once WITHOUT a graph to get per-fingerprint gradients, then
recompute the encoder in chunks to accumulate its parameter gradients.

The output is a saved ckpt with the head + embedder state_dicts and the metadata
needed to reconstruct both for inference.

v1 (this file) runs the encoder in eval() so the pass-1 and pass-2
forwards are bit-identical without any RNG bookkeeping. Validate it with
--sanity-check before trusting it (compares the two-pass encoder grads against a
naive single-pass autograd reference). Encoder dropout + per-chunk RNG handling is
a later addition; see the design notes below.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_embed.models.fp_head_model import (
    MLPModel, LinearModel, FPSumHeadModel, FPRefDeltaSumHeadModel, _LearnedStandardizer,
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
   yet). By default the head is loaded from a train_head.py checkpoint
   (--head-checkpoint) — architecture from its head_config, weights from its
   state_dict — so e2e fine-tunes an already-trained head. --init-head-from-scratch
   opts back into a fresh head. Either way, no standardizer warm-start here: the
   fingerprint embeddings move every step, so fixed init stats would be stale (and a
   loaded checkpoint already carries a trained standardizer).
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

  python train_pipeline/train_end2end.py --output trained_embeddings/linear_sum/model.pt --log-every 1 --epochs 3 --chunk-size 64 --batch-size 32 --head-checkpoint trained_heads/linear_sum/model.pt
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

# Genome windowing — must match the cache/head being fine-tuned (the dataset is
# rebuilt here for the split + targets, so a mismatch breaks the head's ref index).
parser.add_argument("--half-window", type=int, default=500,
                    help="Windowing half-window; must match the loaded head/cache.")
parser.add_argument("--buffer", type=int, default=0,
                    help="Windowing buffer; must match the loaded head/cache.")

# Head. By default the head is loaded from a train_head.py checkpoint so the e2e
# run fine-tunes an already-trained head rather than a fresh one. The architecture
# args below are only consulted with --init-head-from-scratch; when loading a
# checkpoint the architecture comes from its head_config (so the state_dict fits).
parser.add_argument("--head-checkpoint", type=str, default=None,
                    help="Path to a head model.pt produced by train_head.py. The head "
                         "architecture is taken from the checkpoint's head_config and its "
                         "weights are loaded. This is the default starting point for e2e.")
parser.add_argument("--init-head-from-scratch", action="store_true",
                    help="Opt out of loading a checkpoint and initialize a fresh head from "
                         "the --head/--hidden-dim/--n-layers/--dropout/--no-normalize args.")
parser.add_argument("--head", choices=["linear", "mlp"], default="mlp",
                    help="Fresh-init only (--init-head-from-scratch).")
parser.add_argument("--hidden-dim", type=int, default=None,
                    help="Fresh-init only (--init-head-from-scratch).")
parser.add_argument("--n-layers", type=int, default=2,
                    help="Fresh-init only (--init-head-from-scratch).")
parser.add_argument("--dropout", type=float, default=0.0,
                    help="Fresh-init only (--init-head-from-scratch).")
parser.add_argument("--no-normalize", action="store_true",
                    help="Disable the FPSumHeadModel standardizer. Fresh-init only "
                         "(--init-head-from-scratch).")
parser.add_argument("--subtract-reference", action="store_true",
                    help="Fresh-init only: build an FPRefDeltaSumHeadModel (subtract each "
                         "window's reference embedding before pooling) instead of "
                         "FPSumHeadModel. When loading --head-checkpoint this is taken "
                         "from the checkpoint's head_config.")

# Optimization
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 decay, applied only to inner head Linear weights.")
parser.add_argument("--batch-size", type=int, default=8,
                    help="Samples per step.")
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
parser.add_argument("--ckpt-path", type=str, default=None,
                    help="Resumable training-state checkpoint (model + optimizer + RNG + "
                         "epoch). Written at the start of every epoch and auto-loaded on "
                         "startup if present. Defaults to <output>.train_state.pt.")
parser.add_argument("--no-resume", action="store_true",
                    help="Ignore any existing --ckpt-path and start from epoch 0 (new "
                         "checkpoints are still written each epoch).")

#

args = parser.parse_args()
torch.manual_seed(args.seed)

if not args.sanity_check and args.output is None:
    parser.error("--output is required unless --sanity-check is set.")
if args.head_checkpoint and args.init_head_from_scratch:
    parser.error("--head-checkpoint and --init-head-from-scratch are mutually exclusive.")
# Loading a train_head.py checkpoint is the default head source; fresh init is an
# explicit opt-in. Sanity-check only exercises the gradient machinery (architecture-
# independent), so it falls back to a fresh head when no checkpoint is given.
if not args.head_checkpoint and not args.init_head_from_scratch and not args.sanity_check:
    parser.error("provide --head-checkpoint <train_head.py model.pt> (default), or pass "
                 "--init-head-from-scratch to build a fresh head from the architecture args.")
if args.model_path is None:
    args.model_path = DEFAULT_MODEL_PATHS[args.backend]
if args.ckpt_path is None and args.output is not None:
    # Sits next to the final model: dir/model.pt -> dir/model.train_state.pt
    args.ckpt_path = str(Path(args.output).with_suffix(".train_state.pt"))

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

def _chunk_starts(n, desc):
    """Chunk-start indices wrapped in a leave=False tqdm bar (it's an inner loop run
    many times per epoch, so the bar clears itself instead of stacking up)."""
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
    """Append each window's variant-free reference fingerprint (chrom, ws, we, ())
    to the batch's window set when it isn't already present, and return a LOCAL
    reference index mapping every (extended) row to the row of its reference.

    The e2e head only embeds the batch's unique windows, but a window's reference
    is in that set only if some batch sample is all-reference there. Adding the
    missing references lets FPRefDeltaSumHeadModel do its `E - E[ref]` subtraction
    against the batch-local tensor (and the encoder gets gradients for the
    reference windows too). `inverse` stays valid: appended rows go after the
    originals, so the original local indices are unchanged.
    """
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
        [fp_to_local[(fp[0], fp[1], fp[2], ())] for fp in fps], dtype=torch.long
    )
    return seqs, fps, ref_local


def predict_nograd(encoder, head, dataset, sample_indices):
    """Full forward with no graph: chunked-embed the batch's windows, then head."""
    sequences, fingerprints, inverse = dataset.gather_batch(sample_indices.cpu())
    ref_local = None
    if isinstance(head, FPRefDeltaSumHeadModel):
        sequences, fingerprints, ref_local = extend_with_references(sequences, fingerprints)
        ref_local = ref_local.to(device)
    E = embed_chunked_nograd(encoder, sequences, fingerprints)
    with torch.no_grad(), autocast_ctx():
        if ref_local is None:
            return head(E, inverse.to(device))
        return head(E, inverse.to(device), ref_index=ref_local)


def _grad_global_norm(params):
    """L2 norm of the concatenated .grad over an iterable of params (skips None)."""
    sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return (torch.stack(sq).sum().sqrt().item() if sq else float("nan"))


DIAG_STEPS = 5   # fire first_batch_diagnostics on this many leading steps

@torch.no_grad()
def first_batch_diagnostics(step, head, E, inv, pred, loss, ref_index=None):
    """Probe the first few steps, BEFORE opt.step(), to watch the head's operating
    point drift across updates. Run after both backward passes so grads exist.

    The hypothesis (from the step-0 numbers): head.norm is a _LearnedStandardizer
    whose `mean` was frozen at the cache/initial-encoder operating point. It cancels
    a ~1500-magnitude per-dim offset down to an O(0.1) residual — condition number
    ~1e4. Frozen in train_head (encoder fixed) this is fine; here the encoder moves,
    so `summed` peels away from the frozen `mean` and the residual (hence the inner
    model's input, hence the prediction) explodes. Watch `resid` and `normed.std`
    grow while `summed`/`mean` stay ~1500.
    """
    # The head pools E (plain) or the reference-delta E - E[ref] (refdelta).
    table  = E.detach() if ref_index is None else (E.detach() - E.detach()[ref_index])
    summed = F.embedding_bag(inv, table, mode="sum")   # (B, D) — the head.norm input
    normed = head.norm(summed)                               # what the inner model actually sees
    print(f"\n── diagnostics @ step {step} (pre-update) ──")
    print(f"  loss (pre-update)        : {loss.item():.4f}")
    print(f"  per-window E             : mean={E.mean():+.4e}  std={E.std():.4e}")
    print(f"  summed (head.norm input) : mean={summed.mean():+.4e}  std={summed.std():.4e}  "
          f"|.|mean={summed.abs().mean():.4e}")
    if isinstance(head.norm, _LearnedStandardizer):
        scale = head.norm.log_scale.exp()
        resid = summed - head.norm.mean                     # the cancelled quantity
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


# ── Sanity check: two-pass grads vs. naive single-pass reference ──────────────

def run_sanity_check(encoder, head, dataset, n_traits):
    """
    Validate the two-pass gradient machinery against a naive single-pass autograd
    reference (embed all windows WITH grad → head → backward), both in eval() so
    they're deterministic.

    this runs on a SMALL synthetic problem, NOT a real batch. 
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

    # A refdelta head's stored ref_index is global (full cache); for the synthetic
    # window set, pass a local index. Use a roll (window i -> i+1) rather than
    # identity so the deltas (and thus the encoder grads being validated) are
    # nonzero and the subtraction path is actually exercised.
    ref_local = ((torch.arange(n_win, device=device) + 1) % n_win
                 if isinstance(head, FPRefDeltaSumHeadModel) else None)

    def call_head(emb):
        return head(emb, inv) if ref_local is None else head(emb, inv, ref_index=ref_local)

    tracked = list(encoder.named_parameters()) + [("head." + n, p) for n, p in head.named_parameters()]

    def snapshot():
        return {n: p.grad.detach().clone() for n, p in tracked if p.grad is not None}

    # Naive reference: one forward over all n_win windows, with grad (fits — n_win small).
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    with autocast_ctx():
        E_ref = encoder(sequences, fps)
        loss_ref = masked_mse(call_head(E_ref), targets)
    loss_ref.backward()
    ref = snapshot()

    # Manual two-pass.
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    E = embed_chunked_nograd(encoder, sequences, fps).requires_grad_(True)
    with autocast_ctx():
        loss_tp = masked_mse(call_head(E), targets)
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

data = prepare_data(split_path=SPLIT_PATH,
                    half_window=args.half_window, buffer=args.buffer)
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

def build_head(head_config: dict) -> FPSumHeadModel:
    """Construct an FPSumHeadModel from a head_config dict (as saved by train_head.py).

    No standardizer warm-start in e2e — the fingerprint embeddings shift every step,
    so fixed init stats would be stale (and a loaded checkpoint already carries a
    trained standardizer in its state_dict).

    Dropout is forced to 0 here regardless of the checkpoint/CLI value. Dropout
    carries no parameters, so this never affects load_state_dict; head_config is
    updated in place so the saved metadata reflects the 0 actually used.
    """
    if head_config.get("dropout", 0.0) != 0.0:
        print(f"WARNING: overriding head dropout {head_config['dropout']} -> 0.0 "
              f"(dropout is force-disabled in e2e for now).")
    head_config["dropout"] = 0.0
    if head_config["head"] == "linear":
        inner: nn.Module = LinearModel(emb_dim, n_traits)
    else:
        inner = MLPModel(emb_dim, n_traits, hidden_dim=head_config["hidden_dim"],
                         n_layers=head_config["n_layers"], dropout=head_config["dropout"])
    if head_config.get("subtract_reference") or head_config.get("model_class") == "FPRefDeltaSumHeadModel":
        # Reference-delta head. The global ref index (this dataset's fingerprints,
        # cache-row order) matches the train_refdelta_head.py checkpoint's ref_index
        # buffer so the state_dict loads; per-batch LOCAL indices are passed at forward.
        ref_index = FPRefDeltaSumHeadModel.build_ref_index(dataset.unique_fingerprints)
        return FPRefDeltaSumHeadModel(
            inner, emb_dim=emb_dim, ref_index=ref_index,
            normalize=head_config["normalize"],
        ).to(device)
    return FPSumHeadModel(inner, emb_dim=emb_dim, normalize=head_config["normalize"]).to(device)


if args.head_checkpoint:
    # Default path: start from a head trained by train_head.py. The architecture is
    # taken from the checkpoint's head_config so the state_dict loads cleanly; the
    # --head/--hidden-dim/... CLI args are ignored here.
    print(f"Loading head from {args.head_checkpoint}")
    ckpt = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
    head_config = dict(ckpt["head_config"])
    if head_config["emb_dim"] != emb_dim:
        raise SystemExit(
            f"Head checkpoint emb_dim ({head_config['emb_dim']}) != encoder emb_dim "
            f"({emb_dim}). The head was trained on a different embedder/cache."
        )
    if head_config["n_traits"] != n_traits:
        raise SystemExit(
            f"Head checkpoint n_traits ({head_config['n_traits']}) != current n_traits "
            f"({n_traits}). The head was trained on a different trait set/split."
        )
    head = build_head(head_config)
    head.load_state_dict(ckpt["head_state_dict"])
else:
    # Opt-in fresh init from the architecture CLI args (the pre-checkpoint behavior).
    head_config = {
        "head": args.head, "emb_dim": emb_dim, "n_traits": n_traits,
        "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
        "dropout": args.dropout, "normalize": not args.no_normalize,
        "subtract_reference": args.subtract_reference,
        "model_class": "FPRefDeltaSumHeadModel" if args.subtract_reference else "FPSumHeadModel",
    }
    head = build_head(head_config)

is_refdelta = isinstance(head, FPRefDeltaSumHeadModel)
if is_refdelta:
    print("Reference-delta head: subtracting per-window reference embeddings each step "
          "(batches extended with their missing reference windows).")

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

# ── Resumable training state (model + optimizer + RNG + epoch) ─────────────────

ckpt_path = Path(args.ckpt_path)

def save_training_state(epoch, step):
    """Snapshot enough to resume at the START of `epoch`. Written atomically so a
    crash during the save can't corrupt an existing checkpoint."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch, "step": step,
        "head_state_dict": head.state_dict(),
        "embedder_state_dict": encoder.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
    }
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, ckpt_path)  # atomic on POSIX

# Restore if a checkpoint is already there (unless --no-resume). Load on CPU and let
# load_state_dict copy into the on-device params (optimizer state follows its params).
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
    # Checkpoint at the START of the epoch: a crash mid-epoch resumes by replaying
    # this whole epoch from a clean boundary (model/optimizer are exactly as here).
    save_training_state(epoch, step)
    # v1: encoder stays in eval() (dropout off) so the two passes match without
    # RNG bookkeeping. The head trains normally.
    encoder.eval()
    head.train()
    for global_indices, targets in train_loader:
        targets = targets.to(device)
        sequences, fingerprints, inverse = dataset.gather_batch(global_indices.cpu())
        inv = inverse.to(device)

        # Reference-delta head: extend the batch's window set with the missing
        # reference windows and build a batch-local reference index so the head can
        # subtract each window's live reference embedding. `inverse` is unchanged.
        ref_local = None
        if is_refdelta:
            sequences, fingerprints, ref_local = extend_with_references(sequences, fingerprints)
            ref_local = ref_local.to(device)

        # PASS 1 — no encoder graph; get head grads + E.grad (= dL/dE).
        E = embed_chunked_nograd(encoder, sequences, fingerprints,
                                 desc="embed (pass 1)").requires_grad_(True)
        opt.zero_grad()
        with autocast_ctx():
            pred = head(E, inv) if ref_local is None else head(E, inv, ref_index=ref_local)
            loss = masked_mse(pred, targets)
        loss.backward()

        # Probe the first few steps, before each update, to watch the drift.
        if step < DIAG_STEPS:
            first_batch_diagnostics(step, head, E, inv, pred, loss, ref_index=ref_local)

        # PASS 2 — recompute chunks WITH grad to accumulate encoder grads.
        accumulate_encoder_grads(encoder, sequences, fingerprints, E.grad)

        # Probe the first few steps, before each update, to watch the drift.
        if step < DIAG_STEPS:
            first_batch_diagnostics(step, head, E, inv, pred, loss, ref_index=ref_local)

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
    "head_config": head_config,
    "embedder_config": {
        "backend": args.backend, "model_path": args.model_path,
        "max_length": args.max_length,
    },
    "head_checkpoint": args.head_checkpoint,
    "split_path": SPLIT_PATH,
    "trait_cols": trait_cols,
    "sample_ids": dataset.samples,
    "args": vars(args),
}, out_path)
print(f"\nSaved to {out_path}")

# Training finished cleanly — drop the resumable checkpoint so a later run with the
# same --output starts fresh instead of resuming a completed run.
if ckpt_path.exists():
    ckpt_path.unlink()

logger.close()

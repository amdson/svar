"""
demo_train.py
-------------
Train a head model on rice phenotypes using per-window DNA embeddings, with
either cached embeddings (frozen embedder) or end-to-end through DNABERT-2.

Four supported configurations. For trainable-embedder runs (3 and 4), the
script does a two-phase pass by default: head-only against a precomputed
FixedWindowEmbedder for --head-only-steps first, then end-to-end through
DNABERT-2 wrapped in a BatchedWindowEmbedder for --steps. Set --head-only-steps
0 to skip the warmup phase.

  # 1. Frozen-model, cached embeddings, linear head
  python scripts/demo_train.py \\
      --head linear \\
      --cache checkpoints/v1/sativas413_embeddings.ckpt.pt \\
      --steps 2000 --batch-size 32 --lr 1e-3

  # 2. Frozen-model, cached embeddings, attention head (with positional encoding)
  python scripts/demo_train.py \\
      --head attention --positional \\
      --cache checkpoints/v2/sativas413_embeddings.ckpt.pt \\
      --steps 2000 --batch-size 32 --lr 1e-3

  # 3. Trainable model (end-to-end), linear head — head warmup, then full
  python scripts/demo_train.py \\
      --head linear \\
      --head-only-steps 500 --head-only-lr 1e-3 \\
      --steps 500 --lr 1e-5 \\
      --batch-size 8 --precision bf16

  # 4. Trainable model (end-to-end), attention head with positional — same two-phase
  python scripts/demo_train.py \\
      --head attention --positional \\
      --head-only-steps 500 --head-only-lr 1e-3 \\
      --steps 500 --lr 1e-5 \\
      --batch-size 8 --precision bf16


python scripts/demo_train.py \
    --head linear \
    --cache checkpoints/v2/sativas413_embeddings.ckpt.pt \
    --steps 2000 --batch-size 32 --lr 1e-3 \
    --weight-decay 1e-2


Outputs (head state dict, embedder state dict, args, trait names) are saved
under {output-dir}/{run-name}/model.pt.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# crop_embed and DNABERT2_modules live one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed import (
    AttentionHead,
    BatchedWindowEmbedder,
    FixedWindowEmbedder,
    LinearHead,
    MLPHead,
    SampleDataset,
    SNPWindowPartitioner,
    UniqueWindowDataset,
    WindowEmbedder,
    train,
    window_position_features,
)
from crop_embed.data.coords import FASTA_PATH
from crop_embed.data.preprocessing import (
    DEFAULT_PHENO_PATH,
    align_targets_to_dataset,
    load_phenotypes,
    scale_phenotypes,
)
from crop_embed.data.vcf import load_snps_from_vcf
from DNABERT2_modules import load_dnabert2

# ── CLI ───────────────────────────────────────────────────────────────────────

DEFAULT_VCF_PATH = "/home/andrew.dickson/rice_data/sativas413_msu7_final.vcf"
DEFAULT_MODEL    = "zhihan1996/DNABERT-2-117M"

parser = argparse.ArgumentParser(description="Train a head on DNA window embeddings.")

# Embedder selection: cached (frozen) vs trainable
parser.add_argument("--cache", type=str, default=None,
                    help="Path to a fill_embedding_table checkpoint. If set, "
                         "the embedder is frozen and only the head trains.")

parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL,
                    help="HuggingFace repo or local dir for the DNA encoder "
                         "(only used when --cache is not given).")

# Head selection
parser.add_argument("--head", choices=["linear", "mlp", "attention"], required=True)
parser.add_argument("--positional", action="store_true",
                    help="Add learnable per-chrom + sinusoidal position embeddings "
                         "to the windows before the head. Attention head only.")
parser.add_argument("--n-queries",  type=int, default=8)
parser.add_argument("--n-heads",    type=int, default=8)
parser.add_argument("--hidden-dim", type=int, default=None,
                    help="MLP hidden width for --head mlp and --head attention. "
                         "Defaults to emb_dim.")
parser.add_argument("--mlp-dropout", type=float, default=0.0,
                    help="Dropout between MLP layers (--head mlp only).")

# Data paths and windowing
parser.add_argument("--vcf-path",   type=str, default=DEFAULT_VCF_PATH)
parser.add_argument("--fasta-path", type=str, default=FASTA_PATH)
parser.add_argument("--pheno-path", type=str, default=str(DEFAULT_PHENO_PATH))
parser.add_argument("--half-window", type=int, default=500)
parser.add_argument("--buffer",      type=int, default=0)
parser.add_argument("--max-length",  type=int, default=2048,
                    help="Tokenizer truncation length (trainable embedder only).")

# Training hyperparameters
parser.add_argument("--steps",      type=int,   default=1000,
                    help="Main training steps. With a trainable embedder this "
                         "is the end-to-end phase; with --cache it's the only phase.")
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 weight decay applied only to head Linear weights "
                         "(biases and embedder params are unregularized).")
parser.add_argument("--head-only-steps", type=int, default=500,
                    help="Head-only warmup steps before unfreezing the embedder. "
                         "Ignored when --cache is set (embedder has no params). "
                         "Set to 0 to skip the warmup phase entirely.")
parser.add_argument("--head-only-lr", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int,   default=32)
parser.add_argument("--precision",  choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="Mixed precision. fp16 uses GradScaler; bf16 doesn't need one.")
parser.add_argument("--log-every",  type=int,   default=50)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--test-split", type=float, default=0.30,
                    help="Fraction of samples held out for test evaluation (default 0.30). "
                         "Split is rebuilt deterministically from --seed every run.")

# I/O
parser.add_argument("--output-dir", type=str, default="trained_heads")
parser.add_argument("--run-name",   type=str, default=None,
                    help="Subdirectory under --output-dir; default is auto-named "
                         "from --head / --cache / --positional.")

# Checkpointing + wandb
parser.add_argument("--checkpoint-every", type=int, default=20,
                    help="Save a {head, embedder, optimizer} checkpoint every N steps.")
parser.add_argument("--no-checkpoint", action="store_true",
                    help="Disable periodic checkpointing.")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint (.ckpt.pt) to resume training. "
                         "Training checkpoints carry a 'phase' key ('phase1', "
                         "'phase2', or 'cached'); the script routes to the right "
                         "train() call automatically.")
parser.add_argument("--wandb-project", type=str, default=None,
                    help="If set, log training to this wandb project.")
parser.add_argument("--wandb-name", type=str, default=None,
                    help="Optional wandb run name; defaults to wandb's auto name.")

args = parser.parse_args()
torch.manual_seed(args.seed)

if args.positional and args.head != "attention":
    parser.error("--positional only applies to --head attention")

# ── Build dataset ─────────────────────────────────────────────────────────────

print("Loading SNPs from VCF …")
snps_by_chrom, samples = load_snps_from_vcf(args.vcf_path)
n_snps = sum(len(v) for v in snps_by_chrom.values())
print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {len(samples)} samples")

print("Building SNPWindowPartitioner …")
partitioner = SNPWindowPartitioner(
    snps_by_chrom, half_window=args.half_window, buffer=args.buffer
)
stats = partitioner.snps_per_window_stats()
print(f"  {stats['n_windows']:,} windows "
      f"(mean {stats['mean']:.1f} SNPs/window)")

print("Building UniqueWindowDataset …")
dataset = UniqueWindowDataset(args.vcf_path, args.fasta_path, partitioner)
print(f"  {len(dataset):,} unique windows; {len(dataset.samples)} samples")

# ── Load and align phenotypes ─────────────────────────────────────────────────

print(f"Loading phenotypes from {args.pheno_path} …")
pheno_df, trait_cols = load_phenotypes(args.pheno_path)
Y_np = align_targets_to_dataset(dataset, pheno_df, trait_cols)
Y_np = scale_phenotypes(Y_np)                          # per-trait z-score, NaN-safe
Y    = torch.tensor(Y_np, dtype=torch.float32)
n_with_pheno = (~torch.isnan(Y).all(dim=1)).sum().item()
print(f"  {Y.shape[1]} traits; {n_with_pheno}/{Y.shape[0]} samples have phenotype data")

# ── Train / test split ────────────────────────────────────────────────────────

_n_total  = len(dataset.samples)
_rng      = torch.Generator().manual_seed(args.seed)
_perm     = torch.randperm(_n_total, generator=_rng)
_n_test   = round(_n_total * args.test_split)
test_idx  = _perm[:_n_test]
train_idx = _perm[_n_test:]
print(f"Train/test split: {len(train_idx)} train / {_n_test} test  (seed={args.seed}, split={args.test_split})")

train_dataset = SampleDataset(dataset, Y, train_idx)
val_dataset   = SampleDataset(dataset, Y, test_idx)

# ── Inspect resume checkpoint (if any) to figure out which phase to enter ────

# Each train() call gets routed its own resume path. Only one of these will be
# non-None at most, since a checkpoint comes from exactly one phase.
cached_resume: str | None = None
phase1_resume: str | None = None
phase2_resume: str | None = None
skip_phase1:   bool       = False

if args.resume:
    print(f"\nInspecting resume checkpoint {args.resume} …")
    _meta = torch.load(args.resume, map_location="cpu", weights_only=False)
    saved_phase = _meta.get("phase")
    saved_step  = _meta.get("step", "?")
    if saved_phase == "cached":
        cached_resume = args.resume
        print(f"  cached-phase checkpoint (step {saved_step})")
    elif saved_phase == "phase1":
        phase1_resume = args.resume
        print(f"  phase1 checkpoint (step {saved_step}) — phase1 will resume, then phase2 runs fresh")
    elif saved_phase == "phase2":
        phase2_resume = args.resume
        skip_phase1   = True
        print(f"  phase2 checkpoint (step {saved_step}) — skipping phase1, resuming phase2")
    else:
        parser.error(f"Resume checkpoint missing/unknown 'phase' key: {saved_phase!r}")
    del _meta

# ── Build the BERT-side WindowEmbedder and/or load the cache ────────────────
#
# When --cache points to a missing file, we still need BERT to do the precompute
# (which runs after the head is built, just before training).

def _load_dnabert() -> tuple[WindowEmbedder, int]:
    print(f"\nLoading DNABERT-2 from {args.model_path} …")
    _local    = os.path.isdir(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False, local_files_only=_local,
    )
    model, _  = load_dnabert2(
        repo_id=args.model_path,
        add_pooling_layer=False,
        config_overrides={"pad_token_id": tokenizer.pad_token_id},
    )
    we = WindowEmbedder(model, tokenizer, max_length=args.max_length)
    return we, model.config.hidden_size

window_emb: WindowEmbedder | None = None
fixed:      FixedWindowEmbedder | None = None

if args.cache is None:
    window_emb, emb_dim = _load_dnabert()
    print(f"  emb_dim = {emb_dim}")
elif os.path.exists(args.cache):
    print(f"\nLoading cached embeddings from {args.cache} …")
    fixed   = FixedWindowEmbedder.from_file(args.cache, dataset)
    emb_dim = int(fixed.cache.shape[1])
    if fixed.cache.shape[0] != len(dataset.unique_fingerprints):
        parser.error(
            f"Cache has {fixed.cache.shape[0]:,} entries but current dataset has "
            f"{len(dataset.unique_fingerprints):,} unique windows. Likely a "
            f"windowing mismatch (different --half-window / --buffer / VCF). "
            f"Delete {args.cache} to regenerate."
        )
    if (fixed.sample_fp_index.shape != dataset.sample_fp_index.shape
            or not torch.equal(fixed.sample_fp_index, dataset.sample_fp_index)):
        parser.error(
            f"Cache sample→fingerprint index doesn't match the current dataset. "
            f"Delete {args.cache} to regenerate."
        )
    print(f"  {fixed.cache.shape[0]:,} cached fingerprints × {emb_dim} dims")
else:
    print(f"\nCache file {args.cache} doesn't exist — will precompute and save")
    window_emb, emb_dim = _load_dnabert()
    print(f"  emb_dim = {emb_dim}")

# ── Build head ────────────────────────────────────────────────────────────────

n_traits = Y.shape[1]
if args.head == "linear":
    head = LinearHead(emb_dim=emb_dim, n_traits=n_traits)
elif args.head == "mlp":
    head = MLPHead(
        emb_dim=emb_dim, n_traits=n_traits,
        hidden_dim=args.hidden_dim, dropout=args.mlp_dropout,
    )
else:
    head_kwargs = dict(
        emb_dim=emb_dim, n_traits=n_traits,
        n_queries=args.n_queries, n_heads=args.n_heads,
        hidden_dim=args.hidden_dim,
    )
    if args.positional:
        chroms, positions = window_position_features(dataset)
        head_kwargs.update(
            positional=True, window_chroms=chroms, window_positions=positions,
        )
    head = AttentionHead(**head_kwargs)

n_head_params = sum(p.numel() for p in head.parameters())
print(f"Head: {args.head}{' (+positional)' if args.positional else ''}  "
      f"— {n_head_params:,} params")
if window_emb is not None:
    n_emb_params = sum(p.numel() for p in window_emb.parameters())
    print(f"WindowEmbedder trainable params: {n_emb_params:,}")

# ── Resolve run name + paths (used for checkpoints, wandb, and final save) ───

if args.run_name is None:
    mode_tag = "cached" if args.cache else "e2e"
    pos_tag  = "_pos"  if args.positional else ""
    args.run_name = f"{args.head}_{mode_tag}{pos_tag}"

out_dir = Path(args.output_dir) / args.run_name
out_dir.mkdir(parents=True, exist_ok=True)

def _ckpt_path(name: str) -> str | None:
    if args.no_checkpoint:
        return None
    return str(out_dir / f"{name}.ckpt.pt")

# ── wandb (optional) ──────────────────────────────────────────────────────────

if args.wandb_project:
    try:
        import wandb
    except ImportError:
        parser.error("--wandb-project set but wandb is not installed; pip install wandb")
    wandb.init(project=args.wandb_project, name=args.wandb_name or args.run_name,
               config=vars(args))

# ── Train ─────────────────────────────────────────────────────────────────────
#
# Three execution shapes, decided by --cache and --head-only-steps:
#   --cache set                         → one train() call against FixedWindowEmbedder
#   else, --head-only-steps > 0         → phase1 (FixedWindowEmbedder) + phase2 (BatchedWindowEmbedder)
#   else                                → phase2 (BatchedWindowEmbedder) only
#
# The "current" embedder is tracked in `final_embedder` for the model.pt save.
final_embedder: torch.nn.Module

if args.cache is not None:
    # Cache file was missing at startup → precompute and save it now.
    if fixed is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        window_emb.to(device)
        print(f"\nPrecomputing {len(dataset.unique_fingerprints):,} window embeddings on {device} …")
        fixed = FixedWindowEmbedder.from_embedder(
            window_emb, dataset, batch_size=args.batch_size, device=device,
        )
        print(f"  Done — saving to {args.cache}")
        fixed.save(args.cache)
        # Free BERT memory; cached-only training doesn't need it again.
        window_emb = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTraining head against cached embeddings — {args.steps} steps at lr={args.lr}")
    train(
        fixed, head, train_dataset, val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps=args.steps,
        precision=args.precision,
        log_every=args.log_every,
        checkpoint_path=_ckpt_path("cached"),
        checkpoint_every=args.checkpoint_every,
        resume_from=cached_resume,
        phase="cached",
        trait_names=trait_cols,
    )
    final_embedder = fixed

else:
    # Phase 1: head-only warmup against a precomputed FixedWindowEmbedder.
    if args.head_only_steps > 0 and not skip_phase1:
        print(f"\nPhase 1: head-only warmup — {args.head_only_steps} steps at lr={args.head_only_lr}")
        if phase1_resume:
            print(f"  Reconstructing FixedWindowEmbedder from resume checkpoint (no re-precompute)")
            fixed = FixedWindowEmbedder.from_checkpoint(phase1_resume)
        else:
            print(f"  Precomputing window embeddings …")
            fixed = FixedWindowEmbedder.from_embedder(
                window_emb, dataset, batch_size=args.batch_size,
            )
            print(f"  Done  ({fixed.cache.shape[0]:,} × {fixed.cache.shape[1]})")
        train(
            fixed, head, train_dataset, val_dataset,
            batch_size=args.batch_size,
            lr=args.head_only_lr,
            weight_decay=args.weight_decay,
            steps=args.head_only_steps,
            precision=args.precision,
            log_every=args.log_every,
            checkpoint_path=_ckpt_path("phase1"),
            checkpoint_every=args.checkpoint_every,
            resume_from=phase1_resume,
            phase="phase1",
            trait_names=trait_cols,
        )

    # Phase 2: full end-to-end through DNABERT-2.
    print(f"\nPhase 2: end-to-end — {args.steps} steps at lr={args.lr}, precision={args.precision}")
    batched = BatchedWindowEmbedder(window_emb, dataset)
    train(
        batched, head, train_dataset, val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps=args.steps,
        precision=args.precision,
        log_every=args.log_every,
        checkpoint_path=_ckpt_path("phase2"),
        checkpoint_every=args.checkpoint_every,
        resume_from=phase2_resume,
        phase="phase2",
        trait_names=trait_cols,
    )
    final_embedder = window_emb   # save the trained backbone, not the wrapper

if args.wandb_project:
    import wandb
    wandb.finish()

# ── Save final model ──────────────────────────────────────────────────────────

out_path = out_dir / "model.pt"
torch.save({
    "head_state_dict":     head.state_dict(),
    "embedder_state_dict": final_embedder.state_dict(),
    "trait_cols":          trait_cols,
    "sample_ids":          dataset.samples,
    "args":                vars(args),
}, out_path)
print(f"\nSaved to {out_path}")
